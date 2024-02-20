import ast, argparse, sys, time, math, os
import numpy as np
import pandas as pd
from Bio import SeqIO
from io import TextIOWrapper

####### HELP FUNCTIONS #######


def read_parameters(file_loop: str, file_stacking: str) -> tuple[pd.array, pd.array]: 
    """
    Read parameters from two .csv files:
    The .csv files are converted to Pandas tables and defined as the global variable 'parameters'

    Args: 
        file_loop: csv file containing the parameters for different loop types and lengths
        file_stacking: csv file containing the parameters for base paring between different nucleotides
    """
    try:
        loops = pd.read_csv(file_loop) 
        stacking = pd.read_csv(file_stacking, index_col=0)
    except FileNotFoundError:
        raise FileNotFoundError("One or both parameter files not found")
    except pd.errors.EmptyDataError:
        raise ValueError("One or both parameter files are empty or in an unexpected file format")
    
    global parameters
    parameters = (loops, stacking)
    
    return parameters


def declare_global_variable(seq) -> None: 
    """
    Declares the global variables used troughout all the other functions. 

    Args: 
        - b_stacking: Is stacking retained for bulge loops of size 1 [True/False] (default = False)
        - closing: Is closing penalty added for GU/UG and AU/UA base pairs that closses interior loops [True/False] (default = False)
        - asymmetry: Is a penalty added for asymmetric interior loops [True/False] (default = False)
    """
    global basepairs, sequence

    basepairs = {'AU', 'UA', 'CG', 'GC', 'GU', 'UG'}
    sequence = seq



####### FOLD FUNCTIONS #######
def make_asymmetric_penalty(f: list, penalty_max: int) -> callable: 
    """
    f has to be a list. In articles writen as f(1), f(2) and so forth

    The asymmetry function is used to calculate a penalty to internal loops that are asymmetric. 
    This penalty does not exists in the orginial paper, but is added later

    This functions returns a function that uses the given parameters to calculate the penalty for asymmetric loops of given size

    args: 
        - f: list that gives the values from f(0) to f(m_max). 
        - penalty_max: that maximum penalty that can be addded to ad asymmetric interior loop
    """

    M_max = len(f)

    def asymmetry_func(i: int, ip: int, j: int, jp: int) -> float:
        """
        Calculates the penalty to add to the asymmetric interior loop enclosed by the base pairs i and j and i' and p' 
        """
        N1 = (ip-i-1)
        N2 =(j-jp-1)
        N = abs(N1-N2)
        M = min(M_max, N1, N2)-1
        penalty = min(penalty_max, N*f[M]) 
        return penalty
    
    global asymmetric_penalty_function
    asymmetric_penalty_function = asymmetry_func

    return asymmetric_penalty_function

def loop_greater_10(loop_type: str, length: int) -> float:
    """
    Calculates the energy parameters for loops with a size greater than 10 
    The parameter is calculated as described in 'Improved predictions of secondary structures for RNA'

    Args: 
        loop_type: type of loop to calculate energy for (HL, IL or BL)
        length: length of the loop
    """
    R = 0.001987 #In kcal/(mol*K)
    T = 310.15 #In K
    G_max = parameters[0].at[10, loop_type]

    G = G_max + 1.75*R*T*math.log(length/10)

    return G

### LOOP ENERGIES ###
def stacking(i: int, j: int, V: np.array) -> float: 
    """
    Find the energy parameter for basepairing of Sij and Si+1j-1, which results in basepair stacking
    If Si+1 and Sj+1 cannot basepair the energy is infinity
    Allows for Watson-Crick basepairs and wobble basepairs
    """

    prev_bp = sequence[i+1] + sequence[j-1]   
    
    #If previous bases can form a base pair stacking is possible
    if prev_bp in basepairs: 
        current_bp = sequence[i] + sequence[j]
        energy = round(parameters[1].at[current_bp, prev_bp] + V[i+1, j-1], 5)
    
    else: 
        energy = float('inf')
    
    return energy

def bulge_loop_3end(i: int, j: int, V: np.array) -> tuple[float, int]: 
    """
    Find the energy parameter of introducing a bulge loop on the 3' end of the strand 
    """
    energy = float('inf')
    ij = None

    #Try all sizes of bulge loop and save the one that gives the lowest energy
    for jp in range(i+2,j-1):  
        bp = sequence[i+1]+sequence[jp]
        if bp in basepairs: 
            size = j-jp-1
            if size <= 10:
               BL_energy = parameters[0].at[size, "BL"] + V[i+1, jp]
               if size == 1: #Add stacking parameter if stacking for bulge loops is retained
                   BL_energy += parameters[1].at[(sequence[i]+sequence[j]), bp]
            else: 
                BL_energy = loop_greater_10("BL", size) + V[i+1, jp]
            
            if BL_energy < energy: 
                energy = BL_energy
                ij = jp
    
    return round(energy, 5), ij

def bulge_loop_5end(i: int, j: int, V: np.array) -> tuple[float, int]:
    """
    Find the energy parameter of introducing a bulge loop on the 5' end of the strand 
    """
    energy = float('inf')
    ij = None
    
    #Try all sizes of bulge loop and save the one that gives the lowest energy
    for ip in range(i+2,j-1):  
        bp = sequence[ip]+sequence[j-1]
        if bp in basepairs: 
            size = ip-i-1
            if size <= 10:
                BL_energy = parameters[0].at[size, "BL"] + V[ip, j-1]
                if size == 1: #Add stacking parameter if stacking for bulge loops is retained
                    BL_energy += parameters[1].at[(sequence[i]+sequence[j]), bp] 
            else: 
                BL_energy = loop_greater_10("BL", size) + V[ip, j-1]

            if BL_energy < energy: 
                energy = BL_energy
                ij = ip

    return round(energy, 5), ij

def interior_loop(i: int, j: int, V: np.array) -> tuple[float, tuple[int, int]]: 
    """
    Find the energy parameter of adding a interior loop
    """
    
    energy = float('inf')
    ij = None

    for ip in range(i+2, j-2): #Try loop of any size between i and i'
        for jp in range(ip+3, j-1): #Try loop of any size between j and j'
            bp_prime = sequence[ip] + sequence[jp]
            if bp_prime in basepairs:
                size = (ip-i-1)+(j-jp-1)

                IL_energy = (parameters[0].at[size, "IL"] + V[ip, jp]) if size <= 10 else (loop_greater_10("IL", size) + V[ip, jp])
                
                #Add penalty to energy if loop is asymmetric
                if (ip-i-1) != (j-jp-1): 
                    IL_energy += asymmetric_penalty_function(i, ip, j, jp)

                #Add penalty if closing base pairs are AU og GU base pairs
                bp = sequence[i] + sequence[j]
                if bp in ['GU', 'UG', 'AU', 'UA']: 
                    IL_energy += 0.9
                if bp_prime in ['GU', 'UG', 'AU', 'UA']:
                    IL_energy += 0.9 
                
                #Check if energy is smaller than current min
                if IL_energy < energy: 
                    energy = IL_energy
                    ij = (ip, jp)
    
    return round(energy, 5), ij

def find_E1(i: int, j: int) -> float:
    """
    E1 are the energy of base pairing between Si and Sj with one internal edge (hairpin loop) 
    """
    size = j-i-1    

    energy = parameters[0].at[size,"HL"] if size <= 10 else loop_greater_10("HL", size)

    return round(energy, 5)

def find_E2(i: int, j: int, V: np.array) -> float: 
    """
    E2 is the energy of basepairing between i and j and i' and j' resulting in two internal edges (stacking, bulge loop or internal loop)
    i<i'<j'<j
    Returns the minimum of the 3 options  
    """
    energy = min(stacking(i, j, V), 
                 bulge_loop_3end(i, j, V)[0], 
                 bulge_loop_5end(i, j, V)[0], 
                 interior_loop(i, j, V)[0])
    return energy

def find_E3(i: int, j: int, W: np.array) -> float: 
    """
    E3 is the energy of a structure that contains more than two internal edges (bifurcating loop)
    The energy is the energy of the sum of the substructures 
    i+1<i'<j-2
    """
    energy = float('inf')
    ij = None

    #Try all combinations of substructure and save the one that gives the lowest energy
    for ip in range(i+2, j-2):  
        loop_energy = W[i+1, ip] + W[ip+1, j-1]
        if loop_energy < energy: 
            energy = round(loop_energy, 5)
            ij = (ip, ip+1)
    return energy, ij

def find_E4(i: int, j: int, W: np.array) -> tuple[float, tuple[int, int]]: 
    """
    E4 is the energy when i and j are both in base pairs, but not with each other. 
    It find the minimum of combinations of two possible subsequences containing i and j
    """
    energy = float('inf')
    ij = None

    for ip in range(i+1, j-1): 
        subsequence_energy = W[i, ip] + W[ip+1, j]
        
        if subsequence_energy < energy: 
            energy = round(subsequence_energy, 5)
            ij = (ip, ip+1)

    return energy, ij

def penta_nucleotides(W: np.array, V: np.array) -> None:
    """
    Fills out the first entries in the matrices V and W 
    The shortest possible subsequences are of length 5 and can only form hairpin loops of size 3 if i and j basepair
    """
    N = len(sequence)

    for i in range(0, N-4): 
        j = i+4
        bp = sequence[i]+sequence[j]
        if bp not in basepairs:
            V[i,j] = W[i,j ]= float('inf')
        else: 
            V[i,j] = W[i,j] = parameters[0].at[3, "HL"] 

### FILL V AND W ###
def compute_V(i: int, j: int, W: np.array, V: np.array) -> None: 
    """
    Computes the minimization over E1, E2 and E3, which will give the value at V[i,j]
    """

    if sequence[i] + sequence[j] in basepairs:
        v = min(find_E1(i, j), 
                find_E2(i, j, V), 
                find_E3(i, j, W)[0])

    else: 
        v = float('inf')

    V[i, j] = v

def compute_W(i: int, j: int, W: np.array, V: np.array) -> None:
    """
    Computes the minimization over possibilities for W and fills out the entry at W[i,j]
     Possibilities are: 
    - i or j in a structure (W[i+1, j] or W[i, j-1])
    - i and j basepair with each other (V[i,j])
    - i and j both base pair but not with each other (E4)
    """
    w = min(W[i+1,j], W[i,j-1], V[i,j], find_E4(i, j, W)[0])

    W[i,j] = w


def fold_rna() -> tuple[np.array, np.array]: 
    """
    Fills out the W and V matrices to find the fold that gives the minimum free energy
    Follows Mfold as desribed by M. Zuker

    The V matrix contains the minimum free energy for the subsequences i and j, if i and j has to form a pair. 
    If i and j are not able to basepair the energy will be infinity (not a possible structure)

    The W matrix contains the minimum free energy for the subsequences i and j where base pairing between i and j is not nessecary.
    """
    N = len(sequence)
    W, V = np.full([N, N], float('inf')), np.full([N, N], float('inf'))


    #Fills out the table with all posible penta nucleotide subsequences
    # Penta nucleotides are the base cases. If subsequences are shorter they cannot be folded
    penta_nucleotides(W, V) 

    for l in range(5, N): #Computes the best score for all subsequences that are longer than 5 nucleotides with increasing length
        for i in range(0, N-5): 
            j = i+l
            if j < N: 
                compute_V(i, j, W, V) 
                compute_W(i, j, W, V)

    return W, V

def find_optimal(W: np.array) -> float: 
    """
    Find the final energy of the folded RNA
    """
    return W[0, -1]

### BACTRACKING ### 
#TODO - CHANGE TO MAKE PAIRING OR MATRIX INSTEAD
def trace_V(i: int, j: int, W: np.array, V: np.array, dotbracket: list) -> None: 
    """
    Traces backwards trough the V matrix recursively to find the secondary structure
    """
    if V[i,j] == find_E1(i, j): 
        dotbracket[i], dotbracket[j] = '(', ')'
        for n in range(i+1, j): 
            dotbracket[n] = '.'
    
    elif V[i,j] == stacking(i, j, V): 
        dotbracket[i], dotbracket[j] = '(', ')'
        trace_V(i+1, j-1, W, V, dotbracket)
    
    elif V[i,j] == bulge_loop_3end(i, j, V)[0]: 
        jp = bulge_loop_3end(i, j, V)[1]
        dotbracket[i], dotbracket[j] = '(', ')'
        for n in range(jp, j): 
            dotbracket[n] = '.'
        trace_V(i+1, jp, W, V)
    
    elif V[i,j] == bulge_loop_5end(i, j, V)[0]: 
        ip = bulge_loop_5end(i, j, V)[1]
        dotbracket[i], dotbracket[j] = '(', ')'
        for n in range(i+1, ip): 
            dotbracket[n] = '.'
        trace_V(ip, j-1, W, V, dotbracket)
    
    elif V[i,j] == interior_loop(i, j, V)[0]:
        ij = interior_loop(i, j, V)[1]
        dotbracket[i], dotbracket[j] = '(', ')' 
        for n in range(i+1, ij[0]): 
            dotbracket[n] = '.'
        for n in range(ij[1]+1, j): 
            dotbracket[n] = '.'
        trace_V(ij[0], ij[1], W, V, dotbracket)
    
    elif V[i, j] == find_E3(i, j, W)[0]: 
        ij = find_E3(i, j, W)[1]
        dotbracket[i], dotbracket[j] = '(', ')' 
        trace_W(i+1, ij[0], W, V, dotbracket), trace_W(ij[1], j-1, W, V, dotbracket)

def trace_W(i: int, j: int, W: np.array, V: np.array, dotbracket: list) -> None: 
    """
    Traces backwards trough the W matrix recursively to find the secondary structure
    """
    if W[i,j] == W[i+1, j]: 
        dotbracket[i] = '.'
        trace_W(i+1, j, W, V, dotbracket)

    elif W[i,j] == W[i, j-1]: 
        dotbracket[j] = '.'
        trace_W(i, j-1, W, V, dotbracket)

    elif W[i, j] == V[i, j]: 
        trace_V(i, j, W, V, dotbracket)

    elif W[i,j] == find_E4(i, j, W)[0]: 
        ij = find_E4(i,j,W)[1] 
        trace_W(i, ij[0], W, V, dotbracket), trace_W(ij[1], j, W, V, dotbracket)



def backtrack(W: np.array, V: np.array) -> str: 
    """
    Backtracks trough the W, V matrices to find the final fold
    Returns the fold as a dotbracket structure
    """
    #Allocate the dot bracket structure
    dotbracket =  ['?' for x in range(W.shape[0])]
    
    j = W.shape[0]-1
    i = 0
    
    #Fill out db
    trace_W(i, j, W, V,dotbracket)

    return "".join(dotbracket)


def Mfold(sequence: str, matrix: np.array): 
    """
    """
    declare_global_variable(sequence)
    make_asymmetric_penalty() #TODO - Change! Do I want to do it this way or as part of 'declare_global_variable' and what parameters?

    #Find absolute path of script, to be able to open csv files whereever the script is called from
    script_dir = os.path.dirname(os.path.abspath(__file__))
    stacking_file = os.path.join(script_dir, 'parameters', 'stacking_1988.csv')
    loop_file = os.path.join(script_dir, 'parameters', 'loop_1989.csv')
    read_parameters(loop_file, stacking_file) #TODO - Consider whether I want this as part of 'declare_global_variable'

    W, V = fold_rna()
    fold = backtrack(W, V)

    return fold