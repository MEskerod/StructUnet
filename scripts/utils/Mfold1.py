import numpy as np

####### HELP FUNCTIONS #######

def declare_global_variable(seq: str, M: np.ndarray) -> None: 
    """
    Declares the global variables used troughout all the other functions. 

    Parameters:
    - seq (str): The RNA sequence to fold
    - M (np.ndarray): The energy matrix used to calculate the energy of the different basepairs

    Returns:
    - None
    """
    global basepairs, sequence, matrix

    basepairs = {'AU', 'UA', 'CG', 'GC', 'GU', 'UG'}
    sequence = seq
    matrix = M


### LOOP ENERGIES ###
def find_E1(i: int, j: int) -> float:
    """
    E1 are the energy of base pairing between Si and Sj with one internal edge (hairpin loop) 

    Parameters:
    - i (int): The start index of the subsequence
    - j (int): The end index of the subsequence

    Returns:
    - energy (float): The energy of the basepairing
    """
    
    energy = matrix[i, j]

    return energy

def find_E2(i: int, j: int, V: np.ndarray) -> tuple[float, tuple[int, int]]: 
    """
    E2 is the energy of basepairing between i and j and i' and j' resulting in two internal edges (stacking, bulge loop or internal loop)
    i<i'<j'<j 

    Parameters:
    - i (int): The start index of the subsequence
    - j (int): The end index of the subsequence
    - V (np.ndarray): The V matrix

    Returns:
    - energy (float): The minimum energy of the basepairing given that two internal edges are formed
    - ij (tuple[int, int]): The indices of the second edge of subsequence that gives the minimum energy
    """  
    energy = float('inf')
    ij = None

    for ip in range(i+1, j-2): 
        for jp in range(ip+3, j): 
            if (sequence[ip] + sequence[jp]) in basepairs:
                energy_loop = matrix[i, j] + V[ip, jp]
                if energy_loop < energy: 
                    energy = energy_loop
                    ij = (ip, jp) 
    return energy, ij

def find_E3(i: int, j: int, W: np.ndarray) -> tuple[float, tuple[int, int]]: 
    """
    E3 is the energy of a structure that contains more than two internal edges (bifurcating loop)
    The energy is the energy of the sum of the substructures 
    i+1<i'<j-2

    Parameters:
    - i (int): The start index of the subsequence
    - j (int): The end index of the subsequence
    - W (np.ndarray): The W matrix

    Returns:
    - energy (float): The minimum energy of the basepairing given that more than two internal edges are formed
    - ij (tuple[int, int]): The indices of the subsequence that gives the minimum energy
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

def find_E4(i: int, j: int, W: np.ndarray) -> tuple[float, tuple[int, int]]: 
    """
    E4 is the energy when i and j are both in base pairs, but not with each other. 
    It find the minimum of combinations of two possible subsequences containing i and j

    Parameters:
    - i (int): The start index of the subsequence
    - j (int): The end index of the subsequence
    - W (np.ndarray): The W matrix

    Returns:
    - energy (float): The minimum energy of the basepairing given that i and j are both in base pairs
    - ij (tuple[int, int]): The indices of the subsequence that gives the minimum energy
    """
    energy = float('inf')
    ij = None

    for ip in range(i+1, j-1): 
        subsequence_energy = W[i, ip] + W[ip+1, j]
        
        if subsequence_energy < energy: 
            energy = round(subsequence_energy, 5)
            ij = (ip, ip+1)

    return energy, ij

def penta_nucleotides(W: np.ndarray, V: np.ndarray) -> None:
    """
    Fills out the first entries in the matrices V and W 
    The shortest possible subsequences are of length 5 and can only form hairpin loops of size 3 if i and j basepair

    Parameters:
    - W (np.ndarray): The W matrix
    - V (np.ndarray): The V matrix

    Returns:
    - None
    """
    N = len(sequence)

    for i in range(0, N-4): 
        j = i+4
        bp = sequence[i]+sequence[j]
        if bp not in basepairs:
            V[i,j] = W[i,j ]= float('inf')
        else: 
            V[i,j] = W[i,j] = matrix[i, j] 

### FILL V AND W ###
def compute_V(i: int, j: int, W: np.ndarray, V: np.ndarray) -> None: 
    """
    Computes the minimization over E1, E2 and E3, which will give the value at V[i,j]

    Parameters:
    - i (int): The start index of the subsequence
    - j (int): The end index of the subsequence
    - W (np.ndarray): The W matrix
    - V (np.ndarray): The V matrix

    Returns:
    - None
    """

    if sequence[i] + sequence[j] in basepairs:
        v = min(find_E1(i, j), 
                find_E2(i, j, V)[0], 
                find_E3(i, j, W)[0])

    else: 
        v = float('inf')

    V[i, j] = v

def compute_W(i: int, j: int, W: np.ndarray, V: np.ndarray) -> None:
    """
    Computes the minimization over possibilities for W and fills out the entry at W[i,j]
     Possibilities are: 
    - i or j in a structure (W[i+1, j] or W[i, j-1])
    - i and j basepair with each other (V[i,j])
    - i and j both base pair but not with each other (E4)

    Parameters:
    - i (int): The start index of the subsequence
    - j (int): The end index of the subsequence
    - W (np.ndarray): The W matrix
    - V (np.ndarray): The V matrix

    Returns:
    - None
    """
    w = min(W[i+1,j], W[i,j-1], V[i,j], find_E4(i, j, W)[0])

    W[i,j] = w


def fold_rna() -> tuple[np.ndarray, np.ndarray]: 
    """
    Fills out the W and V matrices to find the fold that gives the minimum free energy
    Follows Mfold as desribed by M. Zuker

    The V matrix contains the minimum free energy for the subsequences i and j, if i and j has to form a pair. 
    If i and j are not able to basepair the energy will be infinity (not a possible structure)

    The W matrix contains the minimum free energy for the subsequences i and j where base pairing between i and j is not nessecary.

    The floats in the matrix M is used as energy parameters for pairing of the different nucleotides.

    Returns:
    - W (np.ndarray): The W matrix
    - V (np.ndarray): The V matrix
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


### BACTRACKING ### 

def backtrack(W: np.ndarray, V: np.ndarray) -> list: 
    """
    Backtracks trough the W, V matrices to find the final fold
    Returns the fold as a list of tuples containing the indices of the basepairs

    Parameters:
    - W (np.ndarray): The W matrix
    - V (np.ndarray): The V matrix

    Returns:
    - pairs (list): The secondary structure of the RNA
    """
    pairs = []

    N = W.shape[0]-1
    
    j = W.shape[0]-1
    i = 0

    def trace_V(i: int, j: int) -> None: 
        """
        Traces backwards trough the V matrix recursively to find the secondary structure
        """
        if V[i,j] == find_E1(i, j): 
            pairs.append((i, j))
        
        elif V[i,j] == find_E2(i, j, V)[0]:
            ij = find_E2(i, j, V)[1]
            pairs.append((i, j))
            trace_V(ij[0], ij[1])
    
        elif V[i, j] == find_E3(i, j, W)[0]: 
            ij = find_E3(i, j, W)[1]
            pairs.append((i, j))
            trace_W(i+1, ij[0]), trace_W(ij[1], j-1)

    def trace_W(i: int, j: int) -> None: 
        """
        Traces backwards trough the W matrix recursively to find the secondary structure
        """
        if i < N and W[i,j] == W[i+1, j]: 
            trace_W(i+1, j)

        elif j > 0 and W[i,j] == W[i, j-1]: 
            trace_W(i, j-1)

        elif W[i, j] == V[i, j]: 
            trace_V(i, j)

        elif W[i,j] == find_E4(i, j, W)[0]: 
            ij = find_E4(i,j,W)[1] 
            trace_W(i, ij[0]), trace_W(ij[1], j)
    
    #Fill out the pairs list with the secondary structure
    trace_W(i, j)

    return pairs


def Mfold(sequence: str, matrix: np.ndarray) -> list: 
    """
    Folds the RNA sequence to find the secondary structure with the minimum free energy
    Uses the matrix as energy parameters for the different basepairs

    Parameters:
    - sequence (str): The RNA sequence to fold
    - matrix (np.ndarray): The energy matrix used to calculate the energy of the different basepairs

    Returns:
    - fold (np.ndarray): The secondary structure of the RNA
    """
    declare_global_variable(sequence, matrix)

    W, V = fold_rna()
    fold = backtrack(W, V)

    return fold