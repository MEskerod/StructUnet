import heapq, sys, subprocess, torch

import numpy as np

from functools import cached_property
from enum import IntEnum

s_scores = {}

class HotSpot: 
    """
    Class to represent a hotspot
    """
    def __init__(self, pairs: list, energy: float) -> None:
        self.pairs = pairs
        self.energy = energy
        

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        return f'HotSpot with {len(self.pairs)} pairs and energy {self.energy}'
    
    @cached_property
    def bases(self): 
        """
        Returns the bases of the hotspot. The bases are the bases in the pairs and the single unpaired bases in the hotspot
        """
        strand1 = sorted([x[0] for x in self.pairs])
        strand2 = sorted([x[1] for x in self.pairs])
        
        return list(range(strand1[0], strand1[-1]+1)) + list(range(strand2[0], strand2[-1]+1))


class Node: 
    """
    Node class for the tree. Each node has a set of hotspots and a list of children. 
    """
    def __init__(self, hotspots = []) -> None:
        self.children = []
        self.hotspots = hotspots
        self.StrSeq = None
    
    def __str__(self) -> str:
        return f'{self.hotspots}'
    
    def __repr__(self) -> str:
        return self.__str__()
    
    @cached_property
    def bases(self): 
        """
        Returns the bases of the node, which is the union of the bases of the hotspots in the node
        """
        if self.hotspots is not None: 
            if len(self.hotspots) == 1:
                return self.hotspots[0].bases
            bases = []
            for hotspot in self.hotspots: 
                bases.extend(hotspot.bases)
            return sorted(bases)
        else:
            return []
    
    @cached_property
    def energy(self):
        """
        Returns the energy of the node, which is the sum of the energies of the hotspots in the node
        """ 
        if self.hotspots is not None: 
            return torch.sum(torch.tensor([hotspot.energy for hotspot in self.hotspots], device=dv))
        else: 
            return 0

class Tree:
    """
    Class that represents a tree. The tree has a root node and a list of nodes. The nodes are added to the tree in the order they are created
    """
    def __init__(self) -> None:
        self.root = Node()
        self.nodes = [self.root]

    def __str__(self) -> str:
        return f"Treee with {len(self)} nodes at {len(self.nodes[-1].hotspots)} level(s)"

    def __len__(self) -> int:
        return len(self.nodes)-1 #Root doesn't count as part of the length

    def print_tree(self):
        """
        Prints the tree
        """
        self._print_tree(self.root, 0)
    
    def _print_tree(self, node: Node, level: int):
        hotspot_string = ", ".join(str(x) for x in node.hotspots) if node.hotspots is not None else "root"
        print(("  " * level) + hotspot_string)
        for child in node.children:
            self._print_tree(child, level + 1)
    
    def add_node(self, parent: Node, node: Node):
        """
        Adds a node to the tree

        Parameters:
        - parent (Node): The parent node
        - node (Node): The node to add
        """
        self.nodes.append(node)
        parent.children.append(node)
        
class Trace(IntEnum):
    STOP = 0 
    LEFT = 1
    DOWN = 2
    DIAG = 3

def run_simfold(sequence: str, contraints: str) -> str:
    """
    Runs Simfold with the given sequence and constraints and returns the structure with the lowest energy
    Runs Simfold using subprocess

    Parameters:
    - sequence (str): The sequence to run Simfold on
    - contraints (str): The constraints to use for the sequence

    Returns:
    - str: The structure with the lowest energy given the constraints
    """
    simfold_dir = '../simfold/'
    command = [simfold_dir + 'simfold', '-s', sequence, '-r', contraints]

    try:
        p = subprocess.Popen(command, cwd=simfold_dir, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        p.wait()
        output, error = p.communicate()

        if p.returncode == 0: #Check that command succeeded
            result = output.decode().split()
            return result[3]
        else:
            error_msg = error.decode() if error else 'Unknown error'
            raise Exception(f'Simfold execution failed: {error_msg}')
    
    except Exception as e:
        raise Exception(f'An error occured: {e}')

def smith_waterman(sequence: str, matrix: torch.Tensor, treshold: float = 5.0) -> tuple:
    """
    Implementation of Smith-Waterman algorithm to align an RNA sequence with itself.
    The algorithm is used to find the top k local alignments in the sequence.

    Parameters:
    - sequence (str): The sequence to align
    - matrix (torch.Tensor): The matrix to use for scoring the alignment
    - treshold (float): The treshold to use for the alignment

    Returns:
    - list: A list of the top cells in the matrix
    - torch.Tensor: The tracing matrix
    """
    top_cells = []
    basepairs = {'AU', 'UA', 'CG', 'GC', 'GU', 'UG'}

    N = len(sequence)

    # Initialize the scoring matrix.
    score_matrix = torch.zeros((N+1, N+1), device=dv)
    tracing_matrix = torch.zeros((N+1, N+1), device=dv)

    #Calculating the scores for all cells in the matrix
    for l in range(4, N): 
        for i in range(N-l+1):
            j = i + l - 1

            if j > N: #Should not go there, but just in case
                break

            # It is necesary to substract one to i indices into matrix, since it doesn't have the 0 row and column
            #Calculate match score
            diagonal = score_matrix[i+1, j-1] + matrix[i, j-1] if (sequence[i] + sequence[j-1]) in basepairs else -float('inf')
            #Calculate gaps scores - should ensure that only gaps of size 1 (on one or both strands) are allowed
            vertical = score_matrix[i+1, j] - gp if (not(tracing_matrix[i+1, j] == Trace.LEFT and tracing_matrix[i+1, j-1] == Trace.DOWN)) and (not tracing_matrix[i+1, j]==Trace.DOWN) else -float('inf')
            horizontal = score_matrix[i, j-1] - gp if (not (tracing_matrix[i, j-1]==Trace.DOWN and tracing_matrix[i+1, j-1] == Trace.LEFT)) and (not tracing_matrix[i, j-1] == Trace.LEFT) else -float('inf')

            #Update the score matrix
            score_matrix[i, j] = max(0, diagonal, vertical, horizontal)

            #Fill the tracing matrix
            if score_matrix[i, j] == 0: 
                tracing_matrix[i, j] = Trace.STOP
            elif score_matrix[i, j] == diagonal: 
                tracing_matrix[i, j] = Trace.DIAG
                #Update queue of top cells
                if score_matrix[i, j] > treshold: 
                    top_cells.append((score_matrix[i, j], (i, j)))
            elif score_matrix[i, j] == vertical: 
                tracing_matrix[i, j] = Trace.DOWN
            elif score_matrix[i, j] == horizontal: 
                tracing_matrix[i, j] = Trace.LEFT
    
    # Sort the top cells list in descending order
    top_cells.sort(key=lambda x: x[0])
    
    return top_cells, tracing_matrix

def traceback_smith_waterman(trace_matrix: torch.Tensor, i: int, j: int) -> list:
    """
    Traces back the path in the matrix to find the alignment starting from cell (i, j)

    Parameters:
    - trace_matrix (torch.Tensor): The tracing matrix
    - i (int): The row index to start from
    - j (int): The column index to start from

    Returns:
    - list: A list of pairs of indices in the alignment
    """
    pairs = []
    while trace_matrix[i, j] != Trace.STOP: 
        if trace_matrix[i, j] == Trace.DIAG: 
            i += 1
            j -= 1
            pairs.append((i-1, j))
        elif trace_matrix[i, j] == Trace.DOWN: 
            i += 1
        elif trace_matrix[i, j] == Trace.LEFT: 
            j -= 1

    return pairs


def initialize_tree(sequence: str, matrix: torch.Tensor, k: int = 20) -> Tree:
    """
    Initializes the tree with the first k hotspots with are found using the Smith-Waterman algorithm

    Parameters:
    - sequence (str): The sequence to initialize the tree with
    - matrix (torch.Tensor): The matrix to use for scoring the alignment
    - k (int): The number of hotspots to initialize the tree with

    Returns:
    - Tree: The initialized tree
    """
    #Find the top hotspots using Smith-Waterman
    pq, trace = smith_waterman(sequence, matrix)

    tree = Tree()
    bases  = []

    #Backtrack trough top hotspots to find k best hotspots
    while len(tree) < k and pq:
        score, (i, j) = pq.pop()
        #If they overlap with hotspot already in the tree, skip
        if i in bases and j in bases:
            continue
        node = Node([HotSpot(traceback_smith_waterman(trace, i, j), score)])
        node.StrSeq = SeqStr(sequence, node, matrix)
        bases.extend(node.bases)
        tree.add_node(tree.root, node)
    
    return tree


def constrained_structure(bases: list, N: int) -> str: 
    """
    Makes a constrained structure of a sequence with a set of bases that are not allowed to form pairs. 
    It creates a constrainted structure of length N with '.' in the positions of the bases that are not allowed to form pairs

    Parameters:
    - bases (list): The list of bases that are not allowed to form pairs
    - N (int): The length of the sequence

    Returns:
    - str: The constrained structure
    """
    #Constrained structure of a sequence with a set of bases that are not allowed to form pairs
    structure = np.full(N, '_', dtype=str)
    structure[bases] = '.'

    return ''.join(structure)


def db_to_pairs(structure: str) -> list:
    """
    Converts a dot-bracket structure to a list of pairs of indices

    Parameters:
    - structure (str): The dot-bracket structure

    Returns:
    - list: A list of pairs of indices that forms base pairs in the structure
    """
    stack = []
    pairs = []

    for i, char in enumerate(structure):
        if char == '(':
            stack.append(i)
        elif char == ')':
            j = stack.pop()
            pairs.append((j, i))
    pairs.sort()
    return pairs

def identify_hotspots(structure: str, matrix: torch.Tensor, k: int = 20, treshold:float = 2.0) -> list:
    """
    Identifies hotspots in a structure by finding stems in the structure and calculating the energy of the stems.

    Parameters:
    - structure (str): The structure to identify hotspots in
    - matrix (torch.Tensor): The matrix to use for scoring the structure
    - k (int): The number of hotspots to return. Default is 20
    - treshold (float): The treshold to use for the hotspots. Default is 2.0

    Returns:
    - list: A list of the top k hotspots
    """
    pairs = db_to_pairs(structure)
    if not pairs:
        return []
    
    stems = []
    current_hotspot = [pairs[0]]

    for i, pair in enumerate(pairs):
        if i > 0: 
            if (pair[0] - pairs[i-1][0] <=2) and (pairs[i-1][1] - pair[1] <= 2): 
                current_hotspot.append(pair)
            else:
                stems.append(current_hotspot)
                current_hotspot = [pair]
    stems.append(current_hotspot)

    #Get energy of hotspots and eliminate those under treshold
    #Return top k hotspots
    hotspots = []
    for hotspot in stems:
        #Get energy
        energy = 0
        for i, pair in enumerate(hotspot): 
            energy += matrix[pair[0], pair[1]]
            if i > 0: 
                if pair[0] - hotspot[i-1][0] > 1: 
                    energy -= gp
                if hotspot[i-1][1] - pair[1] > 1:
                    energy -= gp
        if energy > treshold:
            hotspots.append(HotSpot(hotspot, energy))
    
    hotspots.sort(key=lambda x: x.energy, reverse=True)
            
    return hotspots

def energy_from_structure(structure: str, matrix: torch.Tensor) -> float:
    """
    Calculates the energy of a structure given a matrix with scores for the structure

    Parameters:
    - structure (str): The structure to calculate the energy of
    - matrix (torch.Tensor): The matrix to use for scoring the structure

    Returns:
    - float: The energy of the structure
    """
    energy = 0
    pairs = db_to_pairs(structure)
    for i, pair in enumerate(pairs):
        energy += matrix[pair[0], pair[1]]
        if i > 0: 
            if pair[0] - pairs[i-1][0] > 1: 
                energy -= gp+gp*0.25*(pair[0] - pairs[i-1][0] - 2) #Add penalty for every gap inserted, with a higher opening penalty
            if pairs[i-1][1] - pair[1] > 1:
                energy -= gp+gp*0.25*(pairs[i-1][1] - pair[1] - 2) #Add penalty for every gap inserted, with a higher opening penalty
    return energy

def SeqStr(S: str, H: Node, matrix: torch.Tensor, output_pairs: bool = False): 
    """
    Secondary structure of sequence S with hotspot set H
    s1, s2, ..., sl are the sequences obtained from S when removing the bases that are in hotspots of H
    Mfold/SimFold is used to obtain the energy of the segmeents 
    SeqStr is the union of the energies of the l segments and the hotspots in H

    Parameters:
    - S (str): The sequence to calculate the structure of
    - H (Node): The node with the hotspots to use for the structure
    - matrix (torch.Tensor): The matrix to use for scoring the structure
    - output_pairs (bool): If True, return the pairs of the structure. Default is False

    Returns:
    - float: The energy of the structure if output_pairs is False
    - list: The pairs of the structure if output_pairs is True
    """   
    
    if not H.bases: 
        if output_pairs:
            return db_to_pairs(run_simfold(S, '_'*len(S)))
        return energy_from_structure(run_simfold(S, '_'*len(S)), matrix)
    
    s_list = [] if H.bases[0] == 0 else [(0, H.bases[0]-1)]

    for i in range(1, len(H.bases)):
        if H.bases[i] - H.bases[i-1] > 1:
            s_list.append((H.bases[i-1]+1, H.bases[i]-1))
    
    if H.bases[-1] != len(S)-1:
        s_list.append((H.bases[-1]+1, len(S)-1))

    if output_pairs:
        pairs = []
        for pair in s_list:
            pairs.extend(db_to_pairs(run_simfold(S[pair[0]:pair[1]+1], '_'*(pair[1]-pair[0]+1))))
        for hotspot in H.hotspots:
            pairs.extend(hotspot.pairs)
        return pairs
    
    for pair in s_list: 
        if pair in s_scores:
            continue
        s_scores[pair] = energy_from_structure(run_simfold(S[pair[0]:pair[1]+1], '_'*(pair[1]-pair[0]+1)), matrix)
    
    energies = [s_scores[pair] for pair in s_list]

    return H.energy + torch.sum(torch.tensor(energies, device=dv))

def grow_tree(tree: Tree, sequence: str, matrix: torch.Tensor, k: int = 20) -> Tree:
    """
    Grows the tree by adding children to the nodes in the tree. The children are added based on the hotspots in the nodes
    Follows the HotKnots algorithm described in the paper "HotKnots: Heuristic prediction of RNA secondary structures including pseudoknots" by Ren, Rastegari, Condon and Hoos

    Parameters:
    - tree (Tree): The tree to grow
    - sequence (str): The sequence to grow the tree with
    - matrix (torch.Tensor): The matrix to use for scoring the structure
    - k (int): The number of children to add to each node. Default is 20

    Returns:
    - Tree: The grown tree
    """
    N = len(sequence)
    L = [node.hotspots[0] for node in tree.root.children]

    treshold = SeqStr(sequence, tree.root, matrix)*tp
    tree.root.StrSeq = treshold

    def build_node(node: Node, L): 
        #Use constraints and SimFold to obtain the structure of the sequence
        
        L = L + identify_hotspots(run_simfold(sequence, constrained_structure(node.bases, N)), matrix, k)
        
        children = []
        for i, hotspot in enumerate(L): 
            #Remove hotspot from list if it overlaps with hotspots in the node
            if any([x in hotspot.bases for x in node.bases]):
                L.pop(i)
                continue
            
            #Create new node with hotspotset and calculate its StrSeq
            #If energy above treshold add to children
            new_node = Node(node.hotspots + [hotspot])
            new_node.StrSeq = SeqStr(sequence, new_node, matrix)
            if new_node.StrSeq >= treshold:
                children.append(new_node)
        
        #Find k best children and add them to the tree
        children.sort(key=lambda x: x.StrSeq, reverse=True)
        for child in children[:k]:
            tree.add_node(node, child)

        #Grow tree based on good hotspots
        if node.children:
            for child in node.children: 
                build_node(child, L)
 
    #Build nodes for all children of the curret node
    for node in tree.root.children: 
        build_node(node, L)

    return tree


def hotknots(matrix: torch.Tensor, sequence: str, device: str, k: int = 20, gap_penalty: float = 0.5, treshold_prop: float = 0.5) -> list:
    """
    HotKnots algorithm to predict RNA secondary structure with pseudoknots
    It takes a matrix with scores returned from a neural network and a sequence and returns the pairs of the structure

    Parameters:
    - matrix (torch.Tensor): The matrix to use for scoring the structure
    - sequence (str): The sequence to predict the structure of
    - device (str): The device to use for the matrix
    - k (int): The number of children to add to each node. Default is 20
    - gap_penalty (float): The penalty for gaps in the structure. Default is 0.5
    - treshold_prop (float): The porportion of the naive structure to use as treshold for adding new hotspots. Default is 0.5

    Returns:
    - list: The pairs of the structure
    """
    global gp, tp, dv
    gp = gap_penalty
    tp = treshold_prop
    dv = device

    tree = initialize_tree(sequence, matrix, k)
    grow_tree(tree, sequence, matrix, k)

    best = sorted(tree.nodes, key=lambda x: x.StrSeq, reverse=True)[0]

    pairs = SeqStr(sequence, best, matrix, output_pairs=True)
  
    return pairs
