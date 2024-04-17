import heapq, sys, subprocess

import numpy as np

from functools import cached_property
from enum import IntEnum

s_scores = {}

class HotSpot: 
    def __init__(self, pairs, energy) -> None:
        self.pairs = pairs
        self.energy = energy
        

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        return f'HotSpot with {len(self.pairs)} pairs and energy {self.energy}'
    
    @cached_property
    def bases(self): 
        #strand1 = sorted([x[0] for x in self.pairs])
        #strand2 = sorted([x[1] for x in self.pairs])

        strand1 = np.sort(np.array([x[0] for x in self.pairs]))
        strand2 = np.sort(np.array([x[1] for x in self.pairs]))

        return np.concatenate([np.arange(strand1[0], strand1[-1]+1), np.arange(strand2[0], strand2[-1]+1)])
        
        #return list(range(strand1[0], strand1[-1]+1)) + list(range(strand2[0], strand2[-1]+1))


class Node: 
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
        if self.hotspots is not None: 
            bases = np.concatenate([hotspot.bases for hotspot in self.hotspots])
            #for hotspot in self.hotspots: 
            #    bases.extend(hotspot.bases)
            return np.sort(bases)
        else:
            return np.array([])
    
    @cached_property
    def energy(self): 
        if self.hotspots is not None: 
            return np.sum([hotspot.energy for hotspot in self.hotspots])
        else: 
            return 0

class Tree:
    def __init__(self) -> None:
        self.root = Node()
        self.nodes = [self.root]

    def __str__(self) -> str:
        return f"Treee with {len(self)} nodes at {len(self.nodes[-1].hotspots)} level(s)"

    def __len__(self) -> int:
        return len(self.nodes)

    def print_tree(self):
        self._print_tree(self.root, 0)
    
    def _print_tree(self, node, level):
        hotspot_string = ", ".join(str(x) for x in node.hotspots) if node.hotspots is not None else "root"
        print(("  " * level) + hotspot_string)
        for child in node.children:
            self._print_tree(child, level + 1)
    
    def add_node(self, parent: Node, node: Node):
        self.nodes.append(node)
        parent.children.append(node)
        
class Trace(IntEnum):
    STOP = 0
    LEFT = 1
    DOWN = 2
    DIAG = 3

def run_simfold(sequence, contraints):
    """
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

def smith_waterman(seq1, matrix, treshold = 2.0):
    """
    """
    top_cells = []
    basepairs = {'AU', 'UA', 'CG', 'GC', 'GU', 'UG'}

    N = len(seq1)

    # Initialize the scoring matrix.
    score_matrix = np.zeros((N+1, N+1))
    tracing_matrix = np.zeros((N+1, N+1))

    #Calculating the scores for all cells in the matrix
    for l in range(4, N): 
        for i in range(N-l+1):
            j = i + l - 1

            if j > N: #Should not go there, but just in case
                break

            # It is necesary to substract one to i indices into matrix, since it doesn't have the 0 row and column
            #Calculate match score
            diagonal = score_matrix[i+1, j-1] + matrix[i, j-1] if (seq1[i] + seq1[j-1]) in basepairs else -float('inf')
            #Calculate gaps scores - should ensure that only gaps of size 1 (on one or both strands) are allowed
            vertical = score_matrix[i+1, j] - 0.5 if (not(tracing_matrix[i+1, j] == Trace.LEFT and tracing_matrix[i+1, j-1] == Trace.DOWN)) and (not tracing_matrix[i+1, j]==Trace.DOWN) else -float('inf')
            horizontal = score_matrix[i, j-1] - 0.5 if (not (tracing_matrix[i, j-1]==Trace.DOWN and tracing_matrix[i+1, j-1] == Trace.LEFT)) and (not tracing_matrix[i, j-1] == Trace.LEFT) else -float('inf')

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

def traceback_smith_waterman(trace_matrix, i, j):
    """
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


def initialize_tree(sequence, matrix, k=20):
    """
    """
    pq, trace = smith_waterman(sequence, matrix)

    print(pq)

    tree = Tree()
    bases  = []

    while len(tree) < k and pq:
        score, (i, j) = pq.pop()
        if i in bases and j in bases:
            continue
        node = Node([HotSpot(traceback_smith_waterman(trace, i, j), score)])
        node.StrSeq = SeqStr(sequence, node, matrix)
        bases.extend(node.bases)
        tree.add_node(tree.root, node)
    return tree

def constrained_structure(bases, N): 
    """
    """
    #Constrained structure of a sequence with a set of bases that are not allowed to form pairs
    structure = np.full(N, '_', dtype=str)
    structure[bases] = '.'

    return ''.join(structure)


def db_to_pairs(structure: str):
    """
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

def identify_hotspots(structure: str, matrix: np.ndarray, k=20, treshold = 2.0):
    
    pairs = db_to_pairs(structure)
    
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
    #TODO - Maybe add min length for hotspots
    hotspots = []
    for hotspot in stems:
        #Get energy
        energy = 0
        for i, pair in enumerate(hotspot): 
            energy += matrix[pair[0], pair[1]]
            if i > 0: 
                if pair[0] - hotspot[i-1][0] > 1: 
                    energy -= 0.5
                if hotspot[i-1][1] - pair[1] > 1:
                    energy -= 0.5
        if energy > treshold:
            hotspots.append(HotSpot(hotspot, energy))
    
    hotspots.sort(key=lambda x: x.energy, reverse=True)
            
    return hotspots[:k]

def energy_from_structure(structure, matrix):
    """
    """
    energy = 0
    for i, j in db_to_pairs(structure):
        energy += matrix[i, j]
    return energy

def SeqStr(S: str, H: Node, matrix: np.ndarray, output_pairs = False): 
    """
    Secondary structure of sequence S with hotspot set H
    s1, s2, ..., sl are the sequences obtained from S when removing the bases that are in hotspots of H
    Mfold/SimFold is used to obtain the energy of the segmeents 
    SeqStr is the union of the energies of the l segments and the hotspots in H
    """
    if not H: 
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

    return H.energy + np.sum(energies)

def grow_tree(tree: Tree, sequence, matrix, k=20):
    """
    """
    N = len(sequence)
    L = [node.hotspots[0] for node in tree.root.children]

    treshold = SeqStr(sequence, tree.root.hotspots, matrix)
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
            if new_node.StrSeq > treshold:
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


def hotknots(matrix, sequence, k=20):
    """
    """
    tree = initialize_tree(sequence, matrix, k)
    grow_tree(tree, sequence, matrix, k)

    structures = sorted(tree.nodes, key=lambda x: x.StrSeq, reverse=True)
    best = structures[0]
    return SeqStr(sequence, best, matrix, output_pairs=True)



sequence = "GGCCGGCAUGGUCCCAGCCUCCUCGCUGGCGCCGGCUGGGCAACAUUCCCAGGGGACCGUCCCCUGGGUAAUGGCGAAUGGGACCCA"
"...............................___..................................................___"

pairs = [[(1, 37), (2, 36), (3, 35), (4, 34), (5, 33), (6, 32), (7, 31)],
         [(10, 86), (11, 85), (12, 84), (13, 83), (14, 82), (15, 81), (16, 80)],
         [(17, 30), (18, 29), (19, 28)], 
         [(44, 73), (45, 72), (46, 71), (47, 70), (48, 68), (49, 67), (50, 66), (51, 65), (52, 64), (53, 63), (54, 62), (55, 61), (56, 60)]]

matrix = np.zeros((len(sequence), len(sequence)))
for i, j in [pair for sublist in pairs for pair in sublist]:
    matrix[i-1, j-1] = matrix[j-1, i-1] = 1

import time

print()
start_time = time.time()
final = hotknots(matrix, sequence, k=10)
print("--- %s seconds ---" % (time.time() - start_time))

print(final)
