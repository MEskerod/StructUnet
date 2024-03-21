import heapq, sys

import numpy as np

from functools import cached_property
from enum import IntEnum

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
        strand1 = sorted([x[0] for x in self.pairs])
        strand2 = sorted([x[1] for x in self.pairs])

        return list(range(strand1[0], strand1[-1])) + list(range(strand2[0], strand2[-1]))



class Node: 
    def __init__(self, hotspots = None) -> None:
        self.level = None
        self.children = []
        self.hotspots = hotspots
    
    def __str__(self) -> str:
        return f'{self.hotspots}'
    
    def __repr__(self) -> str:
        return self.__str__()
    
    @cached_property
    def bases(self): 
        if self.hotspots is not None: 
            bases = []
            for hotspot in self.hotspots: 
                bases.extend(hotspot.bases)
            return sorted(bases)
        else: 
            return []
    
    @cached_property
    def energy(self): 
        if self.hotspots is not None: 
            return sum(hotspot.energy for hotspot in self.hotspots)
        else: 
            return 0

class Tree:
    def __init__(self) -> None:
        self.root = Node()
        self.root.level = 0
        self.nodes = [[self.root]]
        self.levels = 0

    def __str__(self) -> str:
        return f"Treee with {len(self)} nodes at {self.levels} level(s)"

    def __repr__(self) -> str:
        pass

    def __len__(self) -> int:
        length = 0
        for level in self.nodes:
            length += len(level)
        return length

    def __getitem__(self, key):
        pass

    def __iter__(self):
        pass

    def print_tree(self):
        self._print_tree(self.root, 0)
    
    def _print_tree(self, node, level):
        hotspot_string = ", ".join(str(x) for x in node.hotspots) if node.hotspots is not None else "root"
        print(("  " * level) + hotspot_string)
        for child in node.children:
            self._print_tree(child, level + 1)
    
    def add_node(self, parent: Node, hotspots):
        node = Node(hotspots)
        node.level = parent.level + 1

        assert node.level == len(hotspots)

        if node.level > self.levels: 
            self.levels = node.level
            self.nodes.append([])

        parent.children.append(node)
        self.nodes[node.level].append(node)
        
class Trace(IntEnum):
    STOP = 0
    LEFT = 1
    DOWN = 2
    DIAG = 3


def smith_waterman(seq1, matrix, k=10, treshold = 2.0):
    """
    """
    top_cells = []
    heapq.heapify(top_cells)
    
    basepairs = {'AU', 'UA', 'CG', 'GC', 'GU', 'UG'}

    #seq2 = seq1[::-1]
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
                    if len(top_cells) < k: 
                        heapq.heappush(top_cells, (score_matrix[i, j], (i, j)))
                    else:
                        # If heap is full, compare the smallest value with the new value
                        min_value, _ = top_cells[0]
                        if score_matrix[i, j] > min_value: 
                            heapq.heappop(top_cells) #Pop smallest value
                            heapq.heappush(top_cells, (score_matrix[i, j], (i, j)))
            elif score_matrix[i, j] == vertical: 
                tracing_matrix[i, j] = Trace.DOWN
            elif score_matrix[i, j] == horizontal: 
                tracing_matrix[i, j] = Trace.LEFT
    
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
    hotspots, trace = smith_waterman(sequence, matrix, k)
    tree = Tree()
    for score, (i, j) in hotspots: 
        tree.add_node(tree.root, [HotSpot(traceback_smith_waterman(trace, i, j), score)])

    return tree

def constrained_structure(constrained, N): 
    """
    """
    #Constrained structure of a sequence with a set of basepairs 
    structure = ['_'] * N
    for base in constrained:
        structure[base] = '.'

    return ''.join(structure)

def identify_hotspots(structure: str, matrix: np.ndarray, k=20, treshold = 2.0):
    """
    """

    #Identify hotspots that doesn't have more than one gap between opening brackets
    prelim_hotspots = []
    current_stem = [[]]
    number_hotspots = 0
    len_current_hotspot = 0
    in_hotspot = False

    for i, s in enumerate(structure):
        if s == '(':
            in_hotspot = True
            current_stem[number_hotspots].append([i])
        
        elif s == '.':
            if in_hotspot and structure[i-1] == '.' and current_stem[number_hotspots] != []:
                current_stem.append([])
                number_hotspots += 1
        
        elif s == ')': 
            in_hotspot = False

            if len_current_hotspot == 0:
                current_stem = current_stem[:-1]
                number_hotspots -= 1
                len_current_hotspot = len(current_stem[number_hotspots]) 
            
            current_stem[number_hotspots][len_current_hotspot-1].append(i)
            len_current_hotspot -= 1

            if len_current_hotspot == 0:
                number_hotspots -= 1
                len_current_hotspot = len(current_stem[number_hotspots])
                
                if number_hotspots < 0:
                    for hotspot in current_stem:
                        prelim_hotspots.append(hotspot)
                    current_stem = [[]]
                    len_current_hotspot = 0
                    in_hotspot = False
                    number_hotspots = 0

    #Identify larger gaps between closing brackets and break the hotspots
    valid_hotspots = []
    for hotspot in prelim_hotspots: 
        breaking_points = []
        for i in range(len(hotspot)-1):
            if hotspot[i][1] - hotspot[i+1][1] > 2: 
                breaking_points.append(i)
        #Split based on indices
        if breaking_points:
            prev_index = 0
            for i in breaking_points: 
                valid_hotspots.append(hotspot[prev_index:i+1])
                prev_index = i+1
            valid_hotspots.append(hotspot[prev_index:])
        else: 
            valid_hotspots.append(hotspot)
    
    #Get energy of hotspots and eliminate those under treshold
    #Return top k hotspots
    hotspots = []
    for hotspot in valid_hotspots:
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

    def sort_func(h):
        return h.energy
    
    hotspots.sort(key=sort_func, reverse=True)
            
    return hotspots[:k]

def SeqStr(S, H): 
    """
    """
    #Secondary structure of sequence S with hotspot set H
    #s1, s2, ..., sl are the sequences obtained from S when removing the bases that are in hotspots of H
    #Mfold/SimFold is used to obtain the energy of the segmeents 
    #SeqStr is the union of the energies of the l segments and the hotspots in H
    return 

def grow_tree(tree, sequence, matrix, k=20):
    """
    """
    N = len(sequence)
    L = [node.hotspots for node in tree.root.children]
    print(L)

    def build_node(node, L): 
        constraints = constrained_structure(node.bases, N)
        #Use constraints and SimFold to obtain the structure of the sequence
        struture = ''
        hotspots = identify_hotspots(structure, matrix, k)
        L = L + hotspots
        for hotspot in hotspots: 
            L = L
            #Use SeqStr to identify whether the hotspots are good or not
        #Grow tree based on good hotspots
        if node.children:
            for child in node.children: 
                build_node(child, L)
        return
 
    for node in tree.root.children: 
        build_node(node)

    return tree


def output_structure(tree):
    """
    """
    return



def hotknots(matrix, sequence, k=20):
    """
    """
    tree = initialize_tree(sequence, matrix, k)
    grow_tree(tree, sequence, matrix, k)
    pairs = output_structure(tree)
    return pairs



sequence = "GGCCGGCAUGGUCCCAGCCUCCUCGCUGGCGCCGGCUGGGCAACAUUCCCAGGGGACCGUCCCCUGGGUAAUGGCGAAUGGGACCCA"
"...............................___..................................................___"

pairs = [[(1, 37), (2, 36), (3, 35), (4, 34), (5, 33), (6, 32), (7, 31)],
         [(10, 86), (11, 85), (12, 84), (13, 83), (14, 82), (15, 81), (16, 80)],
         [(17, 30), (18, 29), (19, 28)], 
         [(44, 73), (45, 72), (46, 71), (47, 70), (48, 68), (49, 67), (50, 66), (51, 65), (52, 64), (53, 63), (54, 62), (55, 61), (56, 60)]]

matrix = np.zeros((len(sequence), len(sequence)))
for i, j in [pair for sublist in pairs for pair in sublist]:
    matrix[i-1, j-1] = matrix[j-1, i-1] = 1

tree = initialize_tree(sequence, matrix, k=20)

print(tree)
tree.print_tree()

print(tree.nodes[1][0].bases)

structure = '(((((((.........(((........))))))))))..((..((((((((((((.....)))))))).))))))............' 

print(structure)

#print(identify_hotspots(structure, matrix, k=20))


grow_tree(tree, sequence, matrix, k=20)


