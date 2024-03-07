import numpy as np

from functools import cached_property

class HotSpot: 
    def __init__(self, pairs, energy) -> None:
        self.pairs = pairs
        self.energy = energy
        

    def __repr__(self) -> str:
        pass

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
        return str(self.hotspots)
    
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
        



def initialize_tree(sequence, matrix, k=20):
    """
    """

    def find_initial_hotspots():
        N = len(sequence)

        for i in range(N): 
            for j in range(i+4, N): 
                pass
        return
    
    hotspots = find_initial_hotspots()
    tree = Tree()
    for hotspot in hotspots: 
        tree.add_node(tree.root, hotspot)

    return tree

def grow_tree(tree, sequence, matrix, k=20):
    """
    """
    def build_node(node): 
        return
    
    for node in tree.nodes[1]: 
        build_node(node)

    return tree

def SeqStr(S, H): 
    """
    """
    #Secondary structure of sequence S with hotspot set H
    #s1, s2, ..., sl are the sequences obtained from S when removing the bases that are in hotspots of H
    #Mfold/SimFold is used to obtain the energy of the segmeents 
    #SeqStr is the union of the energies of the l segments and the hotspots in H
    return 

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



sequence = 'GGCCGGCAUGGUCCCAGCCUCCUCGCUGGCGCCGGCUGGGCAACAUUCCGAGGGGACCGUCCCCUGGGUAAUGGCGAAUGGGACCCA'

pairs = [[(1, 37), (2, 36), (3, 35), (4, 34), (5, 33), (6, 32), (7, 31)],
         [(10, 86), (11, 85), (12, 84), (13, 83), (14, 82), (15, 81), (16, 80)],
         [(17, 30), (18, 29), (19, 28)], 
         [(44, 73), (45, 72), (46, 71), (47, 70), (48, 68), (49, 67), (50, 66), (51, 65), (52, 64), (53, 63), (54, 62), (55, 61), (56, 60)]]

matrix = np.zeros((len(sequence), len(sequence)))
for i, j in [pair for sublist in pairs for pair in sublist]:
    matrix[i-1, j-1] = matrix[j-1, i-1] = 1

tree = Tree()
for sublist in pairs: 
    tree.add_node(tree.root, [HotSpot(sublist, 0.8)])

#print(tree)
#tree.print_tree()

print(tree.nodes[1][0].bases)
