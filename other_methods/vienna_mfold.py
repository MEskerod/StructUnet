import pickle, os, sys

import RNA as mfold
import numpy as np

from collections import namedtuple
from tqdm import tqdm

def dot_bracket_to_basepair(db): 
	stack1 = []
	stack2 = []
	stack3 = []
	bp = [None] * len(db)
       
	for i, char in enumerate(db):
		if char == '(':
			stack1.append(i)
		elif char == ')':
			j = stack1.pop()
			bp[i] = j
			bp[j] = i
		elif char == '[':
			stack2.append(i)
		elif char == ']':
			j = stack2.pop()
			bp[i] = j
			bp[j] = i
		elif char == '{':
			stack3.append(i)
		elif char == '}':
			j = stack3.pop()
			bp[i] = j
			bp[j] = i
	return bp

def make_matrix_from_basepairs(pairs: list) -> np.ndarray:
	"""
    Takes a list of all which base each position in the sequence is paired with. If a base is unpaired pairs[i] = 0.
    From the list a 2D matrix is made, with each cell coresponding to a base pair encoded as 1 and unpaired bases encoded as 1 at the diagonal

    Parameters:
    - pairs (list): A list of integers representing the pairing state of each base.

    Returns:
    - torch.Tensor: A 2D tensor with shape (len(pairs), len(pairs)).
    """

	N = len(pairs)
	matrix = np.full((N,N), 0, dtype="float32")

	for i, p in enumerate(pairs):
		if isinstance(p, int): 
			matrix[i, pairs[i]] = 1 
		else: 
			matrix[i,i] = 1

	return matrix

def predict_vienna(sequence):
	"""
	"""
	ss, mfe = mfold.fold(sequence)
	pairs = dot_bracket_to_basepair(ss)
	matrix = make_matrix_from_basepairs(pairs)
	return matrix

if __name__ == "__main__":
    RNA = namedtuple('RNA', 'input output length family name sequence')
	
    os.makedirs('steps/vienna_mfold', exist_ok=True)
    
    test = pickle.load(open('data/test.pkl', 'rb'))

    print("Predicting RNA secondary structure using ViennaRNA")
    progress_bar = tqdm(total=len(test), desc="Predicting vienna mfold", unit="file", file=sys.stdout)
    for file in test: 
        progress_bar.update(1)
        sequence = pickle.load(open(file, 'rb')).sequence
        output = predict_vienna(sequence)
        pickle.dump(output, open(os.path.join('steps/vienna_mfold', os.path.basename(file)), 'wb'))
    
    progress_bar.close()
