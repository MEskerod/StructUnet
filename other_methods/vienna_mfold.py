import pickle, os, sys, torch

import RNA as mfold

from collections import namedtuple
from tqdm import tqdm

def dot_bracket_to_basepair(db: str) -> list: 
	"""
	Takes a pseudoknot-free dot-bracket notation and returns a list of base pairs.

	Parameters:
	- db (str): A string of dot-bracket notation.

	Returns:
	- list: A list of integers representing the pairing state of each base.
	"""
	stack1 = []
	bp = [i for i in range(len(db))]
       
	for i, char in enumerate(db):
		if char == '(':
			stack1.append(i)
		elif char == ')':
			j = stack1.pop()
			bp[i] = j
			bp[j] = i
	return bp

def make_matrix_from_basepairs(pairs: list) -> torch.Tensor:
	"""
    Takes a list of all which base each position in the sequence is paired with. If a base is unpaired pairs[i] = i.
    From the list a 2D matrix is made, with each cell coresponding to a base pair encoded as 1 and unpaired bases encoded as 1 at the diagonal

    Parameters:
    - pairs (list): A list of integers representing the pairing state of each base.

    Returns:
    - torch.Tensor: A 2D tensor with shape (len(pairs), len(pairs)).
    """

	N = len(pairs)
	matrix = torch.zeros((N,N), dtype=torch.float32)

	for i, p in enumerate(pairs):
		matrix[i, p] = 1

	return matrix

def predict_vienna(sequence: str) -> torch.Tensor:
	"""
	Predicts the RNA secondary structure using the ViennaRNA package.
	The structure is returned as a 2D tensor with shape (len(sequence), len(sequence)).

	Parameters:
	- sequence (str): The RNA sequence.

	Returns:
	- torch.Tensor: A 2D tensor with shape (len(sequence), len(sequence)).
	"""
	ss, _ = mfold.fold(sequence)
	pairs = dot_bracket_to_basepair(ss)
	matrix = make_matrix_from_basepairs(pairs)
	return matrix

if __name__ == "__main__":
	data_set = sys.argv[1]

	print(f"Predicting RNA secondary structure using ViennaRNA with {data_set}")

	if data_set == 'RNAStrAlign':
		data_path = 'data/test.pkl'
		output_path = 'steps/viennaRNA'
	
	elif data_set == 'ArchiveII':
		data_path = 'data/archiveii.pkl'
		output_path = 'steps/viennaRNA_archive'

	RNA = namedtuple('RNA', 'input output length family name sequence')
	
	os.makedirs(output_path, exist_ok=True)
    
	test = pickle.load(open(data_path, 'rb'))

	print("Predicting RNA secondary structure using ViennaRNA")
	progress_bar = tqdm(total=len(test), desc="Predicting vienna mfold", unit="file", file=sys.stdout)
	for file in test: 
		progress_bar.update(1)
		sequence = pickle.load(open(file, 'rb')).sequence
		output = predict_vienna(sequence)
		pickle.dump(output, open(os.path.join(output_path, os.path.basename(file)), 'wb'))
    
	progress_bar.close()
