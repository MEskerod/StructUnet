#!/usr/bin/env python3
import os, sys, pickle, time, tqdm, datetime, torch
import multiprocessing as mp
from collections import namedtuple
from signal import signal, SIGPIPE, SIG_DFL
signal(SIGPIPE,SIG_DFL) 

sys.path.pop(0)
from hotknots import hotknots as hk

def format_time(seconds: float) -> str:
    """
    Format a time duration in seconds to hh:mm:ss format.
    
    Parameters:
    seconds: Time duration in seconds.
    
    Returns:
    Formatted time string in hh:mm:ss format.
    """
    time_delta = datetime.timedelta(seconds=seconds)
    hours, remainder = divmod(time_delta.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return '{:02d}:{:02d}:{:02d}'.format(hours, minutes, seconds)


def dot_bracket_to_basepair(db: str) -> list: 
	"""
	Takes a dot-bracket notation and converts it to a list of base pairs.
	Is able to handle multiple types of brackets, which means that the function can handle pseudoknots.

	Parameters:
	- db (str): The dot-bracket notation.

	Returns:
	- list: A list of integers representing the pairing state of each base.
	"""
	stack1 = []
	stack2 = []
	stack3 = []
	bp = [i for i in range(len(db))]
       
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

def make_matrix_from_basepairs(pairs: list) -> torch.Tensor:
	"""
    Takes a list of which base each position in the sequence is paired with. If a base is unpaired pairs[i] = i.
    From the list a 2D matrix is made, with each cell coresponding to a base pair encoded as 1 and unpaired bases encoded as 1 at the diagonal

    Parameters:
    - pairs (list): A list of integers representing the pairing state of each base.

    Returns:
    - torch.Tensor: A 2D tensor with shape (len(pairs), len(pairs)).
    """

	N = len(pairs)
	matrix = torch.zeros(N, N, dtype=torch.float32)

	for i, p in enumerate(pairs):
		matrix[i, p] = 1

	return matrix

def process_file(i: int) -> float:
	"""
	Takes a index to a file list and predicts the secondary structure of the sequence in the file using hotknots.
	The result is saved in the steps/hotknots folder.

	Parameters:
	- i (int): Index to the file list.

	Returns:
	- float: The time it took to predict the secondary structure.
	""" 
	start_time = time.time()
	name = os.path.join('steps', 'hotknots', os.path.basename(files[i]))
	sequence = pickle.load(open(files[i], 'rb')).sequence
	sequence = sequence.replace('N', 'A')
	seq, mfe = hk.fold(sequence, model)
	structure = make_matrix_from_basepairs(dot_bracket_to_basepair(seq))
	pickle.dump(structure, open(name, 'wb'))
	result = time.time() - start_time
	process_bar.update(1)
	return result
            

if __name__ == '__main__':
	RNA = namedtuple('RNA', 'input output length family name sequence') #Used for complete data set

	# initialize everything first
	params = os.path.dirname(hk.__file__)

	files = pickle.load(open('data/test.pkl', 'rb'))
	indices = pickle.load(open('data/test_under_600.pkl', 'rb'))

	model = 'DP' #Defines the parameters used for the prediction

	hk.initialize(model, os.path.join(params,"parameters_DP09.txt") , os.path.join(params,"multirnafold.conf"), os.path.join(params,"pkenergy.conf") )

	os.makedirs('steps/hotknots', exist_ok=True)

	print(f"Start predicting with hotknots for {len(indices)} sequences")

	process_bar = tqdm.tqdm(total=len(indices), unit='seq', file=sys.stdout)
	
	#Process all files using multiprocessing
	with mp.Pool(mp.cpu_count()) as pool:
		times = pool.map(process_file, indices)

	process_bar.close()
	
	total_time = sum(times)
	print(f"Finished in {format_time(total_time)}. Average time per sequence: {total_time/len(indices):.5f}")


