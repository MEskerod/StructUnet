#!/usr/bin/env python3
import os, sys, argparse, pickle, time, tqdm, datetime
from collections import namedtuple
from argparse import RawTextHelpFormatter
from signal import signal, SIGPIPE, SIG_DFL
signal(SIGPIPE,SIG_DFL) 
import numpy as np

sys.path.pop(0)
from hotknots import hotknots as hk

def format_time(seconds):
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

            

if __name__ == '__main__':
	RNA = namedtuple('RNA', 'input output length family name sequence') #Used for complete data set

	# initialize everything first
	params = os.path.dirname(hk.__file__)

	files = pickle.load(open('data/test.pkl'))
	indices = pickle.load(open('data/test_under_600.pkl'))

	model = 'DP'

	hk.initialize(model, os.path.join(params,"parameters_DP09.txt") , os.path.join(params,"multirnafold.conf"), os.path.join(params,"pkenergy.conf") )

	print(f"Start predicting with hotknots for {len(indices)} sequences")

	process_bar = tqdm.tqdm(total=len(indices), unit='seq', file=sys.stdout)

	start_time = time.time()

	for i in indices:
		name = os.path.join('steps', 'hotknots', os.path.basename(files[i]))
		sequence = pickle.load(open(files[i].input, 'rb')).sequence
		sequence = sequence.replace('N', 'A')
		seq, mfe = hk.fold(sequence, model)
		structure = make_matrix_from_basepairs(dot_bracket_to_basepair(seq))
		pickle.dump(structure, open(name, 'wb'))
		process_bar.update(1)
	
	process_bar.close()
	
	total_time = time.time() - start_time
	print(f"Finished in {format_time(total_time)}. Average time per sequence: {total_time/len(indices):.5f}")


