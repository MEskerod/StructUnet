#!/usr/bin/env python3
import os, sys, argparse, pickle
from collections import namedtuple
from argparse import RawTextHelpFormatter
from signal import signal, SIGPIPE, SIG_DFL
signal(SIGPIPE,SIG_DFL) 
import numpy as np

sys.path.pop(0)
from hotknots import hotknots as hk

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
	usage = '%s [-opt1, [-opt2, ...]] infile' % __file__
	parser = argparse.ArgumentParser(description='', formatter_class=RawTextHelpFormatter, usage=usage)
	parser.add_argument('infile', type=str, help='input file')
	parser.add_argument('outfile', type=str, help='where to write output')
	parser.add_argument('-m', '--model', type=str, default='DP', choices=['DP', 'CC', 'RE'], help='The model to use [DP]')
	args = parser.parse_args()

	RNA = namedtuple('RNA', 'input output length family name sequence') #Used for complete data set

	# initialize everything first
	params = os.path.dirname(hk.__file__)
	sequence = pickle.load(open(args.infile, 'rb')).sequence
	sequence = sequence.replace('N', 'A')
	print(sequence, file=sys.stdout)
	hk.initialize( args.model, os.path.join(params,"parameters_DP09.txt") , os.path.join(params,"multirnafold.conf"), os.path.join(params,"pkenergy.conf") )
	seq,mfe = hk.fold(sequence , args.model)
	print(seq, file=sys.stdout)

	pairs = dot_bracket_to_basepair(seq)
	matrix = make_matrix_from_basepairs(pairs)

	pickle.dump(matrix, open(args.outfile, 'wb'))



