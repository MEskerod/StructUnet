import pickle, os, time
from collections import namedtuple
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



def pairing_score(i: int, j: int, S: np.array, sequence: str): 
    """
    Returns the score of pairing i and j. 
    If i and j can form a basepair the score is 1 + S[i+1, j-1] 
    Otherwise it is negative infinity (not possible structure)
    """
    basepairs = {'AU', 'UA', 'CG', 'GC', 'GU', 'UG'}

    if sequence[i]+sequence[j] in basepairs: 
        score = 1 + S[i+1, j-1]
    else: 
        score = float('-inf')
    
    return score

def bifurcating_score(i: int, j: int, S: np.array) -> tuple:
    """
    Tries all the possible bifurcating loops and returns the score and k that gives the maximum energy
    """
    score = float('-inf')
    end_k = 0

    for k in range(i+1, j-1): 
        sub_score = S[i, k] + S[k+1, j]
        if sub_score > score: 
            score = sub_score
            end_k = k
    return score, end_k

def fill_S(sequence: str) -> np.array:
    """
    Fills out the S matrix
    """

    N = len(sequence)
    S = np.zeros([N, N])
    
    for l in range(4, N): #Computes the best score for all subsequences that are 5 nucleotides or longer
        for i in range(0, N-4): 
            j = i+l
            if j < N:
                score = max(S[i+1, j],
                            S[i, j-1],
                            pairing_score(i, j, S, sequence),
                            bifurcating_score(i, j, S)[0])
                S[i,j] = score
                
    return S

def backtrack(i: int, j: int, S: np.array, pairs: list, sequence: str) -> None: 
    """
    Backtracks trough the S matrix, to find the structure that gives the maximum energy
    """
    if j-i-1 <= 3: 
        pairs[i], pairs[j] = j, i

    elif S[i, j] == S[i+1, j]: 
        backtrack(i+1, j, S, pairs, sequence)

    elif S[i, j] == S[i, j-1]: 
        backtrack(i, j-1, S, pairs, sequence)
    
    elif S[i, j] == pairing_score(i, j, S, sequence): 
        pairs[i], pairs[j] = j, i
        backtrack(i+1, j-1, S, pairs, sequence)

    elif S[i, j] == bifurcating_score(i, j, S)[0]:
        k = bifurcating_score(i, j, S)[1]
        backtrack(i, k, S, pairs, sequence), backtrack(k+1, j, S, pairs, sequence)

def fold_RNA(S: np.array, sequence: str) -> str: 
    """
    Finds the optimal structure of foldning the sequence and returns a list of each position of the state of pairing of each base
    """
    pairs = [None for x in range(S.shape[0])]
    
    j = S.shape[0]-1
    i = 0

    backtrack(i, j, S, pairs, sequence)

    return pairs


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

def plot_time(time, lengths):
    """
    Plots the time it takes to predict the structure of a sequence using the Nussinov algorithm

    Parameters:
    - time (list): A list of floats representing the time it took to predict the structure of each sequence.
    - lengths (list): A list of integers representing the length of each sequence.
    """

    plt.scatter(lengths, time, facecolor='none', edgecolor = 'C0', s=20, linewidths = 1)
    plt.plot(lengths, time, linestyle = '--', linewidth = 0.8)
    plt.xlabel('Sequence length')
    plt.ylabel('Time (s)')
    plt.grid(linestyle='--')
    plt.tight_layout
    plt.savefig('figures/times_nussinov.png', dpi=300, bbox_inches='tight')


def main() -> None: 
    """
    """
    
    os.makedirs('steps/nussinov', exist_ok=True)
    test = pickle.load(open('data/test.pkl', 'rb'))

    print(f"Predicting structure using Nussinov algorithm.\n Total: {len(test)} sequences.")
    print('-- Predicting --')

    times = []
    lengths = []

    progress_bar = tqdm(total=len(test), unit='sequence')

    for i in range(len(test)): 
        sequence = pickle.load(open(test[i], 'rb')).sequence
        name = os.path.join('steps', 'nussinov', os.path.basename(test[i])) 
        start = time.time()
        pairs = fold_RNA(fill_S(sequence), sequence)  
        output = make_matrix_from_basepairs(pairs)
        pickle.dump(output, open(name, 'wb'))
        times.append(time.time()-start)
        lengths.append(len(sequence))
        progress_bar.update(1)

    progress_bar.close()

    print('-- Predictions done --')

    print('-- Plot and save times --')
    data = {'lengths': lengths, 'times': times}
    df = pd.DataFrame(data)
    df.to_csv('results/times_nussinov.csv', index=False)
    plot_time(times, lengths)

   
if __name__ == '__main__': 
    RNA = namedtuple('RNA', 'input output length family name sequence')
    main()