from utils import post_processing

import torch

import pandas as pd

import time, random, sys, multiprocessing 

from utils.plots import plot_timedict


def generate_random_matrix(N: int) -> torch.Tensor: 
    """
    Generates a random matrix of size NxN.

    Parameters:
    - N: The size of the matrix

    Returns:
    - torch.Tensor: The generated matrix
    """
    return torch.rand((N, N))

def generate_random_sequence(N: int) -> list[str]:
    """
    Generates a random sequence of length N using the alphabet ['A', 'C', 'G', 'U'].

    Parameters:
    - N: The length of the sequence

    Returns:
    - list[str]: The generated sequence
    """ 
    alphabet = ['A', 'C', 'G', 'U']
    return random.choices(alphabet, k=N)


def calculate_lengths(n: int = 51, min_length: int = 60, max_length: int = 600) -> list[int]:
    """
    Calculate the lengths of the slices, to obtain a given number of slices of lengths between a minimum and maximum length, spaced according to a quadratic function. 

    Parameters:
    - num_slices: Number of slices wanted 
    - min_length: Length of the shortest slice 
    - initial_length: Length of the sequence to slice from, which is also equal to the maximum length
    
    Returns:
    - lengths: List of lengths of the slices

    """
    lengths = []

    def quadratic(x): 
        a = (max_length - min_length)/n**2
        return a * x**2 + min_length

    for x in range(1, n+1): 
        lengths.append(int(quadratic(x)))
    return lengths


def time_postprocess(func, n: int, min_length: int, max_length: int) -> list:
    """
    Procceses n number of sequences with lengths between min_length and max_length using a given function.
    The function is timed and the time it takes to process each sequence is returned.

    Parameters:
    - func: The function to time
    - n: The number of sequences to process
    - min_length: The minimum length of the sequences
    - max_length: The maximum length of the sequences

    Returns:
    - t: A list of the time it took to process each sequence
    """ 
    t = []

    lengths = calculate_lengths(n, min_length, max_length)
    for N in lengths:
        matrix = generate_random_matrix(N)
        sequence = generate_random_sequence(N)
        t0 = time.time()
        matrix = func(matrix, sequence)
        t.append(time.time() - t0)
        
    return t



def average_times(func, func_name: str, repeats: int = 5, n: int = 51, min_length: int = 60, max_length: int = 600) -> list[float]: 
    """
    Takes a function and times it for a given number of repeats and sequences of different lengths.
    The average time it takes to process each sequence is returned.
    The repeats are processed in parallel using a multiprocessing pool.

    Parameters:
    - func: The function to time
    - func_name (str): The name of the function
    - repeats (int): The number of repeats to average over
    - n (int): The number of sequences to process
    - min_length (int): The minimum length of the sequences
    - max_length (int): The maximum length of the sequences

    Returns:
    - average: A list of the average time it took to process each sequence
    """
    times = [0] * n

    pool = multiprocessing.Pool()

    print(f'Processing with {func_name}', file=sys.stdout)
    args = [(func, n, min_length, max_length)] * repeats
    all_times = pool.starmap(time_postprocess, args)

    for rep_times in all_times:
        for i in range(n): 
            times[i] += rep_times[i]
    
    pool.close() 
    pool.join()
    
    average = [t/repeats for t in times]
    
    return average[1:]

def main(): 
    functions = {'Mfold w/ matrix as constrains': post_processing.Mfold_constrain_postprocessing,
                 'Mfold w/ matrix as parameters': post_processing.Mfold_param_postprocessing, 
                 'NetworkX blossom w/ self-loops': post_processing.nx_blossum_postprocessing, 
                 'Blossom w/ self-loops': post_processing.blossom_postprocessing, 
                 'Blossom': post_processing.blossom_weak, 
                 'Argmax': post_processing.argmax_postprocessing,
                 }
    
    #Time the functions
    timedict = {func_name: average_times(v, func_name) for func_name, v in functions.items()}

    lengths = calculate_lengths()[1:]
    
    #Save and plot the results
    df = pd.DataFrame(timedict, index = lengths)
    df.to_csv('results/postprocess_time.csv')
    
    plot_timedict(timedict, lengths, 'figures/postprocess_time.png')
    

if __name__ == '__main__': 
    main()
