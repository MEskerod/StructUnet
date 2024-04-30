from utils import prepare_data

import numpy as np
import pandas as pd

import time, random, sys, multiprocessing 

from utils.plots import plot_timedict


def generate_random_sequence(N: int) -> list[str]:
    """
    Generate a random RNA sequence of length N.

    Parameters:
    - N: Length of the sequence

    Returns:
    - sequence: The generated sequence
    """ 
    alphabet = ['A', 'C', 'G', 'U']
    return random.choices(alphabet, k=N)


def calculate_lengths(n: int = 81, min_length: int = 60, max_length: int = 2000) -> list[int]:
    """
    Calculate the lengths of the slices, to obtain a given number of slices of lengths between a minimum and maximum length, spaced according to a quadratic function. 

    Parameters:
    - num_slices (int): Number of slices wanted. Default is 81
    - min_length (int): Length of the shortest slice. Default is 60 
    - initial_length (int): Length of the sequence to slice from, which is also equal to the maximum length. Default is 2000

    Returns:
    - lengths (list): List of the lengths of the slices
    """
    lengths = []

    def quadratic(x): 
        a = (max_length - min_length)/n**2
        return a * x**2 + min_length

    for x in range(1, n+1): 
        lengths.append(int(quadratic(x)))
    return lengths


def time_convert(n: int, min_length: int, max_length: int, func: function) -> list[float]:
    """
    Calculate the time taken to convert N sequences of random RNA sequences of lengths between min_length and max_length using the given function.
    Returns the individual times in a list.

    Parameters:
    - n (int): Number of sequences to convert
    - min_length (int): Minimum length of the sequences
    - max_length (int): Maximum length of the sequences
    - func (function): The function to time

    Returns:
    - t (list): List of the times taken to convert each sequence
    """
    
    t = []

    lengths = calculate_lengths(n, min_length, max_length)
    for N in lengths:
        sequence = generate_random_sequence(N)
        t0 = time.time()
        func(sequence)
        t.append(time.time() - t0)
    
    return t

def average_times(func: function, func_name: str, repeats: int = 5, n: int = 81, min_length: int = 60, max_length: int = 2000) -> list[float]:
    """
    Calculates the average time taken to convert N sequences of random RNA sequences of lengths between min_length and max_length using the given function.
    The time is calculated by averaging the time taken over a number of repeats.
    The function is run in parallel using a multiprocessing Pool.

    Parameters:
    - func (function): The function to time
    - func_name (str): The name of the function
    - repeats (int): Number of times to repeat the timing. Default is 5
    - n (int): Number of sequences to convert. Default is 81
    - min_length (int): Minimum length of the sequences. Default is 60
    - max_length (int): Maximum length of the sequences. Default is 2000

    Returns:
    - average (list): List of the average times taken to convert each sequence
    """
    times = [0] * n

    pool =  multiprocessing.Pool()
    
    print(f'Processing with {func_name}', file=sys.stdout)

    args = [(n, min_length, max_length, func)] * repeats
    all_times = pool.starmap(time_convert, args)

    for rep_times in all_times:
        for i in range(n): 
            times[i] += rep_times[i]
    
    pool.close()
    pool.join()
    
    average = [t/repeats for t in times]
    
    return average[1:]


def main(): 
    functions = {"8-channel": prepare_data.make_matrix_from_sequence_8,
                 "9-channel": prepare_data.make_matrix_from_sequence_9,
                 "17-channel": prepare_data.make_matrix_from_sequence_17,}
    
    timedict = {func_name: average_times(v, func_name) for func_name, v in functions.items()}

    lengths = calculate_lengths()[1:]
    
    df = pd.DataFrame(timedict, index = lengths)
    df.to_csv('results/convert_time.csv')
    
    plot_timedict(timedict, lengths, 'figures/convert_time.png')
    return

if __name__ == '__main__': 
    main()