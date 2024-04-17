from utils import prepare_data

import numpy as np
import pandas as pd

import time, random, sys, multiprocessing 

from utils.plots import plot_timedict


def generate_random_sequence(N): 
    alphabet = ['A', 'C', 'G', 'U']
    return random.choices(alphabet, k=N)


def calculate_lengths(n: int = 31, min_length: int = 60, max_length: int = 100) -> list[int]: #Change to 81, 60, 2000
    """
    Calculate the lengths of the slices, to obtain a given number of slices of lengths between a minimum and maximum length, spaced according to a quadratic function. 

    Args:
        - num_slices: Number of slices wanted 
        - min_length: Length of the shortest slice 
        - initial_length: Length of the sequence to slice from, which is also equal to the maximum length
    """
    lengths = []

    def quadratic(x): 
        a = (max_length - min_length)/n**2
        return a * x**2 + min_length

    for x in range(1, n+1): 
        lengths.append(int(quadratic(x)))
    return lengths


def time_convert(n, min_length, max_length, func):
    
        t = []

        lengths = calculate_lengths(n, min_length, max_length)
        for N in lengths:
            sequence = generate_random_sequence(N)
            t0 = time.time()
            func(sequence)
            t.append(time.time() - t0)
    
        return t

def average_times(func, func_name, repeats = 5, n = 81, min_length = 60, max_length = 2000): #change to 81, 60, 2000
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