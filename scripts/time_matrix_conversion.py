from utils import prepare_data

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import time, random, sys 


def generate_random_sequence(N): 
    alphabet = ['A', 'C', 'G', 'U']
    return random.choices(alphabet, k=N)


def calculate_lengths(n: int = 81, min_length: int = 60, max_length: int = 1000) -> list[int]:
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


def average_times(func, func_name, repeats = 3, n = 81, min_length = 60, max_length = 1000): 
    times = [0] * n

    def time_convert():
    
        t = []

        lengths = calculate_lengths(n, min_length, max_length)
        for N in lengths:
            sequence = generate_random_sequence(N)
            t0 = time.time()
            func(sequence)
            t.append(time.time() - t0)
    
        return t
    
    print(f'Processing with {func_name}', file=sys.stdout)
    for rep in range(repeats): 
        print(f'Repeat {rep+1}/{repeats}', file=sys.stdout)
        t = time_convert()
        for i in range(n): 
            times[i] +=t[i]
    
    average = [t/repeats for t in times]
    
    return average[1:]

def plot_timedict(timedict, lengths, outputfile = None):
    colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
    
    
    fig, ax = plt.subplots(figsize = (10, 6)) 
    handles = []

    for i, func in enumerate(timedict):
        ax.scatter(x= lengths, y=timedict[func], facecolor='none', edgecolor = colors[i], s=20, linewidths = 1)
        ax.plot(lengths, timedict[func], color = colors[i], linestyle = '--', linewidth = 0.8)
        handles.append(Line2D([0], [0], color = colors[i], linestyle = '--', linewidth = 0.8, marker = 'o', markerfacecolor = 'none', markeredgecolor = colors[i], label = func))

    
    ax.legend(handles = handles, loc = 'upper left', bbox_to_anchor = (1.01, 1))
    ax.grid(linestyle = '--')
    ax.set_xlabel("Sequence length")
    ax.set_ylabel("Time (s)")

    plt.tight_layout()
    
    if outputfile: 
        plt.savefig(outputfile, dpi = 300)

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