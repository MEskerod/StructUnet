from utils import post_processing

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import time, torch 



def generate_random_matrix(N): 
    return np.random.random((N, N))


def calculate_lengths(n: int, min_length: int, max_length: int) -> list[int]:
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


def time_postprocess(func, n = 50, min_length = 60, max_length = 600):
    
    t = []

    lengths = calculate_lengths(n, min_length, max_length)
    for N in lengths:
        matrix = generate_random_matrix(N)
        t0 = time.time()
        matrix = func(matrix)
        t.append(time.time() - t0)
    
    return t

def average_times(func, repeats = 3, n = 50, min_length = 60, max_length = 600): 
    times = [0] * n

    print(f'Processing with {func}')
    for rep in range(repeats): 
        print(f'Repeat {rep+1}/{repeats}')
        t = time_postprocess(func, n, min_length, max_length)
        for i in range(n): 
            times[i] +=t[i]
    
    average = [t/repeats for t in times]
    
    return average

def plot_timedict(timedict, lengths, outputfile = None):
    colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
    
    
    fig, ax = plt.subplots() 
    handles = []

    for i, func in enumerate(timedict):
        ax.scatter(x= lengths, y=timedict[func], facecolor='none', edgecolor = colors[i], s=20, linewidths = 1)
        ax.plot(lengths, timedict[func], color = colors[i], linestyle = '--', linewidth = 0.8)
        handles.append(Line2D([0], [0], color = colors[i], linestyle = '--', linewidth = 0.8, marker = 'o', markerfacecolor = 'none', markeredgecolor = colors[i], label = func))

    
    ax.legend(handles = handles, loc = 'upper left', bbox_to_anchor = (1.01, 1))
    ax.grid(linestyle = '--')
    ax.set_xlabel("Sequence length")
    ax.set_ylabel("Time (s)")

    if outputfile: 
        plt.savefig(outputfile)
