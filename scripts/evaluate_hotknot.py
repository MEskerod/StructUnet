import time, os
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from itertools import product

from utils.model_and_training import evaluate
from utils.plots import plot_timedict

from utils.post_processing import hotknots_postprocessing

def plot_f1(categories: list, scores: list, outputfile: str) -> None:
    """
    Function to plot the F1 scores of different combinations of hyperparameters.
    The plot is a bar plot with the F1 scores on the y-axis and the combinations on the x-axis.
    The plot is saved as a .png file.
    
    Parameters:
    - categories (list): list of strings, the combinations of hyperparameters
    - scores (list): list of floats, the F1 scores
    - outputfile (str): string, the path to save the plot

    Returns:
    - Nones
    """
    plt.figure(figsize=(16, 6))
    plt.bar(categories, scores, color='C0', edgecolor='black', linewidth=0.5, zorder = 3)
    plt.xlabel('Combinations')
    plt.ylabel('F1 score')
    plt.xticks(rotation=90)
    plt.grid(axis='y', linestyle='--')
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(outputfile, dpi=300, bbox_inches='tight')

def dot_bracket_to_matrix(db: str) -> np.ndarray:
      """
      Function to convert a dot-bracket notation to a matrix representation.

      Parameters:
      - db (str): string, the dot-bracket notation
      
      Returns:
      - np.ndarray: the matrix representation of the dot-bracket notation
      """
      stack1 = []
      stack2 = []
      stack3 = []

      bp = [None] * len(db)

      # Append the index of the opening bracket to the stack and set the corresponding closing bracket to the current index
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
      
      #Convert the list to a matrix
      matrix = np.zeros((len(db), len(db)))
      for i, j in enumerate(bp):
            if j is not None:
                matrix[i, j] = matrix[j, i] = 1
      return matrix

def read_files(file_path: str) -> list:
    """
    Fimction to read the files in a directory containing .dbn files and return the data in a list.
    The data is a list of tuples, where each tuple contains the length of the sequence, the sequence and the matrix representation of the dot-bracket notation.

    Parameters:
    - file_path (str): the path to the directory containing the .dbn files

    Returns:
    - list: the data in a list of tuples
    """
    data = []
    
    files = os.listdir(file_path)

    for file in files:
        with open(file_path + '/' + file, 'r') as f:
            lines = f.readlines()
            data.append((int(lines[1].split()[1]), lines[3].strip().upper(), dot_bracket_to_matrix(lines[4].strip())))

    return sorted(data, key=lambda x: x[0])


def process_files(k: int, treshold: float, gap_penalty: float) -> tuple:
    """
    Function to process the data using hotknots with the given hyperparameters.
    The function returns the average F1 score and the time it took to process each sequence in 'data'.

    Parameters:
    - k (int): the depth to explore using hotknots
    - treshold (float): the treshold to use for the hotknots algorithm. Defines the proportion of SeqStr to use as tresold
    - gap_penalty (float): the gap penalty to use for the hotknots algorithm

    Returns:
    - tuple: the average F1 score and the time it took to process each sequence in 'data'
    """ 
    F1 = []
    times = []
    
    progess_bar = tqdm(total=len(data), desc=f'k = {k}, treshold = {treshold}, gap penaty = {gap_penalty}', unit='seq')
    
    for d in data:
        start = time.time()
        pred = hotknots_postprocessing(d[2], d[1], k, gap_penalty=gap_penalty, treshold_prop=treshold)
        times.append(time.time() - start)
        F1.append(evaluate(pred, d[2]))
        progess_bar.update(1)
    
    progess_bar.close()
        
    return np.mean(F1), times


if __name__ == '__main__':
    k_range = [1, 2, 3, 4, 5]
    treshold_range = [0.5, 0.8, 1]
    gap_penalty_range = [0, 0.2, 0.5]


    F1_df = pd.DataFrame(index=k_range, columns=['F1'])
    
    file_path = 'data/test_RNA_sample'

    data = read_files(file_path)

    results = []


    #Get F1 score and time for each combination
    for k, treshold, gap_penalty in product(k_range, treshold_range, gap_penalty_range):
        f1_score, times = process_files(k, treshold, gap_penalty)
        results.append((f1_score, f'k={k}, th={treshold}, gp={gap_penalty}', times))
    
    combinations = [x[1] for x in results]
    scores = [x[0] for x in results]

    #Make bar plot of F1 scores
    plot_f1(combinations, scores, 'figures/F1_hotknots.png')

    #Save scores
    F1_df = pd.DataFrame({'combinations': combinations, 'F1': scores})
    F1_df.to_csv('results/F1_hotknots.csv', index=False)

    #Find 5 best options and plot time
    results.sort(key=lambda x: x[0], reverse=True)
    time_dict = {x[1]: x[2] for x in results[:5]}
    lengths = [d[0] for d in data]
    plot_timedict(time_dict, lengths, 'figures/time_hotknots.png')

    #Save times
    time_df = pd.DataFrame(time_dict, index=lengths)
    time_df.to_csv('results/time_hotknots.csv', index=False)




        