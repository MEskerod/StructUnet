import time, os, torch
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import multiprocessing as mp


from utils.model_and_training import evaluate
from utils.plots import plot_timedict

from utils.post_processing import hotknots_postprocessing

def plot_f1(categories: list, scores: list, mean_times: list, outputfile: str) -> None:
    """
    Function to plot the F1 scores of different ks.
    The plot is a bar plot with the F1 scores on the y-axis and the combinations on the x-axis.
    The plot is saved as a .png file.
    
    Parameters:
    - categories (list): list of strings, the combinations of hyperparameters
    - scores (list): list of floats, the F1 scores
    - mean_times (list): list of floats, the average time taken to train the model
    - outputfile (str): string, the path to save the plot

    Returns:
    - None
    """
    width = 1.8 * len(categories)
    
    plt.figure(figsize=(width, 6))
    bars = plt.bar(categories, scores, color='C0', edgecolor='black', linewidth=0.5, zorder=3)
    plt.xlabel('k')
    plt.ylabel('F1 score')
    plt.grid(axis='y', linestyle='--')
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    for bar, score, t in zip(bars, scores, mean_times):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f'F1 score: {score:.2f}\nAverage time: {t:.2f} s', ha='center', va='bottom', fontsize=7)

    plt.tight_layout()
    plt.savefig(outputfile, dpi=300, bbox_inches='tight')

def dot_bracket_to_matrix(db: str) -> torch.Tensor:
      """
      Function to convert a dot-bracket notation to a matrix representation.

      Parameters:
      - db (str): string, the dot-bracket notation
      
      Returns:
      - torch.Tensor: the matrix representation of the dot-bracket notation
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
      matrix = torch.zeros((len(db), len(db)))
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
    
    progess_bar = tqdm(total=len(data), desc=f'k = {k}, ', unit='seq')
    
    for d in data:
        start = time.time()
        pred = hotknots_postprocessing(d[2], d[1], 'cpu', k, gap_penalty=gap_penalty, treshold_prop=treshold)
        times.append(time.time() - start)
        F1.append(evaluate(pred, d[2]))
        progess_bar.update(1)
    
    progess_bar.close()
        
    return np.mean(F1), times

def process_combination(k): 
    f1_score, times = process_files(k, 0.8, 0.5)
    return f1_score, k, times


if __name__ == '__main__':
    k_range = [1, 2, 3]


    F1_df = pd.DataFrame(index=k_range, columns=['F1'])
    
    file_path = 'data/test_RNA_sample'

    data = read_files(file_path)

    #Use multiprocessing to process the combinations
    with mp.Pool(mp.cpu_count()) as pool:
        results = pool.map(process_combination, k_range)

    results.sort(key=lambda x: x[1])

    #Save times
    time_dict = {x[1]: x[2] for x in results}
    lengths = [d[0] for d in data]
    time_df = pd.DataFrame(time_dict, index=lengths)
    time_df.to_csv('results/time_hotknots.csv')

    k = [str(result[1]) for result in results]
    scores = [result[0] for result in results]
    average_times = time_df.mean(axis=0).tolist()

    #Save scores
    F1_df = pd.DataFrame({'k': k, 'F1': scores})
    F1_df.to_csv('results/F1_hotknots.csv', index=False)

    #Make bar plot of F1 scores
    plot_f1(k, scores, average_times, 'figures/F1_hotknots.png')

    





        