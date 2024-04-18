import time, os
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from itertools import product

from utils.hotknots import hotknots
from utils.model_and_training import evaluate
from utils.plots import plot_timedict

def plot_f1(categories, scores, outputfile):
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

def dot_bracket_to_matrix(db):
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
      matrix = np.zeros((len(db), len(db)))
      for i, j in enumerate(bp):
            if j is not None:
                matrix[i, j] = matrix[j, i] = 1
      return matrix

def read_files(file_path):
    data = []
    
    files = os.listdir(file_path)

    for file in files:
        with open(file_path + '/' + file, 'r') as f:
            lines = f.readlines()
            data.append((int(lines[1].split()[1]), lines[3].strip().upper(), dot_bracket_to_matrix(lines[4].strip())))

    return sorted(data, key=lambda x: x[0])


def process_files(k, treshold, gap_penalty): 
    F1 = []
    times = []
    
    progess_bar = tqdm(total=len(data), desc=f'k = {k}, treshold = {treshold}, gap penaty = {gap_penalty}', unit='seq')
    
    for d in data:
        start = time.time()
        pred = hotknots(d[2], d[1], k, gap_penalty=gap_penalty, treshold_prop=treshold)
        times.append(time.time() - start)
        F1.append(evaluate(pred, d[2]))
        progess_bar.update(1)
    
    progess_bar.close()
    print()
        
    return np.mean(F1), times


if __name__ == '__main__':
    k_range = [1, 2, 5, 10, 15, 20]
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




        