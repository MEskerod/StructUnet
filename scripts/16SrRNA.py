import pickle, torch, os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from collections import namedtuple
from scipy.stats import linregress 
from tqdm import tqdm

from utils.model_and_training import evaluate

def count_non_standard_bp(sequence, output): 
    pairs = torch.nonzero(output)

    standard = ['AU', 'UA', 'CG', 'GC', 'GU', 'UG', 'NU', 'NA', 'NG', 'NC', 'UN', 'AN', 'GN', 'CN']

    non_standard = 0

    for i, j in pairs:
        i, j = i.item(), j.item() 
        if i<j and sequence[i]+sequence[j] not in standard: 
            non_standard += 1
    
    return non_standard

def plot_correlation(df): 
    fig, ax = plt.subplots(figsize=(6, 5))

    x = np.array([i*2/j for i, j in zip(df['Non-standard basepairs'], df['Length'])])

    slope, intercept, r_value, p_value, std_err = linregress(x, df['F1 score'])
    r_squared = r_value**2

    ax.scatter(x, df['F1 score'], s=10, alpha=0.5)
    ax.plot(x, intercept+slope*x, color='red', label = f'$y = {slope:.2f}x + {intercept:.2f}$\n$R^2$ = {r_squared:.2f}')

    ax.grid(linestyle = '--')
    ax.set_ylabel("F1 Score")
    ax.set_xlabel("Proportion of Bases in Non-Standard Basepairs")

    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)

    plt.legend(loc='upper right', frameon=False)

    plt.tight_layout()

    plt.savefig('figures/16SrRNA_correlation.png', dpi=300, bbox_inches='tight')




if __name__ == '__main__': 
    files = pickle.load(open('data/test.pkl', 'rb'))

    RNA = namedtuple('RNA', 'input output length family name sequence')

    df = pd.DataFrame(columns=['Length', 'Non-standard basepairs', 'F1 score'])

    progress = tqdm(total=len(files), unit='files')

    for i, file in enumerate(files):
        data = pickle.load(open(file, 'rb'))
        if data.family != '16S_rRNA': 
            progress.update()
            continue

        non_std = count_non_standard_bp(data.sequence, data.output)

        predicted = pickle.load(open(f'steps/RNA_Unet/{os.path.basename(file)}', 'rb'))
        _, _, F1_score = evaluate(predicted, data.output, device='cpu')

        df.loc[len(df)] = [len(data.sequence), non_std, F1_score]

        progress.update()
    
    progress.close()

    df.to_csv('results/16SrRNA_scores.csv', index=False)
    plot_correlation(df['Non-standard basepairs'], df['Length'], df['F1 score'])
