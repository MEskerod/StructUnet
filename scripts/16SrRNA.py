import pickle, torch, os

import pandas as pd

import tqdm as tqdm

import matplotlib.pyplot as plt

from collections import namedtuple 

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

def plot_correlation(non_std, lengths, f1s): 
    fig, ax = plt.subplots(figsize=(8, 8))

    y = [i*2/j for i, j in zip(non_std, lengths)]

    ax.scatter(y, f1s, s=10, alpha=0.5)

    ax.grid(linestyle = '--')
    ax.set_xlabel("F1 score")
    ax.set_ylabel("Proportion of bases in non-standard basepairs")

    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)

    plt.tight_layout()

    plt.savefig('figures/16SrRNA_correlation.png', dpi=300, bbox_inches='tight')




if __name__ == '__main__': 
    files = pickle.load(open('data/test.pkl', 'rb'))

    RNA = namedtuple('RNA', 'input output length family name sequence')

    df = pd.DataFrame(index = range(len(files)), columns=['Length', 'Non-standard basepairs', 'F1 score'])

    progress = tqdm.tqdm(total=len(files), unit='files')

    for i, file in enumerate(files):
        data = pickle.load(open(file, 'rb'))
        if data.family != '16S_rRNA': 
            progress.update()
            continue

        non_std = count_non_standard_bp(data.sequence, data.output)

        predicted = pickle.load(open(f'steps/RNA_Unet/{os.path.basename(file)}', 'rb'))
        _, _, F1_score = evaluate(predicted, data.output, device='cpu')

        df.loc[i] = [len(data.sequence), non_std, F1_score]

        progress.update()
    
    progress.close()

    df.to_csv('results/16SrRNA_scores.csv', index=False)
    plot_correlation(df['Non-standard basepairs'], df['Length'], df['F1 score'])
