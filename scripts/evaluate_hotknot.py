import time
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from utils.hotknots import hotknots
from utils.model_and_training import evaluate
from utils.plots import plot_timedict

def process_files(k):
    times, F1 = np.zeros((len(files))), np.zeros((len(files)))
    
    for file in files:
        start_time = time.time()
        #Add code
        times.append(time.time() - start_time)
        

    return times, np.mean(F1)


def plot_f1(f1, k_range, outputfile): 
    plt.figure()
    plt.scatter(k_range, f1, facecolor='none', edgecolor = 'C0', s=20, linewidths = 1)
    plt.plot(k_range, f1, linestyle = '--', linewidth = 0.8)
    plt.xlabel('k')
    plt.ylabel('mean F1')
    plt.grid(linestyle='--')
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.tight_layout
    plt.savefig(outputfile, dpi=300, bbox_inches='tight')




if __name__ == '__main__':
    k_range = [1, 2, 3, 4, 5, 10, 15, 20]

    times = {}
    F1_df = pd.DataFrame(index=k_range, columns=['F1'])
    
    files = ''

    lengths = ['' for file in files] #FIXME - Add lengths

    for k in k_range:
        times, F1 = process_files(k)
        times[k] = times
        F1_df.loc[k] = F1
    
    time_df = pd.DataFrame(times, index=lengths)
    time_df.to_csv('results/time_hotknots.csv', index=False)
    F1_df.to_csv('results/F1_hotknots.csv', index=False)

    plot_timedict(times, lengths, 'figures/time_hotknots.png')
    plot_f1(F1_df['F1'], k_range, 'figures/F1_hotknots.png')


        