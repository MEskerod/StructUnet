import torch, sys, os, pickle, time

import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm

from collections import namedtuple


from utils.prepare_data import make_matrix_from_sequence_8
from utils.model_and_training import RNA_Unet
from utils.post_processing import blossom #TODO - CHANGE TO THE CHOSEN POST-PROCESSING METHOD


def plot_time(time, lengths):
    """
    Plots the time it takes to predict the structure of a sequence using the Nussinov algorithm

    Parameters:
    - time (list): A list of floats representing the time it took to predict the structure of each sequence.
    - lengths (list): A list of integers representing the length of each sequence.
    """

    plt.scatter(lengths, time, facecolor='none', edgecolor = 'C0', s=20, linewidths = 1)
    plt.plot(lengths, time, linestyle = '--', linewidth = 0.8)
    plt.xlabel('Sequence length')
    plt.ylabel('Time (s)')
    plt.grid(linestyle='--')
    plt.tight_layout
    plt.savefig('figures/times_nussinov.png', dpi=300, bbox_inches='tight')


def predict(sequence, name): 
    input = make_matrix_from_sequence_8(sequence)
    output = model(input)
    #TODO - Update post-processing
    output = blossom(output.squeeze(0).detach().numpy())
    pickle.dump(output, open(f'steps/RNA_Unet/{name}', 'wb'))
    return output

if __name__ == '__main__':
    RNA = namedtuple('RNA', 'input output length family name sequence')

    print('-- Loading model and data --')
    model = RNA_Unet()
    model.load_state_dict(torch.load('RNA_Unet.pth')) 

    test_data = pickle.load(open('data/test.pkl', 'rb'))
    print('-- Model loaded and data --\n')

    os.makedirs('results', exist_ok=True)
    print('-- Predicting --')
    times = []
    lengths = []

    progress_bar = tqdm(total=len(test_data), unit='sequence')
    
    #Predict for all sequences and save the results and times
    #Time all steps of prediction, with conversion to matrix, prediction and post-processing
    for i in range(len(test_data)):
        name = os.path.basename(test_data[i])
        sequence = pickle.load(open(test_data[i], 'rb'))
        
        start = time.time()
        output = predict(sequence, name)
        times.append(time.time()-start) 
        lengths.append(output.size(-1))
        
        progress_bar.update(1)

    progress_bar.close()

    print('-- Predictions done --')
    print('-- Plot and save times --')
    data = {'lengths': lengths, 'times': times}
    df = pd.DataFrame(data)
    df.to_csv('results/times_final.csv', index=False)

