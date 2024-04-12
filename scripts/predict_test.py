import torch, sys, os, pickle, time

import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm

from collections import namedtuple


from utils.prepare_data import make_matrix_from_sequence_8
from utils.model_and_training import RNA_Unet
from utils.post_processing import blossom #TODO - CHANGE TO THE CHOSEN POST-PROCESSING METHOD


def plot_time(time, lengths): 
    return

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
    
    for i in range(len(test_data)):
        name = os.path.basename(test_data[i])
        sequence = pickle.load(open(test_data[i], 'rb'))
        start = time.time()
        input = make_matrix_from_sequence_8(sequence)
        output = model(input)
        #TODO - Add post-processing
        times.append(time.time()-start)
        lengths.append(input.size(-1))
        pickle.dump(output, open(f'steps/RNA_Unet/{name}', 'wb'))
        progress_bar.update(1)

    progress_bar.close()

    print('-- Predictions done --')
    print('-- Plot and save times --')
    data = {'lengths': lengths, 'times': times}
    df = pd.DataFrame(data)
    df.to_csv('results/times_final.csv', index=False)

