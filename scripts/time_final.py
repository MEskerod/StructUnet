import time, torch, pickle

from collections import namedtuple
from tqdm import tqdm

import pandas as pd
import numpy as np

from utils.prepare_data import make_matrix_from_sequence_8
from utils.model_and_training import RNA_Unet
from utils.post_processing import prepare_input, blossom_weak
from utils.plots import plot_timedict

def predict(sequence: str, name: str) -> tuple:
    """
    Uses the model to predict the structure of a given sequence.
    Saves the result and returns the time it took for the prediction.
    The time is split into the time without post-processing, the time for only prediction, the time without conversion and the total time.

    Parameters:
    - sequence (str): The sequence to predict.
    - name (str): The name of the sequence.

    Returns:
    - tuple: The time it took for the prediction in the order (time without post-processing, time for only prediction, time without conversion, total time)
    """
    start1 = time.time()
    input = make_matrix_from_sequence_8(sequence, device=device).unsqueeze(0).to(device)
    start2 = time.time()
    output = model(input) 
    time1 = time.time()-start1 #Time without post-processing
    time2 = time.time()-start2 #Time for only prediction
    output = prepare_input(output.squeeze(0).squeeze(0).detach(), sequence, device)
    output = blossom_weak(output, sequence, device)
    time3 = time.time()-start2 #Total time without conversion
    time4 = time.time()-start1 #Total time
    return time1, time2, time3, time4


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using {device} for prediction\n')

    RNA = namedtuple('RNA', 'input output length family name sequence')

    print('-- Loading model and data --')

    model = RNA_Unet(channels=32).to(device)
    model.load_state_dict(torch.load('RNA_Unet.pth', map_location=torch.device(device)))
    model.to(device)

    test_data = pickle.load(open('data/test.pkl', 'rb'))
    print('-- Model and data loaded --\n')

    repeats = 5

    N = len(test_data)

    times_wo_postprocessing = [[]*N]
    times_only_predict = [[]*N]
    times_wo_conversion = [[]*N]
    times_total = [[]*N]
    lengths = []

    
    #Predict for all sequences and save the times
    #Time all steps of prediction, with conversion to matrix, prediction and post-processing
    for i in range(repeats):
        progress_bar = tqdm(total=len(test_data), unit='sequence', desc=f'Repeat {i+1}/{repeats}')
        for n, file in enumerate(test_data):
            sequence = pickle.load(open(file, 'rb')).sequence
            time1, time2, time3, time4 = predict(sequence)
            times_wo_postprocessing[n].append(time1)
            times_only_predict[n].append(time2)
            times_wo_conversion[n].append(time3)
            times_total[n].append(time4)
            if i == 0:
                lengths.append(len(sequence))
        
            progress_bar.update(1)

        progress_bar.close()
    
    print('-- Saving and plotting results --')
    
    times_wo_postprocessing = [np.mean(times) for times in times_wo_postprocessing]
    times_only_predict = [np.mean(times) for times in times_only_predict]
    times_wo_conversion = [np.mean(times) for times in times_wo_conversion]
    times_total = [np.mean(times) for times in times_total]

    data = {'lengths': lengths, 
            'times w/o post-processing': times_wo_postprocessing, 
            'times for only prediction': times_only_predict,
            'times w/o conversion': times_wo_conversion,
            'times total': times_total}
    
    df = pd.DataFrame(data)
    df = df.sort_values('lengths') #Sort the data by length
    df.to_csv(f'results/time_final_{device}.csv', index=False)

    data = {'times w/o post-processing': df['times w/o post-processing'].tolist(), 
            'times for only prediction': df['times for only prediction'].tolist(),
            'times w/o conversion': df['times w/o conversion'].tolist(),
            'times total': df['times total'].tolist()}
    
    plot_timedict(data, df['lengths'].tolist(), f'figures/time_final_{device}.png')
