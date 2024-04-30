import torch, sys, os, pickle, time

import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm

from collections import namedtuple


from utils.prepare_data import make_matrix_from_sequence_8
from utils.model_and_training import RNA_Unet
from utils.post_processing import prepare_input, blossom #TODO - CHANGE TO THE CHOSEN POST-PROCESSING METHOD
from utils.plots import plot_timedict


def predict(sequence: str, name: str): 
    start1 = time.time()
    input = make_matrix_from_sequence_8(sequence).to(device).unsqueeze(0)
    start2 = time.time()
    output = model(input) 
    time1 = time.time()-start1 #Time without post-processing
    time2 = time.time()-start2 #Time for only prediction
    #TODO - Update post-processing
    output = prepare_input(output.squeeze(0).squeeze(0).detach(), sequence, device)
    output = blossom(output) #TODO - CHANGE TO THE CHOSEN POST-PROCESSING METHOD
    time3 = time.time()-start2 #Total time without conversion
    time4 = time.time()-start1 #Total time
    pickle.dump(output, open(f'steps/RNA_Unet/{name}', 'wb'))
    return time1, time2, time3, time4

if __name__ == '__main__':
    device = 'gpu' if torch.cuda.is_available() else 'cpu'

    RNA = namedtuple('RNA', 'input output length family name sequence')

    print('-- Loading model and data --')
    model = RNA_Unet()
    model.load_state_dict(torch.load('RNA_Unet.pth', map_location=torch.device(device)))

    test_data = pickle.load(open('data/test.pkl', 'rb'))
    print('-- Model and data loaded --\n')

    os.makedirs('steps/RNA_Unet', exist_ok=True)
    print('-- Predicting --')
    times_wo_postprocessing = []
    times_only_predict = []
    times_wo_conversion = []
    times_total = []
    lengths = []

    progress_bar = tqdm(total=len(test_data), unit='sequence')
    
    #Predict for all sequences and save the results and times
    #Time all steps of prediction, with conversion to matrix, prediction and post-processing
    for i in range(len(test_data)):
        name = os.path.basename(test_data[i])
        sequence = pickle.load(open(test_data[i], 'rb'))
        time1, time2, time3, time4 = predict(sequence, name)
        times_wo_postprocessing.append(time1)
        times_only_predict.append(time2)
        times_wo_conversion.append(time3)
        times_total.append(time4)
        lengths.append(len(sequence))
        
        progress_bar.update(1)

    progress_bar.close()

    print('-- Predictions done --')
    print('-- Plot and save times --')
    data = {'lengths': lengths, 
            'times w/o post-processing': times_wo_postprocessing, 
            'times for only prediction': times_only_predict,
            'times w/o conversion': times_wo_conversion,
            'times total': times_total}
    
    df = pd.DataFrame(data)
    df = df.sort_values('lengths') #Sort the data by length
    df.to_csv(f'results/times_final_{device}.csv', index=False)

    data = {'times w/o post-processing': df['times w/o post-processing'].tolist(), 
            'times for only prediction': df['times for only prediction'].tolist(),
            'times w/o conversion': df['times w/o conversion'].tolist(),
            'times total': df['times total'].tolist()}
    
    plot_timedict(data, df['lengths'].tolist(), f'figures/time_final_{device}.png')


