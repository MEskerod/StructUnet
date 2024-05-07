import torch, os, pickle, time, datetime

import pandas as pd

from tqdm import tqdm

from collections import namedtuple


from utils.prepare_data import make_matrix_from_sequence_8
from utils.model_and_training import RNA_Unet
from utils.post_processing import prepare_input, blossom_weak
from utils.plots import plot_timedict

def format_time(seconds: float) -> str:
    """
    Format a time duration in seconds to hh:mm:ss format.
    
    Parameters:
    seconds: Time duration in seconds.
    
    Returns:
    Formatted time string in hh:mm:ss format.
    """
    time_delta = datetime.timedelta(seconds=seconds)
    hours, remainder = divmod(time_delta.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return '{:02d}:{:02d}:{:02d}'.format(hours, minutes, seconds)

def predict(sequence: str, name: str) -> tuple:
    start1 = time.time()
    input = make_matrix_from_sequence_8(sequence, device=device).unsqueeze(0)
    start2 = time.time()
    output = model(input) 
    time1 = time.time()-start1 #Time without post-processing
    time2 = time.time()-start2 #Time for only prediction
    output = prepare_input(output.squeeze(0).squeeze(0).detach(), sequence, device)
    output = blossom_weak(output, sequence, device)
    time3 = time.time()-start2 #Total time without conversion
    time4 = time.time()-start1 #Total time
    if device == 'cpu':
        pickle.dump(output, open(f'steps/RNA_Unet/{name}', 'wb'))
    return time1, time2, time3, time4

if __name__ == '__main__':
    device = 'gpu' if torch.cuda.is_available() else 'cpu'

    RNA = namedtuple('RNA', 'input output length family name sequence')

    print('-- Loading model and data --')
    model = RNA_Unet(channels=32)
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
    for file in test_data:
        name = os.path.basename(file)
        sequence = pickle.load(open(file, 'rb')).sequence
        time1, time2, time3, time4 = predict(sequence, name)
        times_wo_postprocessing.append(time1)
        times_only_predict.append(time2)
        times_wo_conversion.append(time3)
        times_total.append(time4)
        lengths.append(len(sequence))
        
        progress_bar.update(1)

    progress_bar.close()

    print('-- Predictions done --')
    print(f'Total time: {format_time(sum(times_total))}. Average time per sequence: {sum(times_total)/len(test_data):.5f}\n')
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


