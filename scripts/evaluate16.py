import os, sys, tempfile, shutil, tarfile, pickle, multiprocessing, time, datetime, torch

import numpy as np
import pandas as pd

import pandas as pd

from tqdm import tqdm

from collections import namedtuple

import matplotlib.pyplot as plt
import matplotlib as mpl


from utils.prepare_data import make_matrix_from_sequence_16
from utils.model_and_training import RNA_Unet, evaluate
from utils.post_processing import blossom_weak
from utils.plots import plot_timedict
from utils.plots import violin_plot



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

def predict(sequence: str, name: str, outputdir: str) -> tuple:
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
    input = make_matrix_from_sequence_16(sequence).unsqueeze(0).to(device)
    start2 = time.time()
    output = model(input).squeeze(0).squeeze(0).detach() 
    time1 = time.time()-start1 #Time without post-processing
    time2 = time.time()-start2 #Time for only prediction
    output = (output + output.T)/2 #Make the matrix symmetric before post-processing
    output = blossom_weak(output, sequence, device)
    time3 = time.time()-start2 #Total time without conversion
    time4 = time.time()-start1 #Total time
    if device == 'cpu':
        pickle.dump(output, open(f'{outputdir}/{name}', 'wb'))
    return time1, time2, time3, time4, output


def has_pk(pairings: np.ndarray) -> bool:
    """
    Checks if a given pairing has a pseudoknot.

    Parameters:
    - pairings (np.ndarray): The pairing array. Unpaired bases are represented by the index itself.

    Returns:
    - bool: True if the pairing has a pseudoknot, False otherwise.
    """
    for idx in range(len(pairings)):
        i, j = idx, pairings[idx]
        start, end = min(i, j), max(i, j)
        if i==j:
            continue
        if torch.max(pairings[start:end]) > end or torch.min(pairings[start:end]) < start:
            return True
    return False


def scores_pseudoknot(predicted: np.ndarray, target_pk: bool) -> np.ndarray:
    """
    Returns the [TN, FN, FP, TP] for pseudoknots in the predicted and target structure.

    Parameters:
    - predicted (np.ndarray): The predicted structure as a NumPy array.
    - target (np.ndarray): The target structure as a NumPy array.

    Returns:
    - np.ndarray: The [TN, FN, FP, TP] for pseudoknots in the predicted and target structure.
    """
    pk_score = np.array([0, 0, 0, 0])

    predicted = predicted.squeeze()
    i = has_pk(np.argmax(predicted, axis=1))
    j = target_pk
    pk_score[i*2+j] += 1

    return pk_score

def f1_pk_score(pk_score: np.ndarray, epsilon: float = 1e-10) -> float:
    """
    Returns the F1 score for pseudoknots.

    Parameters:
    - pk_score (np.ndarray): The [TN, FN, FP, TP] for pseudoknots.
    - epsilon (float): A small value to avoid division by zero.

    Returns:
    - float: The F1 score for pseudoknots.
    """
    TN, FN, FP, TP = pk_score
    precision = TP / (TP + FP + epsilon)
    recall = TP / (TP + FN + epsilon)
    return 2 * (precision * recall) / (precision + recall + epsilon)

def calculate_weighted_f1(lengths: list, f1s: list) -> float:  
    """
    Calculates the weighted F1 score. The weights are based on the sequence lengths.

    Parameters:
    - lengths (list): The lengths of the sequences.
    - f1s (list): The F1 scores of the sequences.

    Returns:
    - float: The weighted F1 score.
    """ 
    total_length = sum(lengths)

    weighted_f1s = [f1s[i] * lengths[i] / total_length for i in range(len(f1s))]

    return sum(weighted_f1s)


def evaluate_file(predicted, file: str, pseudoknots_all, testfile_dir = 'data/test_files', pseudoknots_under = None) -> list:
    """
    Evaluates the predictions for a given file.

    Parameters:
    - file (str): The name of the file to evaluate.
    - pseudoknots (dict): A dictionary to store the pseudoknot scores.
    - lock (multiprocessing.Lock): A lock to write to the pseudoknots dictionary.

    Returns:
    - list: A list with the evaluation results.
    """
    data = pickle.load(open(f'{testfile_dir}/{file}', 'rb'))

    results = [data.length, data.family]
    target = data.output
    data = None #Clear memory 

    target_pk = has_pk(np.argmax(target, axis=1))


    results.extend(evaluate(predicted, target, device)) #Evaluate the prediction
    results.extend(evaluate(predicted, target, device, allow_shift=True)) #Evaluate the prediction with one base pair shifts allowed


    pseudoknot_score = scores_pseudoknot(predicted, target_pk) #Check if predicted and/or target has pseudoknots
    pseudoknots_all += pseudoknot_score
    if results[0] < 600 and pseudoknots_under is not None:
        pseudoknots_under += pseudoknot_score
    
    return results


def plot_F1(df: pd.DataFrame, outputfile: str, method: str = 'RNAUnet16') -> None:
    """
    Plots the F1 score for each sequence length.

    Parameters:
    - df (pd.DataFrame): The dataframe with the evaluation results.
    - outputfile (str): The name of the output file.
    - method (str): The method to plot.

    Returns:
    - None
    """
    colors = mpl.colormaps['cet_glasbey_dark'].colors

    plt.figure(figsize=(12, 4))
    
    for i, (family, group) in enumerate(df.groupby('family')):
        plt.scatter(group['length'], group[f'{method}_f1'], label=family, s=10, color=colors[i], alpha=0.5)
    
    plt.xlabel('Length')
    plt.ylabel('F1 score')

    plt.legend(loc = 'lower right', frameon = False)
    plt.grid(linestyle='--')
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    plt.tight_layout()

    plt.savefig(outputfile, bbox_inches='tight', dpi=300)

def evaluate_families(df: pd.DataFrame, method: str) -> pd.DataFrame:
    """
    Evaluates the families in the dataframe and returns a new dataframe with the results.
    The evaluation is based on the average scores for each family.

    Parameters:
    - df (pd.DataFrame): The dataframe with the evaluation results.
    - method (str): The method to evaluate.

    Returns:
    - pd.DataFrame: A new dataframe with the average scores for each family.
    """
    families = {family:{'count': 0, 'precision': 0, 'recall':0, 'F1': 0, 'precision_shift': 0, 'recall_shift': 0, 'F1_shift': 0, 'min_len': float('inf'), 'max_len': 0} for family in df['family'].unique()}

    for _, row in df.iterrows():
        family = row['family']
        families[family]['count'] += 1
        families[family]['precision'] += row[f'{method}_precision']
        families[family]['recall'] += row[f'{method}_recall']
        families[family]['F1'] += row[f'{method}_f1']
        families[family]['precision_shift'] += row[f'{method}_precision_shift']
        families[family]['recall_shift'] += row[f'{method}_recall_shift']
        families[family]['F1_shift'] += row[f'{method}_f1_shift']

        #Find min and max length of the family
        if row['length'] < families[family]['min_len']:
            families[family]['min_len'] = row['length']

        if row['length'] > families[family]['max_len']:
            families[family]['max_len'] = row['length']
    
    family_df = pd.DataFrame(index=families.keys(), columns=['count', 'lengths', 'precision', 'recall', 'F1', 'precision_shift', 'recall_shift', 'F1_shift'])
    
    for family in families.keys():
        count = families[family]['count']
        len_range = f"{families[family]['min_len']}-{families[family]['max_len']}"
        family_df.loc[family] = [count, len_range, families[family]['precision']/count, families[family]['recall']/count, families[family]['F1']/count, families[family]['precision_shift']/count, families[family]['recall_shift']/count, families[family]['F1_shift']/count]

    return family_df



if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using {device} for prediction\n')

    RNA = namedtuple('RNA', 'input output length family name sequence')

    print('-- Loading model and data --')
    model = RNA_Unet(in_channels=16, channels=32)
    model.load_state_dict(torch.load('RNA_Unet16.pth', map_location=torch.device(device)))
    model.to(device)


    test_data = pickle.load(open('data/test.pkl', 'rb'))
    print('-- Model and data loaded --\n')

    os.makedirs('steps/RNAUnet16', exist_ok=True)
    print('-- Predicting --')
    times_wo_postprocessing = []
    times_only_predict = []
    times_wo_conversion = []
    times_total = []
    lengths = []

    pseudoknots_under = np.array([0, 0, 0, 0])
    pseudoknots_all = np.array([0, 0, 0, 0])

    metrics = ['precision', 'recall', 'f1', 'precision_shift', 'recall_shift', 'f1_shift']

    pseudoknot_F1 = pd.DataFrame(index = ['under', 'all'], columns = ['RNAUnet16'])
    mean_scores= pd.DataFrame(columns = metrics + ['f1_weighted'])

    columns = [f'RNAUnet16_{metric}' for metric in metrics]
    df_testscores = pd.DataFrame(index = range(len(test_data)), columns = ['length', 'family'] + columns)

    progress_bar = tqdm(total=len(test_data), unit='sequence', file=sys.stdout)
    
    #Predict for all sequences and save the results and times
    #Time all steps of prediction, with conversion to matrix, prediction and post-processing
    for i, file in enumerate(test_data):
        name = os.path.basename(file)
        sequence = pickle.load(open(file, 'rb')).sequence
        time1, time2, time3, time4, output = predict(sequence, name, outputdir='steps/RNAUnet16')
        times_wo_postprocessing.append(time1)
        times_only_predict.append(time2)
        times_wo_conversion.append(time3)
        times_total.append(time4)
        lengths.append(len(sequence))

        results = evaluate_file(output, name, pseudoknots_all, pseudoknots_under=pseudoknots_under) 
        df_testscores.loc[i] = results
        
        progress_bar.update(1)

    progress_bar.close()

    print('-- Predictions done --')
    print(f'Total time RNAStralign: {format_time(sum(times_total))}. Average time per sequence: {sum(times_total)/len(test_data):.5f}\n')

    df_under600 = df_testscores[df_testscores['length'] < 600]

    print('-- Saving results --')
    #Check if testscores can be concatenated to existing file
    df_prev_scores = pd.read_csv('results/testscores.csv')
    if df_prev_scores['length'].equals(df_testscores['length']):
        df_prev_scores = pd.concat([df_prev_scores, df_testscores.drop(columns=['length', 'family'])], axis=1)
        df_prev_scores.to_csv('results/testscores.csv', index=False)
        f1 = df_prev_scores[[f'{method}_f1' for method in ['nussinov', 'viennaRNA', 'contrafold', 'CNNfold', 'RNA_Unet', 'RNAUnet16']]]
        f1 = f1.apply(pd.to_numeric, errors='coerce')
        violin_plot(f1, 'Methods', outputfile='figures/evaluation_predictions_RNAUnet16.png')
        print('\t\t Successfully merged and saved testscores for all')
    else:
        df_testscores.to_csv('results/testscores_RNAUnet16.csv', index=False)
        print('\t\t Unable to merge. Saved new testscores for all')
    
    #Same for under 600
    df_prev_under600 = pd.read_csv('results/testscores_under600.csv')
    if df_prev_under600['length'].equals(df_under600['length']):
        df_prev_under600 = pd.concat([df_prev_under600, df_under600.drop(columns=['length', 'family'])], axis=1)
        df_prev_under600.to_csv('results/testscores_under600.csv', index=False)
        f1 = df_prev_under600[[f'{method}_f1' for method in ['nussinov', 'viennaRNA', 'contrafold', 'hotknots', 'CNNfold', 'Ufold', 'RNA_Unet', 'RNAUnet16']]]
        f1 = f1.apply(pd.to_numeric, errors='coerce')
        violin_plot(f1, 'Methods', outputfile='figures/evaluation_predictions_under600_RNAUnet16.png')
        print('\t\t Successfully merged and saved testscores for under 600')
    else:
        df_under600.to_csv('results/testscores_under600_RNAUnet16.csv', index=False)
        print('\t\t Unable to merge. Saved new testscores for under 600')

    print('-- Calculating mean F1 scores and pseudoknot scores --')
    pseudoknot_F1.loc['under'] = f1_pk_score(pseudoknots_under)
    pseudoknot_F1.loc['all'] = f1_pk_score(pseudoknots_all)

    mean_scores.loc['RNAUnet16_under600'] = df_under600[columns].mean().tolist() + [calculate_weighted_f1(df_under600['length'].tolist(), df_under600['RNAUnet16_f1'].tolist())]
    mean_scores.loc['RNAUnet16'] = df_testscores[columns].mean().tolist() + [calculate_weighted_f1(df_testscores['length'].tolist(), df_testscores['RNAUnet16_f1'].tolist())]

    #Evaluate families
    family_df = evaluate_families(df_testscores, 'RNAUnet16')
    family_df.to_csv('results/family_scores_RNAUnet16.csv')
    
    #Add to existing results
    pseudoknot_df = pd.read_csv('results/pseudoknot_F1.csv', index_col=0)
    pseudoknot_df['RNAUnet16'] = pseudoknot_F1['RNAUnet16']
    pseudoknot_df.to_csv('results/pseudoknot_F1.csv')

    mean_df = pd.read_csv('results/average_scores.csv', index_col=0)
    mean_df = pd.concat([mean_df, mean_scores])
    mean_df.to_csv('results/average_scores.csv')

    
    print('-- Make and save plots --')
    plot_F1(df_testscores, 'figures/per_sequence_F1_RNAUnet16.png')

    print('-- Plot and save times --')
    data = {'lengths': lengths, 
            'times w/o post-processing': times_wo_postprocessing, 
            'times for only prediction': times_only_predict,
            'times w/o conversion': times_wo_conversion,
            'times total': times_total}
    
    df = pd.DataFrame(data)
    df = df.sort_values('lengths') #Sort the data by length
    df.to_csv(f'results/times_final16_{device}.csv', index=False)

    data = {'times w/o post-processing': df['times w/o post-processing'].tolist(), 
            'times for only prediction': df['times for only prediction'].tolist(),
            'times w/o conversion': df['times w/o conversion'].tolist(),
            'times total': df['times total'].tolist()}
    
    plot_timedict(data, df['lengths'].tolist(), f'figures/time_final16_{device}.png')



    #### ARCHIVE II ####
    print('--- Starting evaluation for Archive II ---')
    os.makedirs('steps/RNAUnet16_archive', exist_ok=True)
    files = pickle.load(open('data/archiveii.pkl', 'rb'))
    
    pseudoknots = np.array([0, 0, 0, 0])

    df_testscores_archive = pd.DataFrame(index = range(len(files)), columns = ['length', 'family'] + [f'RNAUnet16_{metric}' for metric in metrics])


    print('--- Predicting and evaluating for Archive II ---')
    progress_bar = tqdm(total=len(files), unit='files', file=sys.stdout)

    total_time = 0 
    for i, file in enumerate(files):
        name = os.path.basename(file)
        sequence = pickle.load(open(file, 'rb')).sequence
        _, _, _, file_time, output = predict(sequence, name, outputdir='steps/RNAUnet16_archive')
        results = evaluate_file(output, name, pseudoknots, testfile_dir='data/archiveii')
        total_time += file_time
        df_testscores_archive.loc[i] = results
        progress_bar.update(1)
    
    progress_bar.close()

    print('--- Evaluation done for Archive II ---')
    print(f'Total time Archive II: {format_time(total_time)}. Average time per sequence: {total_time/len(files):.5f}\n')

    print('--- Summarizing and saving results for Archive II ---')
    # Save testscores
    df_archive = pd.read_csv('results/testscores_archive.csv')
    if df_archive['length'].equals(df_testscores_archive['length']):
        df_archive = pd.concat([df_archive, df_testscores_archive.drop(columns=['length', 'family'])], axis=1)
        df_archive.to_csv('results/testscores_archive.csv', index=False)
        f1 = df_archive[[f'{method}_f1' for method in ['nussinov', 'viennaRNA', 'contrafold', 'CNNfold', 'RNAUnet', 'RNAUnet16']]]
        f1 = f1.apply(pd.to_numeric, errors='coerce')
        violin_plot(f1, 'Methods', outputfile='figures/evaluation_predictions_archive_RNAUnet16.png')
        print('\t\t Successfully merged and saved testscores')
    else:
       df_testscores_archive.to_csv('results/testscores_archive_RNAUnet16.csv', index=False)
       print('\t\t Unable to merge. Saved new testscores')

    #Save pseudoknot score
    df_pseudoknot_archive = pd.read_csv('results/pseudoknot_F1_archive.csv', index_col=0)
    df_pseudoknot_archive["RNAUnet16"] = [f1_pk_score(pseudoknots)]
    df_pseudoknot_archive.to_csv('results/pseudoknot_F1_archive.csv')

    #Save mean scores
    df_average_archive = pd.read_csv('results/average_scores_archive.csv', index_col=0)
    df_average_archive.loc['RNAUnet16'] = df_testscores_archive[[f'RNAUnet16_{metric}' for metric in metrics]].mean().tolist() + [calculate_weighted_f1(df_testscores_archive['length'].tolist(), df_testscores_archive[f'RNAUnet16_f1'].tolist())]
    df_average_archive.to_csv('results/average_scores_archive.csv')

    #Save family scores
    family_df = evaluate_families(df_testscores_archive, 'RNAUnet16')
    family_df.to_csv('results/family_scores_RNAUnet16_archive.csv')

    print('--- Make plots for Archive II ---')
    plot_F1(df_testscores_archive, 'figures/per_sequence_F1_archive_RNAUnet16.png')





