import os, pickle, torch, multiprocessing

from collections import namedtuple
from tqdm import tqdm
from functools import partial

import pandas as pd
import numpy as np

from utils.model_and_training import evaluate
from utils.post_processing import prepare_input, blossom_weak

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

def make_random_prediction(sequence):
    """
    """
    #Make random prediction
    predicted = torch.rand(len(sequence), len(sequence))

    #Apply same post-processing as in the model
    predicted = prepare_input(predicted, sequence, device)
    predicted = blossom_weak(predicted, sequence, device)
    
    return predicted

def evaluate_random_prediction(file, dataset, pseudoknots, lock):
    data = pickle.load(open(file, 'rb'))

    results = [data.length, data.family, dataset]

    target = data.output
    target_pk = has_pk(np.argmax(target, axis=1))

    predicted = make_random_prediction(data.sequence)

    results.extend(evaluate(predicted, target, device))
    results.extend(evaluate(predicted, target, device, allow_shift=True))

    sequence_pk = scores_pseudoknot(predicted, target_pk) #Check if pseudoknots are present in the predicted and target structure
    with lock:
        pseudoknots[dataset] += sequence_pk
    return results 

def evaluate_families(df: pd.DataFrame, dataset = None):
    if dataset:
        df = df[df['dataset'] == dataset]
    
    families = {family:{'count': 0, 'precision': 0, 'recall':0, 'F1': 0, 'precision_shift': 0, 'recall_shift': 0, 'F1_shift': 0, 'min_len': float('inf'), 'max_len': 0} for family in df['family'].unique()}

    for _, row in df.iterrows():
        family = row['family']
        families[family]['count'] += 1
        families[family]['precision'] += row['precision']
        families[family]['recall'] += row['recall']
        families[family]['F1'] += row['f1']
        families[family]['precision_shift'] += row['precision_shift']
        families[family]['recall_shift'] += row['recall_shift']
        families[family]['F1_shift'] += row['f1_shift']

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
    device = 'cpu'
    RNA = namedtuple('RNA', 'input output length family name sequence')

    archive = pickle.load(open('data/archiveii.pkl', 'rb'))
    align = pickle.load(open('data/test.pkl', 'rb'))

    n_files = len(archive) + len(align)

    columns = ['length', 'family', 'dataset', 'precision', 'recall', 'f1', 'precision_shift', 'recall_shift', 'f1_shift']
    df = pd.DataFrame(index= range(n_files), columns=columns)

    num_cores = 10
    print(f'Number of cores: {num_cores}')
    pool = multiprocessing.Pool(num_cores)
    manager = multiprocessing.Manager()
    lock = manager.Lock()

    pseudoknots = manager.dict({method: np.array([0, 0, 0, 0]) for method in ['archiveII', 'RNAStralign']})

    print('Make and evaluate random predictions...')
    with tqdm(total=len(archive), unit='files', desc='ArchiveII') as progress_bar:
        partial_func = partial(evaluate_random_prediction, dataset='archiveII', pseudoknots=pseudoknots, lock=lock)
        for i, result in enumerate(pool.imap_unordered(partial_func, archive)):
            df.loc[i] = result
            progress_bar.update()
    
    
    with tqdm(total=len(align), unit='files', desc='RNAStralign') as progress_bar:
        partial_func = partial(evaluate_random_prediction, dataset='RNAStralign', pseudoknots=pseudoknots, lock=lock)
        for i, result in enumerate(pool.imap_unordered(partial_func, align)):
            df.loc[i+len(archive)] = result
            progress_bar.update()

    
    pool.close()
    pool.join()

    print('Evaluation done. Saving results...')

    df.to_csv('results/testscores_random.csv', index=False)

    print('Pseudoknots:', pseudoknots)

    #Calculate F1 scores for pseudoknots
    pseudoknots['combined'] = pseudoknots['archiveII'] + pseudoknots['RNAStralign']
    pseudoknots = pd.DataFrame(data = {'F1': [f1_pk_score(pseudoknots['archiveII']), f1_pk_score(pseudoknots['RNAStralign']), f1_pk_score(pseudoknots['combined'])]}, index = ['ArchiveII', 'RNAStralign', 'combined'])
    pseudoknots.to_csv('results/pseudoknot_F1_random.csv')


    #Calculate average scores
    mean_scores = pd.DataFrame(columns = ['precision', 'recall', 'f1', 'precision_shift', 'recall_shift', 'f1_shift', 'f1_weighted'])
    archive_df = df[df['dataset'] == 'archiveII']
    align_df = df[df['dataset'] == 'RNAStralign']
    
    mean_scores.loc['archiveII'] = archive_df[['precision', 'recall', 'f1', 'precision_shift', 'recall_shift', 'f1_shift']].mean().tolist() + [calculate_weighted_f1(archive_df['length'], archive_df['f1'])]
    mean_scores.loc['RNAStralign'] = align_df[['precision', 'recall', 'f1', 'precision_shift', 'recall_shift', 'f1_shift']].mean().tolist() + [calculate_weighted_f1(align_df['length'], align_df['f1'])]
    mean_scores.loc['combined'] = df[['precision', 'recall', 'f1', 'precision_shift', 'recall_shift', 'f1_shift']].mean().tolist() + [calculate_weighted_f1(df['length'], df['f1'])]

    mean_scores.to_csv('results/average_scores_random.csv')

    #Evaluate families: 
    family_df = evaluate_families(df)
    family_df.to_csv('results/family_scores_random.csv')

    family_df = evaluate_families(df, 'archiveII')
    family_df.to_csv('results/family_scores_random_archive.csv')

    family_df = evaluate_families(df, 'RNAStralign')
    family_df.to_csv('results/family_scores_random_align.csv')

