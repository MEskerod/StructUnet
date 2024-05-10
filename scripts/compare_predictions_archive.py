import os, pickle, torch, multiprocessing

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from collections import namedtuple
from functools import partial

from tqdm import tqdm

from utils.model_and_training import evaluate
from utils.plots import violin_plot

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



def evaluate_file(file: str, pseudoknots, lock) -> list:
    """
    """
    data = pickle.load(open(f'data/archiveii/{file}', 'rb'))

    results = [data.length, data.family]
    target = data.output
    data = None #Clear memory 

    target_pk = has_pk(np.argmax(target, axis=1))

    for method in methods:
        predicted = pickle.load(open(f'steps/{method}_archive/{file}', 'rb'))
        results.extend(evaluate(predicted, target, device)) #Evaluate the prediction
        results.extend(evaluate(predicted, target, device, allow_shift=True)) #Evaluate the prediction with one base pair shifts allowed

        #Lock before updating the pseudoknots
        with lock:
            pseudoknots[method] += scores_pseudoknot(predicted, target_pk) #Check if predicted and/or target has pseudoknots
    return results

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


def plot_F1(df: pd.DataFrame, outputfile: str, method: str = 'RNAUnet'):
    """
    """
    plt.figure(figsize=(12, 4))
    
    for family, group in df.groupby('family'):
        plt.scatter(group['length'], group[f'{method}_f1'], label=family, s=10)
    
    plt.xlabel('Length')
    plt.ylabel('F1 score')

    plt.legend(loc = 'lower right', frameon = False)
    plt.grid(linestyle='--')
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    plt.tight_layout()

    plt.savefig(outputfile, bbox_inches='tight', dpi=300)
    



if __name__ == "__main__":
    device = 'cpu'
    RNA = namedtuple('RNA', 'input output length family name sequence')

    methods = ['nussinov', 'viennaRNA', 'contrafold', 'CNNfold', 'RNAUnet']
    metrics = ['precision', 'recall', 'f1', 'precision_shift', 'recall_shift', 'f1_shift']

    files = [os.path.basename(file) for file in pickle.load(open('data/archiveii.pkl', 'rb'))]

    #Allocate dataframes
    pseudoknot_F1 = pd.DataFrame(index = ['all'], columns = methods)
    columns = ['length', 'family'] + [f'{name}_{metric}' for name in methods for metric in metrics]
    df = pd.DataFrame(index = range(len(files)), columns=columns)
    mean_scores= pd.DataFrame(columns = metrics)


    print("--- Starting evaluation ---")
    
    num_processes = 1 
    print(f"Number of processes: {num_processes}")
    pool = multiprocessing.Pool(num_processes)

    manager = multiprocessing.Manager()
    lock = manager.Lock()
    pseudoknots = manager.dict({method: [0, 0, 0, 0] for method in methods})
    
    with tqdm(total=len(files), unit='files') as pbar:
        partial_func = partial(evaluate_file, pseudoknots=pseudoknots, lock=lock)
        for i, result in enumerate(pool.imap_unordered(partial_func, files)):
            df.loc[i] = result
            pbar.update()

    #Close the pool 
    pool.close()
    pool.join()



    print("--- Evaluation done ---")

    print("--- Saving results ---")
    df.to_csv('results/testscores_archive.csv', index=False)


    #Add results to dataframes
    for method in methods:
        #Calculate the F1 score for pseudoknots as a balanced average of the under and over 600 nucleotides
        pseudoknot_F1.loc['all', method] = f1_pk_score(pseudoknots[method])
        #Find the average scores for the method over all sequences
        mean_scores.loc[method] = df[[f'{method}_{metric}' for metric in metrics]].mean().tolist()

    pseudoknot_F1.to_csv('results/pseudoknot_F1_archive.csv')
    mean_scores.to_csv('results/average_scores_archive.csv')

    #Evaluate families
    family_df = evaluate_families(df, 'RNAUnet')
    family_df.to_csv('results/family_scores_archive.csv')

    print("--- Making plots ---")
    #Make plots
    f1 = df[[f'{method}_f1' for method in methods]]
    f1 = f1.apply(pd.to_numeric, errors='coerce')
    violin_plot(f1, 'Methods', outputfile='figures/evaluation_predictions_all_archive.png')

    plot_F1(df, 'figures/per_sequence_F1_archive.png')

    