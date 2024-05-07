import os, pickle, torch, multiprocessing

import pandas as pd
import numpy as np

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
    data = pickle.load(open(f'data/test_files/{file}', 'rb'))

    results = [data.length, data.family]
    target = data.output
    data = None #Clear memory 

    target_pk = has_pk(np.argmax(target, axis=1))

    for method in methods:
        predicted = pickle.load(open(f'steps/{method}/{file}', 'rb'))
        results.extend(evaluate(predicted, target, device)) #Evaluate the prediction
        results.extend(evaluate(predicted, target, device, allow_shift=True)) #Evaluate the prediction with one base pair shifts allowed

        #Lock before updating the pseudoknots
        with lock:
            pseudoknots[method] += scores_pseudoknot(predicted, target_pk) #Check if predicted and/or target has pseudoknots
    
    return results




if __name__ == "__main__":
    device = 'cpu'
    RNA = namedtuple('RNA', 'input output length family name sequence')

    methods = ['Ufold', 'hotknots', 'CNNfold', 'viennaRNA', 'nussinov', 'contrafold', 'RNAUnet']
    metrics = ['precision', 'recall', 'f1', 'precision_shift', 'recall_shift', 'f1_shift']

    test = pickle.load(open('data/test.pkl', 'rb'))
    under600 = pickle.load(open('data/test_under_600.pkl', 'rb'))

    files = [os.path.basename(test[i]) for i in under600]

    #Allocate dataframes
    pseudoknot_F1 = pd.DataFrame(index = ['under', 'all'], columns = methods)
    columns = ['length', 'family'] + [f'{name}_{metric}' for name in methods for metric in metrics]
    df_under600 = pd.DataFrame(index = range(len(files)), columns=columns)
    mean_scores= pd.DataFrame(columns = metrics)

    print("--- Starting evaluation ---")
    
    num_processes = 1 #FIXME 
    print(f"Number of processes: {num_processes}")
    pool = multiprocessing.Pool(num_processes)

    manager = multiprocessing.Manager()
    lock = manager.Lock()
    pseudoknots = manager.dict({method: [0, 0, 0, 0] for method in methods})
    
    with tqdm(total=len(files), unit='files') as pbar:
        partial_func = partial(evaluate_file, pseudoknots=pseudoknots, lock=lock)
        for i, result in enumerate(pool.imap_unordered(partial_func, files)):
            df_under600.loc[i] = result
            pbar.update()

    #Close the pool 
    pool.close()
    pool.join()
    
    #Add results to dataframes
    for method in methods:
        pseudoknot_F1.loc['under', method] = f1_pk_score(pseudoknots[method])
        mean_scores.loc[f'{method}_under600'] = df_under600[[f'{method}_{metric}' for metric in metrics]].mean()



    print("--- Evaluation done ---")

    print("--- Saving results ---")
    df_under600.to_csv('results/testscores_under600.csv', index=False)
    pseudoknot_F1.to_csv('results/pseudoknot_F1.csv')
    mean_scores.to_csv('results/average_scores.csv')

    #Make plots
    f1 = df_under600[[f'{method}_f1' for method in methods]]
    f1 = f1.apply(pd.to_numeric, errors='coerce')
    violin_plot(f1, 'Methods', outputfile='figures/evaluation_predictions_under600.png')

    