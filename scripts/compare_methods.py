import os, pickle

import numpy as np
import pandas as pd

from collections import namedtuple

from utils.model_and_training import evaluate

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
        if i>j:
            i, j = j, i
        if i==j:
            continue
        if np.max(pairings[i:j]) > j:
            return True
        if np.min(pairings[i:j]) < i:
            return True
    return False

def scores_pseudoknot(predicted: np.ndarray, target: np.ndarray) -> np.ndarray:
    """
    Returns the [TN, FN, FP, TP] for pseudoknots in the predicted and target structure.

    Parameters:
    - predicted (np.ndarray): The predicted structure as a NumPy array.
    - target (np.ndarray): The target structure as a NumPy array.

    Returns:
    - np.ndarray: The [TN, FN, FP, TP] for pseudoknots in the predicted and target structure.
    """
    pk_score = np.array([0, 0, 0, 0])

    predicted, target = predicted.squeeze(), target.squeeze()
    i = has_pk(np.argmax(predicted, axis=1))
    j = has_pk(np.argmax(target, axis=1))
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


def evaluate_method_under600(method: str) -> tuple:
    """
    Calculated the F1, recall, precision and scores for pseudoknots for the method on the under600 test set.

    Parameters:
    - method (str): The method to evaluate.

    Returns:
    - f1_results (list): The F1 scores for the method.
    - recall_results (list): The recall scores for the method.
    - precision_results (list): The precision scores for the method.
    - pseudoknots (list): Returns the F1 score for prediction of pseudoknots in structure using the method.
    """
    pseudoknots = [0, 0, 0, 0]
    f1_results, recall_results, precision_results = [], [], []

    f1_shifted, recall_shifted, precision_shifted = 0, 0, 0

    assert all(test[i] in os.listdir(f'steps/{method}') for i in under600)

    for i in under600:
        target = pickle.load(test[i]).output.numpy()
        predicted = pickle.load(test[i].replace('data/test_set', f'steps/{method}'))

        #Calculate the scores for the sequences under 600
        precision, recall, f1 = evaluate(predicted, target)
        f1_results.append(f1), recall_results.append(recall), precision_results.append(precision)

        #Calculate the scores for the sequences under 600 with one basepair shift allowed
        precision, recall, f1 = evaluate(predicted, target, allow_shift=True)
        precision_shifted, recall_shifted, f1_shifted = precision_shifted + precision, recall_shifted + recall, f1_shifted + f1

        #Find if structure (true and/or predicted) has pseudoknots
        pseudoknots += scores_pseudoknot(predicted, target)

    return f1_results, recall_results, precision_results, f1_shifted/len(under600), recall_shifted/len(under600), precision_shifted/len(under600), f1_pk_score(pseudoknots)

def evaluate_method(method: str, f1_results: list, recall_results: list, precision_results: list) -> tuple:
    """
    Calculated the F1, recall, precision and scores for pseudoknots for the method on all the sequences longer than 600.
    The scores for f1, recall and precision are appended to the given lists, which should be the scores for the sequences under 600.

    Parameters:
    - method (str): The method to evaluate.
    - f1_results (list): The F1 scores for the method.
    - recall_results (list): The recall scores for the method.
    - precision_results (list): The precision scores for the method.

    Returns:
    - f1_results (list): The F1 scores for the method.
    - recall_results (list): The recall scores for the method.
    - precision_results (list): The precision scores for the method.
    - pseudoknots (list): Returns the F1 score for prediction of pseudoknots in structure using the method.
    """
    pseudoknots = [0, 0, 0, 0]
    f1_shifted, recall_shifted, precision_shifted = 0, 0, 0
    
    assert all(test[i] in os.listdir(f'steps/{method}') for i in over600)

    for i in over600:
        target = pickle.load(test[i]).output.numpy()
        predicted = pickle.load(test[i].replace('data/test_set', f'steps/{method}'))
        
        #Calculate the scores for the sequences over 600
        precision, recall, f1 = evaluate(predicted, target)
        f1_results.append(f1), recall_results.append(recall), precision_results.append(precision)

        #Calculate the scores for the sequences over 600 with one basepair shift allowed
        precision, recall, f1 = evaluate(predicted, target, allow_shift=True)
        precision_shifted, recall_shifted, f1_shifted = precision_shifted + precision, recall_shifted + recall, f1_shifted + f1

        #Find if structure (true and/or predicted) has pseudoknots
        pseudoknots += scores_pseudoknot(predicted, target)
    return f1_results, recall_results, precision_results, f1_shifted/len(over600), recall_shifted/len(over600), precision_shifted/len(over600), f1_pk_score(pseudoknots)

if __name__ == "__main__":
    RNA = namedtuple('RNA', 'input output length family name sequence')

    methods600 = ['Ufold', 'hotknots']
    methods = ['CNNfold', 'vienna_mfold', 'RNAUnet']

    ### CALCULATE F1 SCORES ###

    test = pickle.load(open('data/test.pkl', 'rb'))
    under600 = pickle.load(open('data/under600.pkl', 'rb'))
    over600 = [i for i in range(len(test)) if i not in under600] #All sequences longer than 600

    #Allocate data frames
    pseudoknot_F1  = pd.DataFrame(index=['under', 'all'], columns=methods600+methods)
    df_under600 = pd.DataFrame()
    df_all = pd.DataFrame()
    mean_scores = pd.DataFrame(index=[f'{method}_under600' for method in methods600+methods]+methods, columns=['precision', 'recall', 'f1', 'precision_shift', 'recall_shift', 'f1_shift'])

    #Get the lengths and families of the sequences
    length_under600 = [pickle.load(test[i]).length for i in under600]
    length_over600 = [pickle.load(test[i]).length for i in over600]

    families_under600 = [pickle.load(test[i]).family for i in under600]
    families_over600 = [pickle.load(test[i]).family for i in over600]

    #Add the lengths and families to the data frames
    df_under600['length'] = length_under600
    df_under600['family'] = families_under600

    df_all['length'] = length_under600 + length_over600
    df_all['family'] = families_under600 + families_over600

    #Evaluate all methods for sequences under 600
    for method in methods600+methods:
        f1, recall, precision, f1_shifted, recall_shifted, precision_shifted, pseudoknot_F1 = evaluate_method_under600(method)
        df_under600[f'{method}_precision'] = precision, df_under600[f'{method}_recall'] = recall, df_under600[f'{method}_f1'] = f1
        pseudoknot_F1.loc['under', method] = pseudoknot_F1
        mean_scores.loc[f'{method}_under600'] = [np.mean(precision), np.mean(recall), np.mean(f1), precision_shifted, recall_shifted, f1_shifted]
    
    
    df_under600.to_csv('results/test_scores_under600.csv')

    #Evaluate all methods that could predict all sequences for sequences over 600
    #The scores for the sequences under 600 are appended to the lists and saved
    for method in methods:
        f1, recall, precision = df_under600[f'{method}_f1'].to_list(), df_under600[f'{method}_recall'].to_list(), df_under600[f'{method}_precision'].to_list()
        f1, recall, precision, f1_shifted, recall_shifted, precision_shifted, pseudoknot_F1 = evaluate_method(method, f1, recall, precision)
        df_all[f'{method}_precision'] = precision, df_all[f'{method}_recall'] = recall, df_all[f'{method}_f1'] = f1
        
        #Harmonic mean of values for the two parts of the dataset
        pseudoknot_F1.loc['all', method] = (pseudoknot_F1*len(over600) + pseudoknot_F1.loc['under', method]*len(under600))/(len(under600)+len(over600))
        precision_shifted = (precision_shifted*len(over600) + mean_scores.loc[f'{method}_under600', 'precision_shift']*len(under600))/(len(under600)+len(over600))
        recall_shifted = (recall_shifted*len(over600) + mean_scores.loc[f'{method}_under600', 'recall_shift']*len(under600))/(len(under600)+len(over600))
        f1_shifted = (f1_shifted*len(over600) + mean_scores.loc[f'{method}_under600', 'f1_shift']*len(under600))/(len(under600)+len(over600))

        mean_scores.loc[method] = [np.mean(precision), np.mean(recall), np.mean(f1), precision_shifted, recall_shifted, f1_shifted]
    
    df_all.to_csv('results/test_scores.csv')
    pseudoknot_F1.to_csv('results/f1_pseudoknots.csv')
    mean_scores.to_csv('results/average_scores_methods.csv')

    #Find precision, recall and F1 for each family
    families = {family:{'count':0, 'precision': 0, 'recall': 0, 'F1': 0} for family in set(df_all['family'])}
    for index, row in df_all.iterrows():
        family = row['family']
        families[family]['count'] += 1
        families[family]['precision'] += row['RNAUnet_precision']
        families[family]['recall'] += row['RNAUnet_recall']
        families[family]['F1'] += row['RNAUnet_f1']

    family_df = pd.DataFrame(index=families.keys(), columns=['count', 'precision', 'recall', 'F1'])
    for family in families:
        family_df.loc[family] = [families[family]['count'], families[family]['precision']/families[family]['count'], families[family]['recall']/families[family]['count'], families[family]['F1']/families[family]['count']]
    family_df.to_csv('results/RNAUnet_family_scores.csv')

    #### MAKE PLOTS ####
