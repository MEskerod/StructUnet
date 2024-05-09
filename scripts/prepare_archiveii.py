import torch, os, sys, pickle

import pandas as pd

from tqdm import tqdm

from collections import namedtuple

from utils.prepare_data import make_matrix_from_sequence_8

def make_matrix_from_basepairs(basepairs: list) -> torch.Tensor:
    """
    Converts a list of basepairs to a matrix representation.

    Parameters:
    - basepairs (list): The list of basepairs to convert.

    Returns:
    - torch.Tensor: The matrix representation of the basepairs.
    """
    N = len(basepairs)
    matrix = torch.zeros(N, N)

    for i, j in enumerate(basepairs):
        matrix[i, int(j)] = 1

    return matrix

def read_csv(file_name: str):
    """
    Reads a csv file and returns it as a pandas DataFrame.

    Parameters:
    - file_name (str): The name of the file to read

    """
    names = []
    sequences = []
    pairings = []
    families = []

    with open(file_name, 'r') as file:
        samples = file.read().split('\n')
    
    for sample in samples[:-1]:
        data = sample.split(',')
        names.append(data[0])
        families.append(data[0].split('_')[0])
        sequences.append(data[1])
        pairings.append(data[2:])
    
    return names, sequences, pairings, families

def process_and_save(names: list, sequences: list, pairings: list, families: list, output_folder: str) -> None: 
    assert len(names) == len(sequences) == len(pairings) == len(families) #Check if all lists have the same length

    os.makedirs(output_folder, exist_ok=True)

    N = len(names)

    print(f'Processing {N} files')
    
    count = 0

    progress_bar = tqdm(total=N, unit='sequence', desc='Processing files', file=sys.stdout)

    for i in range(N): 
        try: 
            length = len(sequences[i])
            input_matrix = make_matrix_from_sequence_8(sequences[i])
            output_matrix = make_matrix_from_basepairs(pairings[i])

            if input_matrix.shape[-1] == 0 or output_matrix.shape[-1] == 0:
                continue
            
            sample = RNA(input = input_matrix,
                         output = output_matrix,
                         length = length,
                         family = families[i],
                         name = names[i],
                         sequence = sequences[i])    
            
            pickle.dump(sample, open(os.path.join(output_folder, os.path.splitext(names[i])[0] + '.pkl'), 'wb'))
            count += 1
        except Exception as e: 
            # Skip this file if an unexpected error occurs during processing
            print(f"Skipping {names[i]} due to unexpected error: {e}", file=sys.stderr)
            continue
        progress_bar.update(1)
    progress_bar.close()

    print(f"\nA total of {count} files converted", file=sys.stdout)

    

if __name__ == '__main__': 
    RNA = namedtuple('RNA', 'input output length family name sequence')

    names, sequences, pairings, families = read_csv('data/archiveii.csv')

    process_and_save(names, sequences, pairings, families, 'data/archiveii')

    files = [os.path.join('data/archiveii', file) for file in os.listdir('data/archiveii')]
    pickle.dump(files, open('data/archiveii.pkl', 'wb'))