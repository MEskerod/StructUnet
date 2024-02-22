import os, tempfile, shutil, tarfile, pickle, sys

import random as rd
from collections import namedtuple

from utils.prepare_data import getLength, list_all_files, make_matrix_from_basepairs, make_matrix_from_sequence_8, make_matrix_from_sequence_9, make_matrix_from_sequence_17, read_ct, update_progress_bar, make_pairs_from_list

def getFamily(file_name: str):
  '''
  Returns the family of a file in the RNAStralign data set, based on folder structure
  '''
  return '_'.join(file_name.split(os.sep)[5].split('_')[:-1])

def process_and_save(file_list: list, output_folder: str): 
    """
    """
    converted  = 0

    os.makedirs(output_folder, exist_ok=True)

    for i, file in enumerate(file_list): 
        length = getLength(file)
        if length == 0: 
            continue

        family = getFamily(file)

        sequence, pairs = read_ct(file)

        try: 
            if (i+1) % 100 == 0:
                update_progress_bar(i, len(file_list))
            
            input_matrix = make_matrix_from_sequence_8(sequence)
            output_matrix = make_matrix_from_basepairs(pairs)

            if input_matrix.shape[-1] == 0 or output_matrix.shape[-1] == 0: 
                continue

            sample = RNA_data(input = input_matrix,
                              output = output_matrix,
                              length = length,
                              family = family,
                              name = file,
                              sequence = sequence)

            pickle.dump(sample, open(os.path.join(output_folder, os.path.splitext(os.path.basename(file))[0] + '.pkl'), 'wb'))

        except Exception as e: 
            # Skip this file if an unexpected error occurs during processing
            print(f"Skipping {file} due to unexpected error: {e}", file=sys.stderr)
    
    print(f"\n\n{converted} files converted", file=sys.stdout)

    if __name__ == "__main__": 
        RNA_data = namedtuple('RNA_data', 'input output length family name sequence')

        tar_file_path = 'data/RNAStralign.tar.gz'

        temp_dir = tempfile.mkdtemp()

        try: 
            with tarfile.open(tar_file_path, 'r:gz') as tar: 
                print("Extract files", file=sys.stdout)
                tar.extractall(temp_dir)
        
            file_list = list_all_files(temp_dir)
            print(f'Total of {len(file_list)} files where extracted\n', file=sys.stdout)

            print("Convert matrices\n", file=sys.stdout)
            process_and_save(file_list, "data/complete_set")
        
        finally:
            shutil.rmtree(temp_dir)
