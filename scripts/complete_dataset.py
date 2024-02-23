import os, tempfile, shutil, tarfile, pickle, sys

import random as rd
from collections import namedtuple

from utils import prepare_data
from utils import plots

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
        length = prepare_data.getLength(file)
        if length == 0: 
            continue

        family = getFamily(file)

        sequence, pairs = prepare_data.read_ct(file)

        try: 
            if (i+1) % 100 == 0:
                prepare_data.update_progress_bar(i, len(file_list))
            
            input_matrix = prepare_data.make_matrix_from_sequence_8(sequence)
            output_matrix = prepare_data.make_matrix_from_basepairs(pairs)

            if input_matrix.shape[-1] == 0 or output_matrix.shape[-1] == 0: 
                continue

            sample = RNA(input = input_matrix,
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
    print("RUN")
        
    RNA = namedtuple('RNA', 'input output length family name sequence')

    tar_file_path = 'data/RNAStralign.tar.gz'

    temp_dir = tempfile.mkdtemp()

    try: 
        with tarfile.open(tar_file_path, 'r:gz') as tar: 
            print("Extract files", file=sys.stdout)
            tar.extractall(temp_dir)
        
        file_list = prepare_data.list_all_files(temp_dir)
        print(f'Total of {len(file_list)} files where extracted\n', file=sys.stdout)

        print("Convert matrices\n", file=sys.stdout)
        process_and_save(file_list, "data/complete_set")
        
    finally:
        shutil.rmtree(temp_dir)
        

    print("Splitting data", file=sys.stdout)

    file_list = [os.path.join('data', 'complete_set', file) for file in os.listdir('data/experiment8')]

    train, valid, test = prepare_data.split_data(file_list, validation_ratio=0.2, test_ratio=0.0)

    pickle.dump(train, open('data/train.pkl', 'wb'))
    pickle.dump(valid, open('data/valid.pkl', 'wb'))
    pickle.dump(test, open('data/test.pkl', 'wb'))

    print("Move test files to different folder")
    os.makedirs("data/test_files")
    for file in test: 
        shutil.move(file, "data/test_files")

    print(f"FILES FOR TRAINING: {len(train)} for training, {len(valid)} for validation", file=sys.stdout)

    print("Make family map", file=sys.stdout)
        
    family_map = prepare_data.make_family_map(file_list)
    pickle.dump(family_map, open('data/familymap.pkl', 'wb'))

    print("Plotting data distribution", file=sys.stdout)
        
    plots.plot_families({"train":train, "valid":valid, "test":test}, family_map, output_file='figures/family_distribution.png')
    plots.plot_len_histogram({"train":train, "valid":valid, "test":test}, output_file='figures/length_distribution.png')
     
