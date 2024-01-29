import os, tempfile, shutil, tarfile, pickle, sys

import random as rd
from collections import namedtuple

from utils.prepare_data import make_matrix_from_sequence_8, make_matrix_from_sequence_17, make_matrix_from_basepairs, read_ct, getFamily, getLength, list_all_files

def underGivenLength(length: int, data_size: int, file_list: list):
  '''
  Returns a list of files under length with data_size from EACH family.
  If a family does not have enough data, all data from that family is added.
  '''
  rd.seed(42)
  
  families = list(set([file.split(os.sep)[4] for file in file_list])) #Find all family names and remove duplicates

  data = [[line for line in file_list if family in line and getLength(line)<length] for family in families] #Create list of lists, where each list contains all files from a family

  files = []
  for family in data: #Pick data_size files from each family
      try:
          files+=rd.sample(family, data_size)
      except:
          print("Not enough data in family, adding all")
          files+=family
  return files

def update_progress_bar(current_index, total_indices):
    progress = (current_index + 1) / total_indices
    bar_length = 50
    filled_length = int(bar_length * progress)
    bar = '=' * filled_length + '.' * (bar_length - filled_length)
    sys.stdout.write(f'\r[{bar}] {int(progress * 100)}%')
    sys.stdout.flush()

def process_and_save(file_list, output_file):
  all_files = []
  
  
  for i, file in enumerate(file_list):

    length = getLength(file)
    family = getFamily(file)

    sequence, pairs = read_ct(file)

    try:
        if (i + 1) % 100 == 0:
            update_progress_bar(i, len(file_list))

        input_matrix_8 = make_matrix_from_sequence_8(sequence) 
        input_matrix_17 = make_matrix_from_sequence_17(sequence)
        output_matrix = make_matrix_from_basepairs(sequence, pairs)

        sample = RNA_data_experiment(input_8 = input_matrix_8,
                                     input_17 = input_matrix_17, 
                                     output = output_matrix,
                                     length = length,
                                     family = family,
                                     name = file, 
                                     pairs = [pair for pair in pairs if pair[0] < pair[1]])
        
        all_files.append(sample)

    except Exception as e:
        # Skip this file if an unexpected error occurs during processing
        print(f"Skipping {file} due to unexpected error: {e}")
    
  print("\n\nSave file")
  pickle.dump(all_files, open(output_file, 'wb'))

if __name__ == "__main__": 
    RNA_data_experiment = namedtuple('RNA_data_experiment', 'input_8 input_17 output length family name pairs')

    tar_file_path = 'data/RNAStralign.tar.gz'

    temp_dir = tempfile.mkdtemp()
   
    try: 
        with tarfile.open(tar_file_path, 'r:gz') as tar: 
            print("Extract files")
            tar.extractall(temp_dir)
        
        file_list = underGivenLength(500, 5000, list_all_files(temp_dir))
        print(f"Total of {len(file_list)} files chosen\n")
        
        print("Convert to matrices\n")
        process_and_save(file_list, "data/experiment.pkl")
    
    finally: 
        shutil.rmtree(temp_dir)
      
        