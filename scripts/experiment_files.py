import os, tempfile, shutil, tarfile, pickle, sys

import random as rd
from collections import namedtuple

from utils.prepare_data import make_matrix_from_sequence_8, make_matrix_from_sequence_17, make_matrix_from_basepairs, read_ct, getLength, list_all_files

def getFamily(file_name):
  '''
  '''
  return ''.join(file_name.split(os.sep)[5].split('_')[:-1])

def underGivenLength(length: int, data_size: int, file_list: list):
  '''
  Returns a list of files under length with data_size from EACH family.
  If a family does not have enough data, all data from that family is added.
  '''
  rd.seed(42)
  
  families = list(set([getFamily(file) for file in file_list])) #Find all family names and remove duplicates

  data = {family:[line for line in file_list if family in line and getLength(line)<length] for family in families} #Create list of lists, where each list contains all files from a family

  files = []
  for name, family in data.items(): #Pick data_size files from each family
      try:
          files+=rd.sample(family, data_size)
      except:
          print(f"Not enough data in {name}, adding all", file=sys.stdout)
          files+=family
  return files

def update_progress_bar(current_index, total_indices):
    progress = (current_index + 1) / total_indices
    bar_length = 50
    filled_length = int(bar_length * progress)
    bar = '=' * filled_length + '.' * (bar_length - filled_length)
    sys.stdout.write(f'\r[{bar}] {int(progress * 100)}%')
    sys.stdout.flush()

def process_and_save(file_list, output_folder, matrix_type = '8'):
  converted = 0

  os.makedirs(output_folder, exist_ok=True)
  
  for i, file in enumerate(file_list):

    length = getLength(file)
    family = getFamily(file)

    sequence, pairs = read_ct(file)

    try:
        if (i + 1) % 100 == 0:
            update_progress_bar(i, len(file_list))

        if matrix_type == '8':
            input_matrix = make_matrix_from_sequence_8(sequence)
        elif matrix_type == '17':  
            input_matrix = make_matrix_from_sequence_17(sequence)
        else: 
            raise ValueError("Wrong matrix type")
        
        output_matrix = make_matrix_from_basepairs(sequence, pairs)

        sample = RNA_data(input = input_matrix,
                                     output = output_matrix,
                                     length = length,
                                     family = family,
                                     name = file, 
                                     pairs = [pair for pair in pairs if pair[0] < pair[1]])
        
        pickle.dump(sample, open(os.path.join(output_folder, os.path.splitext(os.path.basename(file))[0] + '.pkl'), 'wb'))
        converted += 1

    except Exception as e:
        # Skip this file if an unexpected error occurs during processing
        print(f"Skipping {file} due to unexpected error: {e}", file=sys.stderr)
    
  print(f"\n\n{converted} files converted", file=sys.stdout)

if __name__ == "__main__": 
    matrix_type = sys.argv[1]
    
    RNA_data = namedtuple('RNA_data', 'input output length family name pairs')

    tar_file_path = 'data/RNAStralign.tar.gz'

    temp_dir = tempfile.mkdtemp()
   
    try: 
        with tarfile.open(tar_file_path, 'r:gz') as tar: 
            print("Extract files", file=sys.stdout)
            tar.extractall(temp_dir)
        
        file_list = underGivenLength(500, 5000, list_all_files(temp_dir))
        print(f"Total of {len(file_list)} files chosen\n", file=sys.stdout)
        
        print("Convert to matrices\n", file=sys.stdout)
        process_and_save(file_list, f"data/experiment{matrix_type}")
    
    finally: 
        shutil.rmtree(temp_dir)
      
        