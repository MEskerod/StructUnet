import os, tempfile, shutil, tarfile, pickle, sys

import random as rd
from collections import namedtuple

from utils.prepare_data import getLength, list_all_files, process_and_save

def getFamily(file_name: str):
  '''
  Returns the family of a file in the RNAStralign data set, based on folder structure
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
        process_and_save(file_list, f"data/experiment{matrix_type}", matrix_type=matrix_type)
    
    finally: 
        shutil.rmtree(temp_dir)
      
        