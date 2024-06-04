import tempfile, tarfile, sys, shutil, os

import pandas as pd

from tqdm import tqdm

from utils.prepare_data import read_ct, list_all_files

def find_basepairs(sequence: str, pairs: list) -> dict:
    """
    """
    watson_crick = {'AU', 'UA', 'CG', 'GC'}
    wobble = {'GU', 'UG'}

    basepairs = {"Non-standard": 0, "Watson-Crick": 0, "Paired with N": 0, "Wobble": 0} 

    for i, j in enumerate(pairs):
        if j == None or j < i:
            continue

        pair = sequence[i]+sequence[j]

        if pair in watson_crick:
            basepairs["Watson-Crick"] += 1
        elif pair in wobble:
            basepairs["Wobble"] += 1
        elif sequence[i] == 'N' or sequence[j] == 'N':
            basepairs["Paired with N"] += 1
        else:
            basepairs["Non-standard"] += 1
    
    return basepairs

def getFamily(file_name: str) -> str:
  '''
  Returns the family of a file in the RNAStralign data set, based on folder structure

  Parameters:
  - file_name (str): The path to the file to extract the family from.
  
  Returns:
  str: The family of the file.

  '''
  return '_'.join(file_name.split(os.sep)[5].split('_')[:-1])




if __name__ == '__main__':
    tar_file_path = 'data/RNAStralign.tar.gz'

    temp_dir = tempfile.mkdtemp()

    try: 
        with tarfile.open(tar_file_path, 'r:gz') as tar: 
            print("Extract files", file=sys.stdout)
            tar.extractall(temp_dir)

        files = list_all_files(temp_dir)
        
        print('Start counting loops')
        progress_bar = tqdm(total=len(files), unit='files')

        pair_list = []
        families = {}

        for file in files:
            try:
                sequence, pairs = read_ct(file)
                p = find_basepairs(sequence, pairs)
                pair_list.append(p)
                
                family = getFamily(file)
                if family not in families:
                    families[family] = {'total': 0, 'non_standard': 0}
                families[family]['total'] += 1

                if p["Non-standard"] != 0:
                    families[family]['non_standard'] += 1

                progress_bar.update(1)
            except:
                pass
        
        progress_bar.close()

  
        df_pair = pd.DataFrame(pair_list)
        df_pair.to_csv('results/pair_counts.csv')
        
        df_families = pd.DataFrame(families).T
        df_families.to_csv('results/non_standard_pr_family.csv')

        below_limit = 0

        #Calculate stats
        for column in df_pair.columns: 
            print(f"Number of {column} basepairs: {df_pair[column].sum()}")
                
        print("Percentage of basepairs that are non-standard: ", df_pair["Non-standard"].sum()/df_pair.sum().sum() *100, "%")
        print("Percentage of sequences with non-standard basepairs: ", (df_pair["Non-standard"] != 0).sum()/len(df_pair)*100, "%")
        
        print("Percentage of basepairs with N: ", df_pair["Paired with N"].sum()/df_pair.sum().sum() * 100, "%")
        print("Percentage of sequences with N: ", (df_pair["Paired with N"] != 0).sum()/len(df_pair)*100, "%")


        print("\nFamilies with non-standard basepairs: ")
        for family, count in families.items():
            print(f"Non-standard basepairs in family {family}: {count['non_standard']} ({count['non_standard']/count['total']*100}%)")
        
    finally:
        shutil.rmtree(temp_dir)