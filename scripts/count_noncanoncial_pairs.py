import tempfile, tarfile, sys, shutil

import pandas as pd

from tqdm import tqdm

from utils.prepare_data import read_ct, list_all_files

def find_basepairs(sequence: str, pairs: list) -> dict:
    """
    """
    watson_crick = {'AU', 'UA', 'CG', 'GC'}
    wobble = {'GU', 'UG'}

    basepairs = {"Non-canonical": 0, "Watson-Crick": 0, "Wobble": 0} 

    for i, j in enumerate(pairs):
        if j == None or j < i:
            continue

        pair = sequence[i]+sequence[j]

        if pair in watson_crick:
            basepairs["Watson-Crick"] += 1
        elif pair in wobble:
            basepairs["Wobble"] += 1
        else:
            basepairs["Non-canonical"] += 1
    
    return basepairs






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

        for file in files:
            try:
                sequence, pairs = read_ct(file)
                pair_list.append(find_basepairs(sequence, pairs))
                progress_bar.update(1)
            except:
                pass
        
        progress_bar.close()

  
        df_pair = pd.DataFrame(pair_list)
        df_pair.to_csv('results/pair_counts.csv')

        below_limit = 0

        #Calculate stats
        for column in df_pair.columns: 
            print(f"Number of {column} basepairs: {df_pair[column].sum()}")

        canoncial = df_pair.loc[:, df_pair.columns != 'Non-canonical']
        
        print("Percentage of basepairs that are non-canonical: ", df_pair["Non-canonical"].sum()/df_pair.sum().sum() *100, "%")
        print("Percentage of sequences with non-canonical basepairs: ", (len(df_pair)-canoncial.apply(lambda row: row.eq(0).all(), axis=1).sum())/len(df_pair)*100, "%")
        
    finally:
        shutil.rmtree(temp_dir)