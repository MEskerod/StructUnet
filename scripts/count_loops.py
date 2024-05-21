import tempfile, tarfile, sys, shutil

import pandas as pd

from tqdm import tqdm

from utils.prepare_data import read_ct, list_all_files

def count_loops(pairs: list, max_loop_length: int = 2) -> dict:
    """
    Count the number of hairpin loops in a given structure. 
    A hairpin loop is a loop that is closed by a single pair of bases.
    It returns a dictionary containing the number of loops of each length up to max_loop_length.

    Parameters:
    - pairs (list): The list of pairs in the structure.
    - max_loop_length (int): The maximum length of a hairpin loop. Default is 2.

    Returns:
    - dict: A dictionary containing the number of loops of each length up to max_loop_length.
    """
    loops = {i: 0 for i in range(max_loop_length+1)}
    loops[f'>{max_loop_length}'] = 0
    for i, j in enumerate(pairs):
        #Remove cases that doesn't pair or where j > i, to avoid double counting
        if j == None or j < i:
            continue
        
        #Check that it is the closing pair of a hairpin loop
        if all(val is None for val in pairs[i+1:j]):    
            dist = j-i-1
            if dist > max_loop_length:
                loops[f'>{max_loop_length}'] += 1
            else:
                loops[dist] += 1
    return loops

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

        loop_list = []

        for file in files:
            _, pairs = read_ct(file)
            loop_list.append(count_loops(pairs))
            progress_bar.update(1)
        
        progress_bar.close()

        df = pd.DataFrame(loop_list)
        df.to_csv('results/loop_counts.csv')

        below_limit = 0

        #Calculate stats
        for column in df.columns:
            print(f"Number of loops of length {column}: {df[column].sum()}")
            if column != '>2':
                below_limit += df[column].sum()

        selected_columns = df.loc[:, df.columns != '>2']

        print("Percentage of loops below limit: ", below_limit/df.sum().sum() *100, "%")
        print("Percentage of sequences with loops below limit: ", (len(df)-selected_columns.apply(lambda row: row.eq(0).all(), axis=1).sum())/len(df)*100, "%" )
        
    finally:
        shutil.rmtree(temp_dir)