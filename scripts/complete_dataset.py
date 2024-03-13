import os, tempfile, shutil, tarfile, pickle, sys, multiprocessing

from collections import namedtuple
from tqdm import tqdm
from functools import partial

from utils import prepare_data
from utils import plots

def get_folder_size(folder: str) -> float:
    """
    Calculate the size of a folder in GB

    Parameters:
    - folder (str): The path to the folder to calculate the size of.

    Returns:
    float: The size of the folder in GB.
    """
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(folder):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)
    
    gb = total_size/1024/1024/1024
    return gb

def getFamily(file_name: str) -> str:
  '''
  Returns the family of a file in the RNAStralign data set, based on folder structure

  Parameters:
  - file_name (str): The path to the file to extract the family from.
  
  Returns:
  str: The family of the file.

  '''
  return '_'.join(file_name.split(os.sep)[5].split('_')[:-1])


def process_file(file: str, output_folder: str):
    """
    Converts a file from the RNAStralign data set to a pickle file containing a namedtuple with the data.
    The data in the named tuple is the input and output matrices, the length of the sequence, the family and the name of the file.
    Output is saved in the output folder.

    Parameters:
    - file (str): The path to the file to process.
    - output_folder (str): The path to the folder to save the output to.

    Returns:
    int: 1 if the file was processed successfully, None if an error occurred.
    """
    try: 
        length = prepare_data.getLength(file)
        if length == 0:
            return
        
        family = getFamily(file)
        sequence, pairs = prepare_data.read_ct(file)
        input_matrix = prepare_data.make_matrix_from_sequence_8(sequence)
        output_matrix = prepare_data.make_matrix_from_basepairs(pairs)

        if input_matrix.shape[-1] == 0 or output_matrix.shape[-1] == 0:
            return
        
        sample = RNA(input = input_matrix,
                            output = output_matrix,
                            length = length,
                            family = family,
                            name = file,
                            sequence = sequence)    
        
        pickle.dump(sample, open(os.path.join(output_folder, os.path.splitext(os.path.basename(file))[0] + '.pkl'), 'wb'))
        return 1

    except Exception as e: 
        # Skip this file if an unexpected error occurs during processing
        print(f"Skipping {file} due to unexpected error: {e}", file=sys.stderr)
        return


def process_and_save(file_list: list, output_folder: str) -> None: 
    """
    Process a list of files from the RNAStralign data set and save the output to a folder.
    The conversion is done in parallel using multiprocessing.

    Parameters:
    - file_list (list): A list of file paths to process.
    - output_folder (str): The path to the folder to save the output to.

    Returns:
    - None
    """
    total_files = len(file_list)

    os.makedirs(output_folder, exist_ok=True)

    partial_process_file = partial(process_file, output_folder=output_folder)
    
    with multiprocessing.Pool() as pool:
        #Map the process_file function to the list of files
        results = list(tqdm(pool.imap_unordered(partial_process_file, file_list), total=total_files, file=sys.stdout, desc="Processing files"))

    converted = sum([result for result in results if result is not None])  
    
    print(f"\n\n{converted} files converted", file=sys.stdout)

if __name__ == "__main__": 
    print("RUN", file=sys.stdout)
        
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

    file_list = [os.path.join('data', 'complete_set', file) for file in os.listdir('data/complete_set')]

    train, valid, test = prepare_data.split_data(file_list, validation_ratio=0.2, test_ratio=0.0)

    pickle.dump(train, open('data/train.pkl', 'wb'))
    pickle.dump(valid, open('data/valid.pkl', 'wb'))
    pickle.dump(test, open('data/test.pkl', 'wb'))

    print("Move test files to different folder")
    os.makedirs("data/test_files", exist_ok=True)
    for file in test: 
        shutil.move(file, "data/test_files")

    print(f"FILES FOR TRAINING: {len(train)} for training, {len(valid)} for validation", file=sys.stdout)

    print("\nPlotting data distribution", file=sys.stdout)
        
    plots.plot_families({"train":train, "valid":valid, "test":test}, output_file='figures/family_distribution.png')
    plots.plot_len_histogram({"train":train, "valid":valid, "test":test}, output_file='figures/length_distribution.png')

    print("DONE", file=sys.stdout)
    print(f"Size of folder for training: {get_folder_size('data/complete_set'):.2f} GB", file=sys.stdout)
    print(f"Size of folder for testing: {get_folder_size('data/test_files'):.2f} GB", file=sys.stdout)
     
