import pickle
from tqdm import tqdm
from collections import namedtuple


if __name__ == '__main__': 
    """
    Make a list of all the files in the test set that have a length under 600 and save it as a pickle file
    Is used later to evaluate some methods
    """
    RNA = namedtuple('RNA', 'input output length family name sequence')
    
    #Load data
    files_idx = []


    test = pickle.load(open('data/test.pkl', 'rb'))

    print(f"Total files: {len(test)}")
    print("Finding files under 600")

    progress = tqdm(total=len(test), unit = 'files')
    
    for i, file in enumerate(test): 
        length = pickle.load(open(file, 'rb')).length
        if length < 600: 
            files_idx.append(i)
        
        progress.update(1)
        
    progress.close()
    

    print(f"Files under 600: {len(files_idx)}")
    print(f"Files over 600: {len(test) - len(files_idx)}")

    pickle.dump(files_idx, open('data/test_under_600.pkl', 'wb'))
