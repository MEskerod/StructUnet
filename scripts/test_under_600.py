import pickle
from tqdm import tqdm
from collections import namedtuple


if __name__ == '__main__': 
    RNA = namedtuple('RNA', 'input output length family name sequence')
    
    #Load data
    files_idx = []


    train = pickle.load(open('data/train.pkl', 'rb'))

    print(f"Total files: {len(train)}")
    print("Finding files under 600")

    progress = tqdm.tqdm(total=len(train), unit = 'files')
    
    for i, file in enumerate(train): 
        length = pickle.load(open(file, 'rb')).length
        if length < 600: 
            files_idx.append(i)
        
        progress.update(1)
        
    progress.close()
    

    print(f"Files under 600: {len(files_idx)}")

    pickle.dump(files_idx, open('data/train_under_600.pkl', 'wb'))
