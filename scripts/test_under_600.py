import pickle
from collections import namedtuple


if __name__ == '__main__': 
    RNA = namedtuple('RNA', 'input output length family name sequence')
    
    #Load data
    files_idx = []


    train = pickle.load(open('data/train.pkl', 'rb'))

    for i, file in enumerate(train): 
        length = pickle.load(open(file, 'rb')).length
        if length < 600: 
            files_idx.append(i)
    

    print(f"Files under 600: {len(files_idx)}")

    pickle.dump(files_idx, open('data/train_under_600.pkl', 'wb'))
