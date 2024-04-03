import pickle, sys, os, tqdm

from collections import namedtuple



if __name__ == '__main__':
    lines = []

    RNA = namedtuple('RNA', 'input output length family name sequence')
    
   #Load data
    train = pickle.load(open('data/test.pkl', 'rb'))
    under_600 = pickle.load(open('data/train_under_600.pkl', 'rb'))

   
    progress = tqdm(total=len(train), unit = 'files')
    
    for i in under_600:
        file = train[i]
        name = os.path.basename(file)
        sequence = pickle.load(open(file, 'rb')).sequence

        lines.extend([f'>{name}\n', f'{sequence}\n'])
        progress.update(1)
        
    progress.close()

    print("Writing to file")
    
    with open('input.txt', 'w') as f:
        f.writelines(lines)

