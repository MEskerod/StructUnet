import os, pickle

from collections import namedtuple

from utils.model_and_training import evaluate


methods600 = ['Ufold', 'hotknots']
methods = []


under600 = pickle.load(open('data/under600.pkl', 'rb'))
test = pickle.load(open('data/test.pkl', 'rb'))

for method in methods600: 
    assert all(test[i] in os.listdir(f'steps/{method}') for i in under600)
    
    results = [evaluate(pickle.load(test[i].replace('data/test_set', f'steps/{method}')), pickle.load(test[i]).output) for i in under600]

    #TODO - Add some way of saving the results 
    # NOTE - Maybe change the for loop to be pr. file instead of pr. method


#TODO - Add evaluation of other methods


if __name__ == "__main__":
    RNA = namedtuple('RNA', 'input output length family name sequence')
        
