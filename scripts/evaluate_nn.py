import pickle, torch, os

import pandas as pd

from collections import namedtuple

from utils import model_and_training as model
from utils import evaluation as eval

def evaluate_output(output, sequence): 
    return



if __name__ == "__main__":
    # Load the model
    RNA_Unet = model.RNA_Unet()
    RNA_Unet.load_state_dict(torch.load('PATH')) #FIXME - Add path to model
    
    # Load the data
    RNA = namedtuple('RNA', 'input output length family name sequence')
    file_list = [os.path.join('data', 'test_files', file) for file in os.listdir('data/test_files')]
    
    # Evaluate the model
    columns = ['family', 'length', 'nn', 'argmax', 'blossum_self', 'blossum_weak', 'Mfold', 'hotknots']
    df = pd.DataFrame(index = range(len(file_list)), columns = columns)
    
    for i, file in enumerate(file_list): 
        data = pickle.load(open(file, 'rb'))
        predicted = RNA_Unet(data.input)
        df.loc[i] = [None]*len(columns) #FIXME - Add the correct values
    

    # Save the results
    df.to_csv('results/evalutation_nn.csv', index=False)


    # Plot the results
    