import pickle, torch, os

import pandas as pd

from collections import namedtuple

from utils import post_processing as post_process
from utils.model_and_training import evaluate, RNA_Unet
from utils.prepare_data import update_progress_bar
from utils.plots import violin_plot

from concurrent.futures import ThreadPoolExecutor, as_completed

def evaluate_output(predicted: torch.tensor, target: torch.tensor, sequence):
    results = list(evaluate(predicted, target))
    
    functions = [post_process.argmax_postprocessing, post_process.blossom_postprocessing, post_process.blossom_weak, post_process.Mfold_param_postprocessing]

    with ThreadPoolExecutor() as executor:
        future_to_func = {executor.submit(func, predicted.squeeze(0).detach().numpy(), sequence): func.__name__ for func in functions}
        for future in as_completed(future_to_func):
            results.extend(evaluate(future.result(), target))

    return results



if __name__ == "__main__":
    # Load the model
    model = RNA_Unet()
    model.load_state_dict(torch.load('PATH')) #FIXME - Add path to model
    model.cpu()
    
    # Load the data
    RNA = namedtuple('RNA', 'input output length family name sequence')
    file_list = [os.path.join('data', 'test_files', file) for file in os.listdir('data/test_files')]
    
    funcs = ['Argmax', 'Blossum w/ self-loops', 'Blossum', 'Mfold']
    
    # Evaluate the model
    columns = ['family', 'length', 'nn'] + [f'{name}_{metric}' for name in funcs for metric in ['precision', 'recall', 'f1']] #FIXME - Add HotKnots
    df = pd.DataFrame(index = range(len(file_list)), columns = columns)
    
    for i, file in enumerate(file_list): 
        data = pickle.load(open(file, 'rb'))
        predicted = model(data.input.unsqueeze(0))
        results = [data.family, data.length] + evaluate_output(predicted.squeeze(0), data.output, data.sequence)
        df.loc[i] = results

        if (i+1) % 100 == 0: 
            update_progress_bar(i+1, len(file_list))
    

    # Save the results
    df.to_csv('results/evalutation_nn.csv', index=False)

    # Plot the results
    f1 = df[df.filter(regex='f1').columns]
    f1 = f1.apply(pd.to_numeric, errors='coerce')
    violin_plot(f1, 'Post-processing methods', outputfile='figures/evaluation_nn.png') 


    # Make table with average scores
    results = pd.DataFrame(index = funcs, columns = ['precision', 'recall', 'f1'])
    for func in funcs: 
        results.loc[func] = [df[f"{func}_precision"].mean(), df[f"{func}_recall"].mean(), df[f"{func}_f1"].mean()]



    