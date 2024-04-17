import pickle, torch

import pandas as pd

from tqdm import tqdm
from collections import namedtuple

from utils import post_processing as post_process
from utils.model_and_training import evaluate, RNA_Unet
from utils.prepare_data import update_progress_bar
from utils.plots import violin_plot

from concurrent.futures import ThreadPoolExecutor, as_completed

def evaluate_output(predicted: torch.tensor, target: torch.tensor, sequence: str) -> list:
    """
    Evaluate the output of the model using different post-processing methods.
    Runs the evaluation in parallel using a ThreadPoolExecutor.

    Args:
    - predicted (torch.tensor): The predicted output of the model.
    - target (torch.tensor): The target output.
    - sequence (str): The RNA sequence.

    Returns:
    - results (list): The results of the evaluation.
    """
    #Convert to numpy
    predicted, target = predicted.squeeze(0).detach().numpy(), target.numpy()
    
    results = list(evaluate(predicted, target))
    
    functions = [post_process.argmax_postprocessing, post_process.blossom_postprocessing, post_process.blossom_weak, post_process.Mfold_param_postprocessing]

    with ThreadPoolExecutor() as executor:
        future_to_func = {executor.submit(func, predicted, sequence): func.__name__ for func in functions}
        for future in as_completed(future_to_func):
            results.extend(evaluate(future.result(), target))

    return results



if __name__ == "__main__":
    # Load the model
    model = RNA_Unet()
    model.load_state_dict(torch.load('RNA_Unet.pth'))
    model.cpu()
    
    # Load the data
    RNA = namedtuple('RNA', 'input output length family name sequence')
    file_list = pickle.load(open('data/test.pkl', 'rb'))
    
    funcs = ['Argmax', 'Blossum w/ self-loops', 'Blossum', 'Mfold']
    
    # Evaluate the model
    columns = ['family', 'length', 'nn'] + [f'{name}_{metric}' for name in funcs for metric in ['precision', 'recall', 'f1']] #FIXME - Add HotKnots
    df = pd.DataFrame(index = range(len(file_list)), columns = columns)
    
    progess_bar = tqdm(total=len(file_list), unit='files')
    
    for i, file in enumerate(file_list): 
        data = pickle.load(open(file, 'rb'))
        predicted = model(data.input.unsqueeze(0))
        results = [data.family, data.length] + evaluate_output(predicted.squeeze(0), data.output, data.sequence) #Evaluate using all methods
        df.loc[i] = results

        progess_bar.update(1)
    
    progess_bar.close()
    

    # Save the results
    df.to_csv('results/evalutation_postprocess.csv', index=False)

    # Plot the results
    f1 = df[df.filter(regex='f1').columns]
    f1 = f1.apply(pd.to_numeric, errors='coerce')
    violin_plot(f1, 'Post-processing methods', outputfile='figures/evaluation_postprocess.png') 


    # Make table with average scores
    results = pd.DataFrame(index = funcs, columns = ['precision', 'recall', 'f1'])
    for func in funcs: 
        results.loc[func] = [df[f"{func}_precision"].mean(), df[f"{func}_recall"].mean(), df[f"{func}_f1"].mean()]
    
    results.to_csv('results/average_scores_postprocess.csv')



    