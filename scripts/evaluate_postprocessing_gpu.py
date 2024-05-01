import pickle, torch, multiprocessing

import pandas as pd

from tqdm import tqdm
from collections import namedtuple

from utils import post_processing as post_process
from utils.model_and_training import evaluate, RNA_Unet
from utils.plots import violin_plot

from concurrent.futures import ThreadPoolExecutor, as_completed

def evaluate_output(file: str, treshold: float = 0.5) -> list:
    """
    Evaluate the output of the model using different post-processing methods.
    Runs the evaluation in parallel using a ThreadPoolExecutor.

    Args:
    - predicted (torch.Tensor): The predicted output of the model.
    - target (torch.Tensor): The target output.
    - sequence (str): The RNA sequence.

    Returns:
    - results (list): The results of the evaluation.
    """
    data = pickle.load(open(file, 'rb'))
    
    with torch.no_grad():
        predicted = model(data.input.unsqueeze(0)).squeeze(0).squeeze(0).detach()
    
    target = data.output
    sequence = data.sequence
    
    results = list(evaluate((predicted >= treshold).float(), target, device=device)) #Evaluate binart raw output

    predicted = post_process.prepare_input(predicted, sequence, device)

    results.extend(list(evaluate((predicted >= treshold).float(), target, device=device))) #Evaluate binary masked output
    
    functions = [post_process.argmax_postprocessing, post_process.blossom_postprocessing, post_process.blossom_weak, post_process.Mfold_param_postprocessing]

    for func in functions:
        results.extend(evaluate(func(predicted, sequence, device), target, device))
    
    return [data.family, data.length] + results



if __name__ == "__main__":
    print("--- Starting evaluation ---")

    print("--- Loading model and data ---")
    device = 'gpu' if torch.cuda.is_available() else 'cpu' 
    print(f"Using {device} device")  
    # Load the model
    model = RNA_Unet(channels=32)
    model.load_state_dict(torch.load('RNA_Unet.pth', map_location=device))
    
    # Load the data
    RNA = namedtuple('RNA', 'input output length family name sequence')
    file_list = pickle.load(open('data/valid.pkl', 'rb'))
    
    funcs = ['No post-processing', 'Only mask', 'Argmax', 'Blossum w/ self-loops', 'Blossum', 'Mfold']
    
    # Evaluate the model
    columns = ['family', 'length'] + [f'{name}_{metric}' for name in funcs for metric in ['precision', 'recall', 'f1']] 
    df = pd.DataFrame(index = range(len(file_list)), columns = columns)
    
    print("--- Evaluating ---")

    progress_bar  = tqdm(total=len(file_list), unit='files')

    for i, file in enumerate(file_list):
        results = evaluate_output(file)
        df.loc[i] = results
        progress_bar.update(1)

    progress_bar.close()


    print("--- Evaluation done ---")
    print("--- Saving results ---")
    
    # Save the results
    df.to_csv('results/evalutation_postprocess_gpu.csv', index=False)

    # Plot the results
    f1 = df[df.filter(regex='f1').columns]
    f1 = f1.apply(pd.to_numeric, errors='coerce')
    violin_plot(f1, 'Post-processing methods', outputfile='figures/evaluation_postprocess_gpu.png') 


    # Make table with average scores
    results = pd.DataFrame(index = funcs, columns = ['precision', 'recall', 'f1'])
    for func in funcs: 
        results.loc[func] = [df[f"{func}_precision"].mean(), df[f"{func}_recall"].mean(), df[f"{func}_f1"].mean()]
    
    results.to_csv('results/average_scores_postprocess_gpu.csv')

    print("--- Results saved ---")




    