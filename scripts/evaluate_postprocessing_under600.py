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
    
    results = list(evaluate((predicted >= treshold).float(), target, device=device)) #Evaluate binary raw output

    predicted = post_process.prepare_input(predicted, sequence, device)

    results.extend(list(evaluate((predicted >= treshold).float(), target, device=device))) #Evaluate binary masked output
    
    functions = [post_process.argmax_postprocessing, post_process.blossom_postprocessing, post_process.blossom_weak, post_process.Mfold_param_postprocessing]

    for func in functions:
        results.extend(evaluate(func(predicted, sequence, device), target, device))
    
    return [data.family, data.length] + results



if __name__ == "__main__":
    print("--- Starting evaluation ---")

    print("--- Loading model and data ---")
    device = torch.device('cpu')
    # Load the model
    model = RNA_Unet(channels=32)
    model.load_state_dict(torch.load('RNA_Unet.pth', map_location=device))
    
    # Load the data
    RNA = namedtuple('RNA', 'input output length family name sequence')
    file_list = pickle.load(open('data/valid_under_600.pkl', 'rb'))
    
    funcs = ['No post-processing', 'Only mask', 'Argmax', 'Blossum w/ self-loops', 'Blossum', 'Mfold']
    
    # Evaluate the model
    columns = ['family', 'length'] + [f'{name}_{metric}' for name in funcs for metric in ['precision', 'recall', 'f1']] 
    df = pd.DataFrame(index = range(len(file_list)), columns = columns)
    
    print("--- Evaluating ---")

    num_processes = 1
    print(f"Number of processes: {num_processes}")
    pool = multiprocessing.Pool(num_processes)
    shared_counter = multiprocessing.Value('i', 0)
    
    #Run processes
    with tqdm(total=len(file_list)) as pbar:
        for i, result in enumerate(pool.imap_unordered(evaluate_output, file_list)):
            df.loc[i] = result
            with shared_counter.get_lock():
                shared_counter.value += 1
            pbar.update()
    
    #Close the pool
    pool.close()
    pool.join()

    print("--- Evaluation done ---")
    print("--- Saving results ---")
    
    # Save the results
    df.to_csv('results/evalutation_postprocess_under600.csv', index=False)

    # Plot the results
    f1 = df[df.filter(regex='f1').columns]
    f1 = f1.apply(pd.to_numeric, errors='coerce')
    violin_plot(f1, 'Post-processing methods', outputfile='figures/evaluation_postprocess_under600.png') 


    # Make table with average scores
    results = pd.DataFrame(index = funcs, columns = ['precision', 'recall', 'f1'])
    for func in funcs: 
        results.loc[func] = [df[f"{func}_precision"].mean(), df[f"{func}_recall"].mean(), df[f"{func}_f1"].mean()]
    
    results.to_csv('results/average_scores_postprocess_under600.csv')

    print("--- Results saved ---")




    