import pickle, os, time, subprocess, datetime, torch
from collections import namedtuple
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt

def format_time(seconds: float) -> str:
    """
    Format a time duration in seconds to hh:mm:ss format.
    
    Parameters:
    seconds: Time duration in seconds.
    
    Returns:
    Formatted time string in hh:mm:ss format.
    """
    time_delta = datetime.timedelta(seconds=seconds)
    hours, remainder = divmod(time_delta.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return '{:02d}:{:02d}:{:02d}'.format(hours, minutes, seconds)


def run_nussinov(sequence: str) -> list:
    """
    Calls the Nussinov executable to predict the secondary structure of an RNA sequence.

    Parameters:
    - sequence (str): The RNA sequence.

    Returns:
    - list: The predicted secondary structure in dot-bracket notation.
    """
    command = ['other_methods/nussinov', sequence]

    try:
        p = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        p.wait()
        output, error = p.communicate()

        if p.returncode == 0: #Check that command succeeded
            result = output.decode().split()
            return result
        else:
            error_msg = error.decode() if error else 'Unknown error'
            raise Exception(f'Simfold execution failed: {error_msg}')
    
    except Exception as e:
        raise Exception(f'An error occured: {e}')


def make_matrix_from_basepairs(pairs: list) -> torch.Tensor:
	"""
    Takes a list of all which base each position in the sequence is paired with. If a base is unpaired pairs[i] = i.
    From the list a 2D matrix is made, with each cell coresponding to a base pair encoded as 1 and unpaired bases encoded as 1 at the diagonal

    Parameters:
    - pairs (list): A list of integers representing the pairing state of each base.

    Returns:
    - torch.Tensor: A 2D tensor with shape (len(pairs), len(pairs)).
    """

	N = len(pairs)
     
	matrix = torch.zeros((N,N), dtype=torch.float32)

	for i, p in enumerate(pairs):
		matrix[i, int(p)] = 1 

	return matrix

def plot_time(time, lengths):
    """
    Plots the time it takes to predict the structure of a sequence using the Nussinov algorithm

    Parameters:
    - time (list): A list of floats representing the time it took to predict the structure of each sequence.
    - lengths (list): A list of integers representing the length of each sequence.
    """

    plt.scatter(lengths, time, facecolor='none', edgecolor = 'C0', s=20, linewidths = 1)
    plt.plot(lengths, time, linestyle = '--', linewidth = 0.8)
    plt.xlabel('Sequence length')
    plt.ylabel('Time (s)')
    plt.grid(linestyle='--')
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.tight_layout
    plt.savefig('figures/times_nussinov.png', dpi=300, bbox_inches='tight')


def main() -> None: 
    
    os.makedirs('steps/nussinov', exist_ok=True)
    test = pickle.load(open('data/test.pkl', 'rb'))

    print(f"Predicting structure using Nussinov algorithm.\n Total: {len(test)} sequences.")
    print('-- Predicting --')

    times = [[]*len(test)]
    lengths = []

    progress_bar = tqdm(total=len(test)*3, unit='sequence')

    start_time = time.time()

    #Predict for all sequences in test set 3 times and save the time it took
    for _ in range(3):
        lengths = []
        for i in range(len(test)): 
            sequence = pickle.load(open(test[i], 'rb')).sequence
            name = os.path.join('steps', 'nussinov', os.path.basename(test[i])) 
            start = time.time()
            output = make_matrix_from_basepairs(run_nussinov(sequence))
            times[i].append(time.time()-start)
            if i == 0:
                pickle.dump(output, open(name, 'wb'))
                lengths.append(len(sequence))
            progress_bar.update(1)
    
    #Calculate average time for each sequence
    times = [sum(t)/len(t) for t in times]
    
    progress_bar.close()
    
    total_time = (time.time()-start_time)/3

    print('-- Predictions done --')

    print(f'Total time: {format_time(total_time)}. Average prediction time: {total_time/len(test):.5f}')

    print('-- Plot and save times --')
    data = {'lengths': lengths, 'times': times}
    df = pd.DataFrame(data)
    df = df.sort_values('lengths')
    df.to_csv('results/times_nussinov.csv', index=False)
    plot_time(df['times'].tolist(), df['lengths'].tolist())

   
if __name__ == '__main__': 
    RNA = namedtuple('RNA', 'input output length family name sequence')
    main()