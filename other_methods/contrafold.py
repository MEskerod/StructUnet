import pickle, subprocess, time, datetime, os, tempfile, torch, sys

from tqdm import tqdm

from collections import namedtuple

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

def contrafold(file: str) -> str:
    """
    Calls the CONTRAfold executable to predict the secondary structure of an RNA sequence.

    Parameters:
    - file (str): The path to the file containing the RNA sequence.

    Returns:
    - str: The predicted secondary structure in dot-bracket notation.
    """ 
    command = ['../contrafold/src/contrafold', 'predict', file]

    try:
        p = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        p.wait()
        output, error = p.communicate()

        if p.returncode == 0: #Check that command succeeded
            result = output.decode().split() 
            return result[3]
        else:
            error_msg = error.decode() if error else 'Unknown error'
            raise Exception(f'CONTRAfold execution failed: {error_msg}')
    
    except Exception as e:
        raise Exception(f'An error occured: {e}')

    

def dot_bracket_to_matrix(db: str) -> torch.Tensor: 
    """
    Takes a pseudoknot-free dot-bracket notation and converts it to a matrix representation.

    Parameters:
    - db (str): The dot-bracket notation.

    Returns:
    - torch.Tensor: The matrix representation of the dot-bracket notation.
    """
    matrix = torch.zeros(len(db), len(db))
    
    stack1 = []

    bp = [i for i in range(len(db))]
    
    for i, char in enumerate(db):
        if char == '(':
            stack1.append(i)
        elif char == ')':
            j = stack1.pop()
            bp[i] = j
            bp[j] = i
    
    for i, j in enumerate(bp):
        matrix[i, j] = 1
        
    return matrix   



if __name__ == '__main__': 
    
    print('--- Loading data ---\n')

    RNA = namedtuple('RNA', 'input output length family name sequence')
    files = pickle.load(open('data/test.pkl', 'rb'))

    os.makedirs('steps/contrafold', exist_ok=True)

    total_time = 0
    
    print('--- Starting predicting ---')

    progress_bar = tqdm(total=len(files), unit='sequence', file=sys.stdout)
    for file in files:
        name = os.path.basename(file)
        #Make temporary file for sequence, since CONTRAfold requires a file
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
            sequence = pickle.load(open(file, 'rb')).sequence
            temp_file.write(sequence)
        
        #Predict structure, save it and measure time
        try: 
            start_time = time.time() 
            structure = contrafold(temp_file.name)
            structure = dot_bracket_to_matrix(structure)
            pickle.dump(structure, open(f'steps/contrafold/{name}', 'wb'))
            total_time += (time.time() - start_time)
        
        finally:
            #Remove temporary file
            os.unlink(temp_file.name) 
        progress_bar.update(1)
    
    progress_bar.close()

    print(f'Prediction done in {format_time(total_time)}. Average time per sequence: {total_time/len(files)}')
