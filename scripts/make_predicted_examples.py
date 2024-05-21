import pickle, os, sys, torch 

from collections import namedtuple

import numpy as np
import pandas as pd

from tqdm import tqdm



from utils.model_and_training import RNA_Unet, evaluate
from utils.post_processing import blossom_weak, prepare_input


def has_pk(pairings: np.ndarray) -> bool:
    """
    Checks if a given pairing has a pseudoknot.

    Parameters:
    - pairings (np.ndarray): The pairing array. Unpaired bases are represented by the index itself.

    Returns:
    - bool: True if the pairing has a pseudoknot, False otherwise.
    """
    for idx in range(len(pairings)):
        i, j = idx, pairings[idx]
        start, end = min(i, j), max(i, j)
        if i==j:
            continue
        if torch.max(pairings[start:end]) > end or torch.min(pairings[start:end]) < start:
            return True
    return False

def output_to_bpseq(output: torch.Tensor, sequence: str) -> str: 
    """
    """
    pairs = torch.nonzero(output)

    bpseq = [[str(i+1), sequence[i], str(0)] for i in range(len(sequence))]

    for row, col in pairs: 
        if not row == col:
            bpseq[row][2] = str(col.item()+1)

    return '\n'.join([' '.join(line) for line in bpseq])


def find_examples() -> dict:
    """
    """
    results = {}


    telomerase, RNAaseP, srp, tmRNA, group_I_intron, tRNA, fiveS_rRNA, sixteeenS_rRNA = False, False, False, False, False, False, False, False
    pseudoknot1, pseudoknot2, pseudoknot3 = False, False, False
    twentythreeS_rRNA, group_II_intron = False, False

    align = pickle.load(open('data/test.pkl', 'rb'))

    i = 0
    while not all([telomerase, RNAaseP, srp, tmRNA, group_I_intron, tRNA, fiveS_rRNA, sixteeenS_rRNA, pseudoknot1, pseudoknot2] ): 
        #Shouldn't be used, but check to ensure infinity loop and out of range indexing
        if i == len(align):
            break
        
        file = pickle.load(open(align[i], 'rb'))

        family = file.family

        i += 1 #Increment the counter to next iteration

        if not telomerase and family == 'telomerase':
            results['telomerase'] = {'sequence': file.sequence, 'output': file.output, 'name': file.name, 'input': file.input, 'family': family}
            telomerase = True
            print('Found telomerase')
            continue

        if not RNAaseP and family == 'RNaseP':
            results['RNAaseP'] = {'sequence': file.sequence, 'output': file.output, 'name': file.name, 'input': file.input, 'family': family}
            RNAaseP = True
            print('Found RNaseP')
            continue

        if not srp and family == 'SRP':
            results['srp'] = {'sequence': file.sequence, 'output': file.output, 'name': file.name, 'input': file.input, 'family': family}
            srp = True
            print('Found SRP')
            continue

        if not tmRNA and family == 'tmRNA':
            results['tmRNA'] = {'sequence': file.sequence, 'output': file.output, 'name': file.name, 'input': file.input, 'family': family}
            tmRNA = True
            print('Found tmRNA')
            continue

        if not group_I_intron and family == 'group_I_intron':
            results['group_I_intron'] = {'sequence': file.sequence, 'output': file.output, 'name': file.name, 'input': file.input, 'family': family}
            group_I_intron = True
            print('Found group I intron') 
            continue

        if not tRNA and family == 'tRNA':
            results['tRNA'] = {'sequence': file.sequence, 'output': file.output, 'name': file.name, 'input': file.input, 'family': family}
            tRNA = True
            print('Found tRNA')
            continue

        if not fiveS_rRNA and family == '5S_rRNA':
            results['5S_rRNA'] = {'sequence': file.sequence, 'output': file.output, 'name': file.name, 'input': file.input, 'family': family}
            fiveS_rRNA = True
            print('Found 5S rRNA')
            continue

        if not sixteeenS_rRNA and family == '16S_rRNA':
            results['16S_rRNA'] = {'sequence': file.sequence, 'output': file.output, 'name': file.name, 'input': file.input, 'family': family}
            sixteeenS_rRNA = True
            print('Found 16S rRNA')
            continue

        #Check if the true structure has pseudoknots
        pk = has_pk(np.argmax(file.output, axis=1))

        if not pseudoknot1 and pk:
            results['pseudoknot1'] = {'sequence': file.sequence, 'output': file.output, 'name': file.name, 'input': file.input, 'family': family}
            pseudoknot1 = True
            print('Found first pseudoknot')
            continue

        if not pseudoknot2 and pk and pseudoknot1 and family != results['pseudoknot1']['family']:
            results['pseudoknot2'] = {'sequence': file.sequence, 'output': file.output, 'name': file.name, 'input': file.input, 'family': family}
            pseudoknot2 = True
            print('Found second pseudoknot')
            continue

        i += 1
    


    archive = pickle.load(open('data/archiveii.pkl', 'rb'))

    i = 0
    while not all([pseudoknot3, twentythreeS_rRNA, group_II_intron]):
        #Shouldn't be used, but check to ensure infinity loop and out of range indexing
        if i == len(archive):
            break
        
        file = pickle.load(open(archive[i], 'rb'))

        family = file.family

        i += 1 #Increment the counter to next iteration

        if not twentythreeS_rRNA and family == '23s':
            results['23S_rRNA'] = {'sequence': file.sequence, 'output': file.output, 'name': file.name, 'input': file.input, 'family': family}
            twentythreeS_rRNA = True
            print('Found 23S rRNA')
            continue

        if not group_II_intron and family == 'grp2':
            results['group_II_intron'] = {'sequence': file.sequence, 'output': file.output, 'name': file.name, 'input': file.input, 'family': family}
            group_II_intron = True
            print('Found group II intron')
            continue

        #Check if the true structure has pseudoknots
        pk = has_pk(np.argmax(file.output, axis=1))

        if not pseudoknot3 and pk and pseudoknot1 and pseudoknot2 and family != results['pseudoknot1']['family'] and family != results['pseudoknot2']['family']:
            results['pseudoknot3'] = {'sequence': file.sequence, 'output': file.output, 'name': file.name, 'input': file.input, 'family': family}
            pseudoknot3 = True
            print('Found third pseudoknot')
            continue

    return results

def predict(examples: dict) -> None: 

    progress_bar = tqdm(total=len(examples), unit='example', desc='Predicting structures', file=sys.stdout)
    
    for _, item in examples.items():
        input = item['input'].unsqueeze(0).to(device)
        sequence = item['sequence']
        output = model(input)
        output = prepare_input(output.squeeze(0).squeeze(0).detach(), sequence, device)
        output = blossom_weak(output, sequence, device)
        item['predicted'] = output
        progress_bar.update(1)
    
    progress_bar.close()
    

def save_examples(examples: dict) -> None:
    """
    Writes the predicted and true structures to a bpseq file.

    Parameters:
    - examples (dict): A dictionary containing the examples to save.

    Returns:
    - None
    """
    for family, example in examples.items(): 
        with open(f'steps/examples/{family}_predicted.bpseq', 'w') as file: 
            file.write(output_to_bpseq(example['predicted'], example['sequence']))
        
        with open(f'steps/examples/{family}_true.bpseq', 'w') as file: 
            file.write(output_to_bpseq(example['output'], example['sequence']))



def record_f1_scores(examples: dict) -> None:
    scores = []

    progress_bar = tqdm(total=len(examples), unit='example', desc='Recording F1 scores', file=sys.stdout)
    
    for family, example in examples.items(): 
        precision, recall, F1 = evaluate(example['predicted'], example['output'], device=device)
        scores.append({'example': family, 'family': example['family'], 'length': len(example['sequence']), 'precision': precision, 'recall': recall, 'F1': F1, 'file': example['name']})
        progress_bar.update(1)

    progress_bar.close()

    
    df = pd.DataFrame(scores)
    df.to_csv('results/scores_examples.csv', index=False)



if __name__ == '__main__':
    RNA = namedtuple('RNA', 'input output length family name sequence')

    print('Finding examples')
    examples = find_examples()

    print('\nLoading model')
    # Load the model
    device = 'cpu'
    model = RNA_Unet(channels=32).to(device)
    model.load_state_dict(torch.load('RNA_Unet.pth', map_location=torch.device(device)))
    model.to(device)

    print('\n')
    predict(examples)

    print('\nSaving examples\n')
    os.makedirs('steps/examples', exist_ok=True)
    save_examples(examples) 
    record_f1_scores(examples)
    




