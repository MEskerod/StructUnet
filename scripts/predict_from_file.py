import pickle, sys, os, torch

from tqdm import tqdm

from collections import namedtuple

from utils.model_and_training import RNA_Unet
from utils.post_processing import prepare_input, blossom_weak

if __name__ == '__main__': 
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    RNA = namedtuple('RNA', 'input output length family name sequence')

    file = sys.argv[1]

    print('-- Loading model and data --')
    model = RNA_Unet(channels=32)
    model.load_state_dict(torch.load('RNA_Unet.pth', map_location=torch.device(device)))
    model.to(device)

    data = pickle.load(open(file, 'rb'))
    print('-- Model and data loaded --\n')

    os.makedirs('results_RNAUnet', exist_ok=True)

    print('-- Predicting --')

    progress_bar = tqdm(total=len(data), unit='sequence')
    
    for file in data:
        #Prepare data
        name = os.path.basename(file)
        file_data = pickle.load(open(file, 'rb'))
        sequence = file_data.sequence
        input = file_data.input.unsqueeze(0).to(device)
        file_data = None #Clear memory

        #Predict
        output = model(input)

        #Post-process
        output = prepare_input(output.squeeze(0).squeeze(0).detach(), sequence, device)
        output = blossom_weak(output, sequence, device)

        #Save results
        pickle.dump(output, open(f'results_RNAUnet/{name}', 'wb'))

        progress_bar.update(1)
    
    progress_bar.close()

    print('-- Prediction done --')
