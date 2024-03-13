import torch, os, pickle, logging
import matplotlib.pyplot as plt
from collections import namedtuple

from torch.utils.data import DataLoader

import utils.model_and_training as utils

def show_hisotry(train_history, valid_history, title = None, outputfile = None):
    assert len(train_history) == len(valid_history)
    
    x = list(range(1, len(train_history)+1))
    
    fig, ax = plt.subplots(figsize=(10,6))
    ax.plot(x, train_history, label = 'Training')
    ax.plot(x, valid_history, label = 'Validation')
    ax.set_xlabel('Epoch')
    ax.set_ylabel(title)
    ax.title(title)
    ax.legend()
    ax.grid(axis = 'y', linestyle = '--')
    ax.set_axisbelow(True)
    plt.tight_layout()
    if outputfile:
        plt.savefig(outputfile, bbox_inches = 'tight')
    plt.show()


def fit_model(model, train_dataset, validtion_dataset, lr = 0.01, weigth_decay = 0, optimizer =utils.adam_optimizer, loss_function = utils.dice_loss, epochs = 50, batch_size = 1, log_interval = 10): 
    """
    """
    train_dl = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
    valid_dl = DataLoader(validtion_dataset, batch_size = batch_size)

    opt = optimizer(model, lr, weigth_decay)

    train_loss_history, train_F1_history, valid_loss_history, valid_F1_history = [], [], [], []

    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    
    model.to(device)

    logging.info(f"Training model with {len(train_dl)} training samples and {len(valid_dl)} validation samples. Device: {device}")

    for epoch in range(epochs): 
        running_loss, running_F1 = 0.0, 0.0
        model.train()

        for input, target in train_dl: 
            input, target = input.to(device), target.to(device)
            target = target.unsqueeze(1) #Since the model expects a channel dimension

            #Forward pass
            opt.zero_grad()
            output = model(input)

            loss = loss_function(output, target)
            
            #Backward pass
            loss.backward()
            opt.step()

            running_loss += loss.item()
            running_F1 += utils.f1_score(output, target).item()
        
        #Validation loss (only after each epoch)
        valid_loss, valid_F1 = 0.0, 0.0
        with torch.no_grad():
            for valid_input, valid_target in valid_dl: 
                valid_input, valid_target = valid_input.to(device), valid_target.to(device)
                valid_target = valid_target.unsqueeze(1)

                valid_output = model(valid_input)
                valid_loss += loss_function(valid_output, valid_target).item()
                valid_F1 += utils.f1_score(valid_output, valid_target).item()
        
        train_loss_history.append(running_loss/len(train_dl))
        train_F1_history.append(running_F1/len(train_dl))
        valid_loss_history.append(valid_loss/len(valid_dl))
        valid_F1_history.append(valid_F1/len(valid_dl))

        logging.info(f"Epoch {epoch+1}/{epochs}: Train loss: {train_loss_history[-1]:.4f}, Train F1: {train_F1_history[-1]:.4f}, Validation loss: {valid_loss_history[-1]:.4f}, Validation F1: {valid_F1_history[-1]:.4f}")
        show_hisotry(train_loss_history, valid_loss_history, title = 'Loss', outputfile = 'training_log/loss_history.png')
        show_hisotry(train_F1_history, valid_F1_history, title = 'F1 score', outputfile = 'training_log/F1_history.png')

    return

if __name__ == "__main__":
    os.makedirs('training_log', exist_ok=True)
    logging.basicConfig(filename='training_log/training_log.txt', level=logging.INFO)
    
    train = pickle.load(open('data/train.pkl', 'rb'))
    valid = pickle.load(open('data/val.pkl', 'rb'))
    
    pass