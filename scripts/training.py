import torch, os, pickle, logging, sys

import torch.utils
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from collections import namedtuple

from torch.utils.data import DataLoader

#import utils.model_and_training as utils
from utils.model_and_training import RNA_Unet, adam_optimizer, dice_loss, f1_score, ImageToImageDataset


def show_history(train_history: list, valid_history: list, title = None, outputfile = None) -> None:
    """
    Plots the training and validation history of a model.

    Parameters:
    - train_history (list): List of training history values (loss, F1 score, etc.).
    - valid_history (list): List of validation history values (loss, F1 score, etc.).
    - title (str): Title of the plot. Is also used as label for the y-axis.
    - outputfile (str): Path to save the plot as an image file.

    Returns:
    - None
    """
    assert len(train_history) == len(valid_history)
    
    x = list(range(1, len(train_history)+1))
    
    fig, ax = plt.subplots(figsize=(10,6))
    ax.plot(x, train_history, label = 'Training')
    ax.plot(x, valid_history, label = 'Validation')
    ax.set_xlabel('Epoch')
    ax.set_ylabel(title)
    ax.set_title(title)
    ax.legend()
    ax.grid(axis = 'y', linestyle = '--')
    ax.set_axisbelow(True)
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    plt.tight_layout()
    if outputfile:
        plt.savefig(outputfile, bbox_inches = 'tight', dpi=300)
    plt.show()
    plt.close()

def onehot_to_image(array: np.ndarray) -> np.ndarray:
  """
  Converts a one-hot encoded array with 8 channels to an RGB image.

  Parameters:
  - array (np.ndarray): A 3D NumPy array with shape (height, width, 8).

  Returns:
  - np.ndarray: A 3D NumPy array with shape (height, width, 3) representing an RGB image.
  """
  channel_to_color = {0: [255, 255, 255], #invalid pairing = white
                      1: [64, 64, 64], #unpaired = gray
                      2: [0, 255, 0], #GC = green
                      3: [0, 128, 0], #CG = dark green
                      4: [0, 0, 255], #UG = blue
                      5: [0, 0, 128], #GU = dark blue
                      6: [255, 0, 0], #UA = red
                      7: [128, 0, 0]} #AU = dark red

  rgb_image = np.zeros((array.shape[0], array.shape[1], 3), dtype=np.uint8)

  for channel, color in channel_to_color.items():
    # Select the indices where the channel has non-zero values
    indices = array[:, :, channel] > 0
    # Assign the corresponding color to those indices in the RGB image
    rgb_image[indices] = color

  return rgb_image

def show_matrices(inputs: torch.Tensor, observed: torch.Tensor, predicted: torch.Tensor, treshold=0.5, output_file = None) -> None:
  """
  Plots the input, observed, predicted, and binary predicted matrices side by side.

  Parameters:
  - inputs (torch.Tensor): The input matrix.
  - observed (torch.Tensor): The observed matrix.
  - predicted (torch.Tensor): The predicted matrix.
  - treshold (float): The treshold to use for the binary predicted matrix. Default is 0.5.
  - output_file (str): OPTIONAL. Path to save the plot as an image file. If None, the plot is displayed.

  Returns:
  - None
  """
  fig, axs = plt.subplots(1, 4, figsize=(6,2))
  
  axs[0].imshow(onehot_to_image(inputs.permute(0, 2, 3, 1).squeeze().detach().cpu().numpy()))
  axs[0].set_title("Input")

  axs[1].imshow(observed.permute(0, 2, 3, 1).squeeze().detach().cpu().numpy(), cmap='binary')
  axs[1].set_title("Observed")

  axs[2].imshow(predicted.permute(0, 2, 3, 1).squeeze().detach().cpu().numpy(), cmap='binary')
  axs[2].set_title("Predicted")

  predicted_binary = (predicted.permute(0, 2, 3, 1).squeeze().detach().cpu() >= treshold).float()
  axs[3].imshow(predicted_binary, cmap='binary')
  axs[3].set_title("Predicted binary")

  plt.tight_layout()

  if output_file:
    plt.savefig(output_file, bbox_inches = 'tight')
  else:
    plt.show()

  plt.close()


def fit_model(model: torch.nn.Module, train_dataset, validtion_dataset, patience: int = 5, lr: float = 0.01, weigth_decay: float = 0.0, optimizer =adam_optimizer, loss_function = dice_loss, epochs: int = 60, batch_size: int = 1) -> dict: 
    """
    Trains a model on a training dataset and validates it on a validation dataset.
    Is implemented with early stopping. 
    The best model is saved as 'RNA_Unet.pth'.

    Parameters:
    - model (torch.nn.Module): The model to train.
    - train_dataset (torch.utils.data.Dataset): The training dataset.
    - validtion_dataset (torch.utils.data.Dataset): The validation dataset.
    - patience (int): The number of epochs without improvement before stopping training. Default is 5.
    - lr (float): The learning rate for the optimizer. Default is 0.01.
    - weigth_decay (float): The weight decay for the optimizer. Default is 0.
    - optimizer (function): The optimizer to use. Default is utils.adam_optimizer.
    - loss_function (function): The loss function to use. Default is utils.dice_loss.
    - epochs (int): The number of epochs to train. Default is 60.
    - batch_size (int): The batch size for training. Default is 1.

    Returns:
    - dict: A dictionary with the training history. Keys are 'train_loss', 'train_F1', 'valid_loss', 'valid_F1'.
    """
    best_score = float('inf')
    early_stopping_counter = 0
    
    train_dl = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
    valid_dl = DataLoader(validtion_dataset, batch_size = batch_size)

    opt = optimizer(model, lr, weigth_decay)

    train_loss_history, train_F1_history, valid_loss_history, valid_F1_history = [], [], [], []
    start_epoch = 0

    if os.path.exists('results/training_history.csv'):
        logging.info('Loading previous training history...')
        df = pd.read_csv('results/training_history.csv')
        train_loss_history = df['train_loss'].tolist()
        train_F1_history = df['train_F1'].tolist()
        valid_loss_history = df['valid_loss'].tolist()
        valid_F1_history = df['valid_F1'].tolist()
        start_epoch = len(train_loss_history)
        best_score = min(valid_loss_history)
        early_stopping_counter = len(valid_loss_history) - valid_loss_history.index(best_score) - 1
        logging.info(f'Starting training from epoch {start_epoch+1}.')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model.to(device)

    logging.info(f"Training model with {len(train_dl)} training samples and {len(valid_dl)} validation samples. Device: {device}")

    for epoch in range(start_epoch, epochs): 
        logging.info(f"\nStarting epoch {epoch+1}/{epochs}")
        progress_bar = tqdm(total = len(train_dl), desc = f'Training of epoch {epoch+1}', unit = 'batch')

        running_loss, running_F1 = 0.0, 0.0
        model.train()

        try:
          for input, target in train_dl: 
              input, target = input.to(device), target.unsqueeze(1) #Since the model expects a channel dimension target needs to be unsqueezed

              #Forward pass
              opt.zero_grad()
              output = model(input)

              output = output.cpu()
              loss = loss_function(output, target)
              loss = loss.to(device)
            
              #Backward pass
              loss.backward()
              opt.step()

              running_loss += loss.item()
              running_F1 += f1_score(output, target).item()

              progress_bar.update(1)
        except Exception as e:
          print(e, file=sys.stderr)
          print("Input shape:", input.shape, file=sys.stderr)
          print("Target shape:", target.shape, file=sys.stderr)
          sys.exit(1)    
        
        
        progress_bar.close()
        
        show_matrices(input, target, output, output_file = 'steps/training_log/matrix_example.png')
        
        #Validation loss (only after each epoch)
        logging.info("Start validation...")
        progress_bar = tqdm(total = len(valid_dl), desc = f'Validation of epoch {epoch+1}', unit = 'sequence')
        valid_loss, valid_F1 = 0.0, 0.0
        with torch.no_grad():
            for input, target in valid_dl: 
                input, target = input.to(device), target.unsqueeze(1)

                output = model(input)
                output = output.cpu()
                valid_loss += loss_function(output, target).item()
                valid_F1 += f1_score(output, target).item()
                progress_bar.update(1)
        progress_bar.close()
        
        val_loss = valid_loss/len(valid_dl)
        
        train_loss_history.append(running_loss/len(train_dl))
        train_F1_history.append(running_F1/len(train_dl))
        valid_loss_history.append(val_loss)
        valid_F1_history.append(valid_F1/len(valid_dl))

        logging.info(f"Epoch {epoch+1}/{epochs}: Train loss: {train_loss_history[-1]:.4f}, Train F1: {train_F1_history[-1]:.4f}, Validation loss: {valid_loss_history[-1]:.4f}, Validation F1: {valid_F1_history[-1]:.4f}")
        if epoch > 0:
            show_history(train_loss_history, valid_loss_history, title = 'Loss', outputfile = 'steps/training_log/loss_history.png')
            show_history(train_F1_history, valid_F1_history, title = 'F1 score', outputfile = 'steps/training_log/F1_history.png')
        
        data = {"train_loss": train_loss_history, "train_F1": train_F1_history, "valid_loss": valid_loss_history, "valid_F1": valid_F1_history}
        df = pd.DataFrame(data)
        df.to_csv('results/training_history.csv')
        

        if val_loss < best_score:
           best_score = val_loss
           early_stopping_counter = 0
           #Save model
           torch.save(model.state_dict(), 'RNA_Unet.pth')
        else: 
           early_stopping_counter += 1
        
        logging.info(f'This epoch: {val_loss}. Best: {best_score}. Epochs without improvement {early_stopping_counter}.\n')

        #Check early stopping condition
        if early_stopping_counter >= patience: 
            logging.info(f'EARLY STOPPING TRIGGERED: No improvement in {patience} epochs. Stopping training.')
            break
    
    #Save data to csv
    data = {"train_loss": train_loss_history, "train_F1": train_F1_history, "valid_loss": valid_loss_history, "valid_F1": valid_F1_history}
    df = pd.DataFrame(data)
    df.to_csv('results/training_history.csv')

    return data


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    os.makedirs('steps/training_log', exist_ok=True)
    logging.basicConfig(filename=f'steps/training_log/training_log.txt', level=logging.INFO)
    
    train = pickle.load(open('data/train.pkl', 'rb'))
    valid = pickle.load(open('data/valid.pkl', 'rb'))

    RNA = namedtuple('RNA', 'input output length family name sequence')

    train_dataset = ImageToImageDataset(train)
    valid_dataset = ImageToImageDataset(valid)  

    model = RNA_Unet(channels=32)
    if os.path.exists('RNA_Unet.pth'):
        model.load_state_dict(torch.load('RNA_Unet.pth'))
        logging.info('\nModel loaded from RNA_Unet.pth')

    logging.info(f"Model has {count_parameters(model)} trainable parameters.")

    fit_model(model, train_dataset, valid_dataset)