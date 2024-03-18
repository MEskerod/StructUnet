import torch, pickle, logging, sys, time

import numpy as np

from collections import namedtuple
from sklearn.model_selection import KFold
from itertools import product
from torch.utils.data import DataLoader, Dataset
from utils import model_and_training as utils

### FUNCTIONS ###
def train_model_on_fold(train_set: Dataset, val_set: Dataset, device: str, parameters: dict, num_epochs: int) -> float:
  """
  Function that trains a model on a single fold of the data set and evaluates it on the validation set.

  Parameters:
  - train_set (Dataset): Pytorch dataset containing the training data
  - val_set (Dataset): Pytorch dataset containing the validation data
  - device (str): The device to train the model on (either 'cuda' or 'cpu')
  - parameters (dict): Dictionary containing the hyperparameters to use for training
  - num_epochs (int): The number of epochs to train the model

  Returns:
  - float: The average loss on the validation set after training
  """
  #Define data loaders
  train_fold_loader = DataLoader(train_set, batch_size=1, shuffle=True)
  val_fold_loader = DataLoader(val_set, batch_size=1)

  #Define model
  model = utils.RNA_Unet(channels = parameters["conv2_filters"])
  model.to(device)
  optimizer = utils.adam_optimizer(model, parameters["lr"], parameters["weight_decay"])

  #Train model
  for epoch in range(num_epochs):
    train_loss = 0.0
    for input, output in train_fold_loader:
      input, output = input.to(device), output.to(device)
      output = output.unsqueeze(1)
      optimizer.zero_grad()
      predicted = model(input)
      loss = utils.dice_loss(predicted, output)
      train_loss += loss.item()
      loss.backward()
      optimizer.step()
    logging.info(f"\t\tFinished epoch {epoch+1}/{num_epochs}. Training loss: {train_loss/len(train_fold_loader)}")

  
  #Evaluate model on validation set after final epoch
  #We don't need to do it after every epoch, since we're only using it to compare hyperparameters
  val_loss = 0.0

  logging.info(f"\t\tEvaluating model on validation set")
  with torch.no_grad():
    for input, output in val_fold_loader:
      input, output = input.to(device), output.to(device)
      output = output.unsqueeze(1)
      predicted = model(input)
      val_loss += (utils.dice_loss(predicted, output)).item()
  val_loss = val_loss/len(val_fold_loader)
  return val_loss

def Kfold_cv(parameters: dict, device: str, train_set, num_epochs, k=5):
    start_time = time.time()
    val_losses = 0.0

    #Split data into k folds:
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(train_set)):
        logging.info(f"\tTraining fold {fold+1}/{k}")
        train_fold = torch.utils.data.Subset(train_set, train_idx)
        val_fold = torch.utils.data.Subset(train_set, val_idx)

        val_loss = train_model_on_fold(train_fold, val_fold, device, parameters, num_epochs)

        val_losses += val_loss

        logging.info(f"\tFinished fold {fold+1}/{k}. Validation loss: {val_loss}")
        logging.info(f"\tElapsed time: {time.time() - start_time}")


    #Return average validation loss
    return sum(val_losses)/k

def adaptive_hyperparameter_search(train_set: Dataset, num_epochs: int, weight_decay_range: list, conv2_filters_range: list, use_cuda: bool = True, k: int = 5) -> dict:
  """
  Function that performs adaptive hyperparameter search for RNAUnet

  Parameters:
  - train_set: Pytorch training data set
  - num_epochs: Maximum number of epochs to train the model
  - weight_decay_range: Range of weight decay values to search
  - conv2_filters_range: Range of numbers of filters for the first hidden layer to search
  - use_cuda: Boolean value indicating whether to use cuda if available
  - trials: Number of trials to perform
  - k: Number of folds for cross-validation

  Returns:
  - best_params: Dictionary containing the est hyperparameters found.
  """

  if use_cuda and torch.cuda.is_available():
    device = torch.device("cuda")
  else:
    device = torch.device("cpu")
  
  logging.info(f"Using device: {device}")

  best_loss = float('inf')
  best_params ={}
  
  trials = len(weight_decay_range) * len(conv2_filters_range)

  params = [{'weight_decay': wd, 'conv2_filters': cf} for wd, cf in product(weight_decay_range, conv2_filters_range)]

  for i, parameters in enumerate(trials):
    #Get search space for this iteration

    logging.info(f"Start trial number {i+1} with parameters: {parameters}")

    val_loss = Kfold_cv(parameters, device, train_set, num_epochs, k)

    #Update best hyperparameters if applicable
    if val_loss < best_loss:
      best_loss = val_loss
      best_params = parameters
      logging.info("New best hyperparameters found: ")
      logging.info(best_params)

    logging.info(f"Trial {i+1} completed. Best hyperparameters found: {best_params} with loss {best_loss}")

  return best_params



### MAIN ###
if __name__ == "__main__":
    logging.basicConfig(filename='hyperparameter_log.txt', level=logging.INFO)
    
    train = pickle.load(open('data/train.pkl', 'rb'))
    
    RNA = namedtuple('RNA', 'input output length family name sequence')

    params = {
       "weight_decay": [0.001, 0.0001, 0],
       "conv2_filters": [32, 64],
       }
    
    # Define your train_dataset and validation_dataset
    train_dataset = utils.ImageToImageDataset(train)

    best_params = adaptive_hyperparameter_search(train_dataset, 10, params["weight_decay"], params["conv2_filters"])

    logging.info(f"Best hyperparameters found: {best_params}")
    print(f"Best hyperparameters found: {best_params}", file=sys.stdout)

    