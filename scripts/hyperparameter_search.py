import torch, os, pickle, logging, sys, time

from collections import namedtuple

from sklearn.model_selection import KFold

import numpy as np

from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F

import utils

from utils import model_and_training as utils

### FUNCTIONS ###
def train_model_on_fold(train_set, val_set, device, parameters, num_epochs):
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

def Kfold_cv(parameters: dict, device, train_set, num_epochs, k=5):
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

def adaptive_hyperparameter_search(train_set, num_epochs, lr_range, weight_decay_range, conv2_filters_range, use_cuda = True, trials = 10, k = 5):
  """
  Function that performs adaptive hyperparameter search for RNAUnet using CuPy (if cuda available)

  Args:
  - train_set: Pytorch training data set
  - val_set: Pytorch validation data set
  - num_epochs: Maximum number of epochs to train the model
  - lr_range: Range og learning rate to search
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

  #Define search space
  params = {
        "lr": lr_range,
        "weight_decay": weight_decay_range,
        "conv2_filters": conv2_filters_range
    }


  for i in range(trials):
    #Get search space for this iteration
    parameters = {'lr': np.random.choice(params["lr"]),
                  'weight_decay': np.random.choice(params["weight_decay"]),
                  'conv2_filters': np.random.choice(params["conv2_filters"])}

    logging.info(f"Start trial number {i+1} with parameters: {parameters}")

    val_loss = Kfold_cv(parameters, device, train_set, num_epochs, k)

    #Update best hyperparameters if applicable
    if val_loss < best_loss:
      best_loss = val_loss
      best_params = parameters
      logging.info("New best hyperparameters found: ")
      logging.info(best_params)

    #Update search space based on performance
    if i > 0:
      if val_loss < prev_val_loss:
        if parameters["lr"] in params["lr"]:
          params["lr"].remove(parameters["lr"])
        if parameters["weight_decay"] in params["weight_decay"]:
          params["weight_decay"].remove(parameters["weight_decay"])
        if parameters["conv2_filters"] in params["conv2_filters"]:
          params["conv2_filters"].remove(parameters["conv2_filters"])

        prev_val_loss = val_loss

    logging.info(f"Trial {i+1} completed. Best hyperparameters found: {best_params} with loss {best_loss}")

  return best_params



### MAIN ###
if __name__ == "__main__":
    logging.basicConfig(filename='hyperparameter_log.txt', level=logging.INFO)
    
    train = pickle.load(open('data/train.pkl', 'rb'))
    
    RNA = namedtuple('RNA', 'input output length family name sequence')

    params = {
       "lr": [0.01, 0.005, 0.001],
       "weight_decay": [0.01, 0.001, 0.0001, 0],
       "conv2_filters": [32, 64],
       }
    
    # Define your train_dataset and validation_dataset
    train_dataset = utils.ImageToImageDataset(train)

    best_params = adaptive_hyperparameter_search(train_dataset, 10, params["lr"], params["weight_decay"], params["conv2_filters"])

    logging.info(f"Best hyperparameters found: {best_params}")
    print(f"Best hyperparameters found: {best_params}", file=sys.stdout)

    