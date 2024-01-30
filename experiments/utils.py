import torch, pickle

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import random as rd
import numpy as np
import matplotlib.pyplot as plt

from collections import defaultdict, namedtuple

RNA_data = namedtuple('RNA_data', 'input output length family name pairs')


def file_length(file):
  return int(pickle.load(open(file, 'rb')).length)

def get_indices(ratios): 
  rd.seed(42)
  
  numbers = list(range(10))
  rd.shuffle(numbers)

  count1 = int(10 * ratios[0])
  count2 = int(10 * ratios[1])

  group1 = numbers[:count1]
  group2 = numbers[count1: count1+count2]
  group3 = numbers[count1+count2:]

  return group1, group2, group3

def split_data(file_list, train_ratio = 0.8, validation_ratio = 0.1, test_ratio = 0.1, input_path = "input", output_path = "output"):
  train_indices, valid_indices, test_indices = get_indices([train_ratio, validation_ratio, test_ratio])
  
  train, valid, test = [], [], []

  family_data = defaultdict(list)
  for file in file_list:
    family = pickle.load(open(file, 'rb')).family
    family_data[family].append((file))

  for family, files in family_data.items():
    N = len(files)
    files.sort(key=file_length)

    for i in range(0, N, 10): 
      train_idx = [n+i for n in train_indices]
      valid_idx = [n+i for n in valid_indices]
      test_idx = [n+i for n in test_indices]

      train.extend([files[i] for i in train_idx if i < N])
      valid.extend(files[i] for i in valid_idx if i < N)
      test.extend(files[i] for i in test_idx if i < N)

  return train, valid, test

class ImageToImageDataset(Dataset):
    """

    """
    def __init__(self, file_list):
        self.file_list = file_list
        self.family_map = {'16SrRNA': torch.tensor([1, 0, 0, 0, 0, 0, 0, 0], dtype = torch.float32),
                    '5SrRNA': torch.tensor([0, 1, 0, 0, 0, 0, 0, 0], dtype = torch.float32),
                    'RNaseP': torch.tensor([0, 0, 1, 0, 0, 0, 0, 0], dtype = torch.float32),
                    'SRP': torch.tensor([0, 0, 0, 1, 0, 0, 0, 0], dtype = torch.float32),
                    'groupIintron': torch.tensor([0, 0, 0, 0, 1, 0, 0, 0], dtype = torch.float32),
                    'tRNA': torch.tensor([0, 0, 0, 0, 0, 1, 0, 0], dtype = torch.float32),
                    'telomerase': torch.tensor([0, 0, 0, 0, 0, 0, 1, 0], dtype = torch.float32),
                    'tmRNA': torch.tensor([0, 0, 0, 0, 0, 0, 0, 1], dtype = torch.float32)}

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
      data = pickle.load(open(self.file_list[idx], 'rb'))
      
      input_image = data.input
      output_image = data.output

      family = data.family
      label = self.family_map[family]

      return input_image, output_image, label


### MODEL RELATED ###
class DynamicPadLayer(nn.Module):
  def __init__(self, stride_product):
    super(DynamicPadLayer, self).__init__()
    self.stride_product = stride_product

  def forward(self, x):
    input_size = x.shape[2]
    padding = self.calculate_padding(input_size, self.stride_product)
    return nn.functional.pad(x, padding)

  def calculate_padding(self, input_size, stride_product):
    p = stride_product - input_size % stride_product
    return (0, p, 0, p)

class MaxPooling(nn.Module):
  """
  Layer for max pooling
  """
  def __init__(self, in_channels, out_channels, kernel_size=2, stride=2, padding=0):
    super(MaxPooling, self).__init__()
    self.max_pool = nn.MaxPool2d(kernel_size = kernel_size, stride = stride)

  def forward(self, x):
    return self.max_pool(x)

class RNA_Unet(nn.Module):
    def __init__(self, channels=32, in_channels=8, output_channels=1, negative_slope = 0.0, pooling = MaxPooling):
        """
        args:
        num_channels: length of the one-hot encoding vector
        num_hidden_channels: number of channels in the hidden layers of both encoder and decoder
        """
        super(RNA_Unet, self).__init__()

        self.negative_slope = negative_slope

        self.pad = DynamicPadLayer(2**4)

        # Encoder
        self.bn11 = nn.BatchNorm2d(channels)
        self.e11 = nn.Conv2d(in_channels, channels, kernel_size = 3, padding = 1)
        self.pool1 = pooling(channels, channels, kernel_size=2, stride=2)

        self.bn21 = nn.BatchNorm2d(channels * 2)
        self.e21 = nn.Conv2d(channels, channels*2, kernel_size=3, padding=1)
        self.pool2 = pooling(channels*2, channels*2, kernel_size=2, stride=2)

        self.bn31 = nn.BatchNorm2d(channels*4)
        self.e31 = nn.Conv2d(channels*2, channels*4, kernel_size=3, padding=1)
        self.pool3 = pooling(channels*4, channels*4, kernel_size=2, stride=2)

        self.bn41 = nn.BatchNorm2d(channels*8)
        self.e41 = nn.Conv2d(channels*4, channels*8, kernel_size=3, padding=1)
        self.pool4 = pooling(channels*8, channels*8, kernel_size=2, stride=2)

        self.bn51 = nn.BatchNorm2d(channels*16)
        self.e51 = nn.Conv2d(channels*8, channels*16, kernel_size=3, padding=1)

        #Decoder
        self.bn61 = nn.BatchNorm2d(channels*8)
        self.upconv1 = nn.ConvTranspose2d(channels*16, channels*8, kernel_size=2, stride=2)
        self.bn62 = nn.BatchNorm2d(channels*8)
        self.d11 = nn.Conv2d(channels*16, channels*8, kernel_size=3, padding=1)

        self.bn71 = nn.BatchNorm2d(channels*4)
        self.upconv2 = nn.ConvTranspose2d(channels*8, channels*4, kernel_size=2, stride=2)
        self.bn72 = nn.BatchNorm2d(channels*4)
        self.d21 = nn.Conv2d(channels*8, channels*4, kernel_size=3, padding=1)

        self.bn81 = nn.BatchNorm2d(channels*2)
        self.upconv3 = nn.ConvTranspose2d(channels*4, channels*2, kernel_size=2, stride=2)
        self.bn82 = nn.BatchNorm2d(channels*2)
        self.d31 = nn.Conv2d(channels*4, channels*2, kernel_size=3, padding=1)

        self.bn91 = nn.BatchNorm2d(channels)
        self.upconv4 = nn.ConvTranspose2d(channels*2, channels, kernel_size=2, stride=2)
        self.bn92 = nn.BatchNorm2d(channels)
        self.d41 = nn.Conv2d(channels*2, channels, kernel_size=3, padding=1)

        self.out = nn.Sequential(nn.Conv2d(channels, output_channels, kernel_size=1),
                                 nn.Sigmoid())

        # Initialize weights
        self.init_weights()

    def init_weights(self):
      for layer in self.modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.ConvTranspose2d):
          gain = nn.init.calculate_gain("leaky_relu", self.negative_slope)
          nn.init.xavier_uniform_(layer.weight, gain=gain)
          nn.init.zeros_(layer.bias)
        elif isinstance(layer, nn.BatchNorm2d):
          nn.init.constant_(layer.weight, 1)
          nn.init.constant_(layer.bias, 0)


    def forward(self, x):
        dim = x.shape[2]
        x = self.pad(x)

        #Encoder
        xe11 = F.leaky_relu(self.bn11(self.e11(x)), negative_slope=self.negative_slope)
        xp1 = self.pool1(xe11)

        xe21 = F.leaky_relu(self.bn21(self.e21(xp1)), negative_slope=self.negative_slope)
        xp2 = self.pool2(xe21)

        xe31 = F.leaky_relu(self.bn31(self.e31(xp2)), negative_slope=self.negative_slope)
        xp3 = self.pool3(xe31)

        xe41 = F.leaky_relu(self.bn41(self.e41(xp3)), negative_slope=self.negative_slope)
        xp4 = self.pool4(xe41)

        xe51 = F.leaky_relu(self.bn51(self.e51(xp4)), negative_slope=self.negative_slope)

        #Decoder
        xu1 = F.leaky_relu(self.bn61(self.upconv1(xe51)), negative_slope=self.negative_slope)
        xu11 = torch.cat([xu1, xe41], dim=1)
        xd11 = F.leaky_relu(self.bn62(self.d11(xu11)), negative_slope=self.negative_slope)

        xu2 = F.leaky_relu(self.bn71(self.upconv2(xd11)), negative_slope=self.negative_slope)
        xu22 = torch.cat([xu2, xe31], dim=1)
        xd21 = F.leaky_relu(self.bn72(self.d21(xu22)), negative_slope=self.negative_slope)

        xu3 = F.leaky_relu(self.bn81(self.upconv3(xd21)), negative_slope=self.negative_slope)
        xu33 = torch.cat([xu3, xe21], dim=1)
        xd31 = F.leaky_relu(self.bn81(self.d31(xu33)), negative_slope=self.negative_slope)

        xu4 = F.leaky_relu(self.bn91(self.upconv4(xd31)), negative_slope=self.negative_slope)
        xu44 = torch.cat([xu4, xe11], dim=1)
        xd41 = F.leaky_relu(self.bn92(self.d41(xu44)), negative_slope=self.negative_slope)

        out = self.out(xd41)

        out = out[:, :, :dim, :dim]

        return out

### TRAINING ###

def f1_score(inputs, targets, epsilon=1e-7, treshold = 0.5):
    # Ensure tensors have the same shape
    assert inputs.shape == targets.shape

    binary_input = (inputs >= treshold).float()

    # Calculate true positives, false positives, and false negatives
    #Black (1) is considered the positive
    true_positives = torch.sum(targets * binary_input)
    false_positives = torch.sum((1 - targets) * binary_input)
    false_negatives = torch.sum(targets * (1-binary_input))

    # Calculate precision and recall
    precision = true_positives / (true_positives + false_positives + epsilon)
    recall = true_positives / (true_positives + false_negatives + epsilon)

    # Calculate F1 score
    f1 = 2 * (precision * recall) / (precision + recall + epsilon)

    return f1

def adam_optimizer(model, lr, weight_decay = 0):
  return torch.optim.Adam(model.parameters(), lr=lr, weight_decay = weight_decay)

def onehot_to_image(array):
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

def show_matrices(inputs, observed, predicted, treshold=0.5, output_file = None):

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
    plt.savefig(output_file)
  else:
    plt.show()

  plt.close()


def show_loss(train_loss, valid_loss, time):
  plt.figure()
  plt.plot(time, train_loss, label = "Training")
  plt.plot(time, valid_loss, label = "Valiation")
  plt.title("Loss")
  plt.legend()
  plt.show()

def show_F1(train_F1, valid_F1, time):
  plt.figure()
  plt.plot(time, train_F1, label = "Training")
  plt.plot(time, valid_F1, label = "Validation")
  plt.title("F1 score")
  plt.legend()
  plt.show()


def fit_model(model, train_dataset, validation_dataset, loss_func = F.binary_cross_entropy, optimizer = adam_optimizer, lr=0.01, bs=1, epochs=5, plots = True):
  train_dl = DataLoader(train_dataset, batch_size=bs, shuffle=True)
  valid_dl = DataLoader(validation_dataset, batch_size=bs)

  opt = optimizer(model, lr)


  #Add stuff to track history
  train_loss_history = []
  train_F1_history = []
  valid_loss_history = []
  valid_F1_history = []
  plot_time = []

  t = 0 #To keep track of time

  device = 'cpu'
  if torch.cuda.is_available():
    device = 'cuda:0'

  model.to(device)


  #Train for a given number of epochs
  for epoch in range(epochs):
    t+=1
    running_loss = 0.0
    running_F1 = 0.0
    model.train()

    for input, output, label in train_dl:
      output = output.unsqueeze(1) #Since output is made from NxN matrix, we need to unsqueeze to get channel dimension

      #Forward pass
      predicted = model(input.to(device))

      #Compute loss
      loss = loss_func(predicted, output.to(device))
      running_loss += loss.item()
      running_F1 += f1_score(predicted, output.to(device)).item()

      # Backpropagation, optimization and zeroing the gradients
      loss.backward()
      opt.step()
      opt.zero_grad()

    #Validation loss (only after each epoch)
    valid_loss, valid_F1 = 0.0, 0.0
    model.eval()
    with torch.no_grad():
      for valid_input, valid_output, valid_label in valid_dl:
        valid_output =  valid_output.unsqueeze(1)
        predicted_valid = model(valid_input.to(device))

        valid_loss += loss_func(predicted_valid, valid_output.to(device)).item()
        valid_F1 += f1_score(predicted_valid, valid_output.to(device)).item()

    train_loss = running_loss/len(train_dl)
    train_F1 = running_F1/len(train_dl)
    valid_loss = valid_loss/len(valid_dl)
    valid_F1 = valid_F1/len(valid_dl)

    valid_loss_history.append(valid_loss)
    valid_F1_history.append(valid_F1)
    train_loss_history.append(train_loss)
    train_F1_history.append(train_F1)
    plot_time.append(t)

    show_matrices(input, output, predicted)

    print(f"Epoch [{epoch + 1}/{epochs}], Loss: {round(train_loss,4)}, Validation: {round(valid_loss, 4)}")

  #Loss plot
  if plots:
    show_loss(train_loss_history, valid_loss_history, plot_time)
    show_F1(train_F1_history, valid_F1_history, plot_time)

  return train_loss_history, train_F1_history, valid_loss_history, valid_F1_history, plot_time