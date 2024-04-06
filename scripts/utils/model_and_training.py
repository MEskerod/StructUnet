import pickle

from collections import namedtuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset

RNA = namedtuple('RNA', 'input output length family name sequence')

#### LOSS FUNCTIONS AND ERROR METRICS ####
def dice_loss(inputs: torch.Tensor, targets: torch.Tensor, smooth: float=1e-7) -> torch.Tensor:
  """
  Calculate the dice loss for a batch of inputs and targets

  Parameters:
  - inputs (torch.Tensor): The input tensor.
  - targets (torch.Tensor): The target tensor.
  - smooth (float): A small number to avoid division by zero. Default is 1e-5.

  Returns:
  torch.Tensor: The dice loss.
  """
  intersection = torch.sum(targets * inputs, dim=(1,2,3))
  sum_of_squares_pred = torch.sum(torch.square(inputs), dim=(1,2,3))
  sum_of_squares_true = torch.sum(torch.square(targets), dim=(1,2,3))
  dice = (2 * intersection + smooth) / (sum_of_squares_pred + sum_of_squares_true + smooth)
  return 1-dice

def f1_score(inputs: torch.Tensor, targets: torch.Tensor, epsilon: float=1e-7, treshold:float = 0.5) -> torch.Tensor:
    """
    Calculate the F1 score for a batch of inputs and targets

    Parameters:
    - inputs (torch.Tensor): The input tensor.
    - targets (torch.Tensor): The target tensor.
    - epsilon (float): A small number to avoid division by zero. Default is 1e-7.
    """
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

#### OPTIMIZER ####
def adam_optimizer(model: nn.Module, lr: float, weight_decay: float = 0.0) -> torch.optim.Optimizer:
  """
  Wrapper for the Adam optimizer

  Parameters:
  - model (nn.Module): The model to optimize.
  - lr (float): The learning rate.
  - weight_decay (float): The weight decay. Default is 0.0.

  Returns:
  torch.optim.Optimizer: The Adam optimizer.
  """
  return torch.optim.Adam(model.parameters(), lr=lr, weight_decay = weight_decay)

### OTHER FUNCTIONS ###
class ImageToImageDataset(Dataset):
    """
    Dataset class for image to image translation
    For each sample, the dataset returns a tuple of input and output images
    Is initialized with a list of file paths to pickle files containing the data
    """
    def __init__(self, file_list: list) -> None:
        self.file_list = file_list

    def __len__(self) -> int:
        return len(self.file_list)

    def __getitem__(self, idx: int) -> tuple:
      data = pickle.load(open(self.file_list[idx], 'rb'))
      
      input_image = data.input
      output_image = data.output 

      return input_image, output_image
    

### MODEL ARCHITECTURES ###
class DynamicPadLayer(nn.Module):
  """
  Layer for dynamic padding
  For the RNA Unet, the input size must be divisible by a number of times equal to the stride product (stride*stride*....*stride = stride_product)
  Adds zero padding at bottom and right of the input tensor to make the input a compatible size
  """
  def __init__(self, stride_product):
    super(DynamicPadLayer, self).__init__()
    self.stride_product = stride_product

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    input_size = x.shape[2]
    padding = self.calculate_padding(input_size, self.stride_product)
    return nn.functional.pad(x, padding)

  def calculate_padding(self, input_size: int, stride_product: int) -> tuple:
    p = stride_product - input_size % stride_product
    return (0, p, 0, p)

class MaxPooling(nn.Module):
  """
  Wrapper for the max pooling layer
  The wrapper is needed to make the pooling layer have the same inputs as convolutional layers
  """
  def __init__(self, in_channels, out_channels, kernel_size=2, stride=2, padding=0):
    super(MaxPooling, self).__init__()
    self.max_pool = nn.MaxPool2d(kernel_size = kernel_size, stride = stride)

  def forward(self, x):
    return self.max_pool(x)
  

class RNA_Unet(nn.Module):
    def __init__(self, channels=64, in_channels=8, output_channels=1, negative_slope = 0.01, pooling = MaxPooling):
        """
        Pytorch implementation of a Unet for RNA secondary structure prediction

        Parameters:
        - channels (int): number of channels in the first hidden layer.
        - in_channels (int): number of channels in the input layer
        - output_channels (int): number of channels in the output layer
        - negative_slope (float): negative slope for the LeakyReLU activation function
        - pooling (nn.Module): the pooling layer to use
        """
        super(RNA_Unet, self).__init__()

        self.negative_slope = negative_slope

        #Add padding layer to make input size compatible with the Unet
        self.pad = DynamicPadLayer(2**4)

        # Encoder
        self.e1 = nn.Sequential(
           nn.Conv2d(in_channels, channels, kernel_size=3, padding=1),
           nn.BatchNorm2d(channels),
           nn.LeakyReLU(negative_slope=negative_slope),
           nn.Conv2d(channels, channels, kernel_size=3, padding=1),
           nn.BatchNorm2d(channels),
           nn.LeakyReLU(negative_slope=negative_slope),
        )
        self.pool1 = pooling(channels, channels, kernel_size=2, stride=2)

        self.e2 = nn.Sequential(
            nn.Conv2d(channels, channels*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels*2),
            nn.LeakyReLU(negative_slope=negative_slope),
            nn.Conv2d(channels*2, channels*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels*2),
            nn.LeakyReLU(negative_slope=negative_slope),
        )
        self.pool2 = pooling(channels*2, channels*2, kernel_size=2, stride=2)

        self.e3 = nn.Sequential(
            nn.Conv2d(channels*2, channels*4, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels*4),
            nn.LeakyReLU(negative_slope=negative_slope),
            nn.Conv2d(channels*4, channels*4, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels*4),
            nn.LeakyReLU(negative_slope=negative_slope),
        )
        self.pool3 = pooling(channels*4, channels*4, kernel_size=2, stride=2)

        self.e4 = nn.Sequential(
            nn.Conv2d(channels*4, channels*8, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels*8),
            nn.LeakyReLU(negative_slope=negative_slope),
            nn.Conv2d(channels*8, channels*8, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels*8),
            nn.LeakyReLU(negative_slope=negative_slope),
        )
        self.pool4 = pooling(channels*8, channels*8, kernel_size=2, stride=2)

        self.e5 = nn.Sequential(
            nn.Conv2d(channels*8, channels*16, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels*16),
            nn.LeakyReLU(negative_slope=negative_slope),
            nn.Conv2d(channels*16, channels*16, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels*16),
            nn.LeakyReLU(negative_slope=negative_slope),
        )

        #Decoder
        self.upconv1 = nn.Sequential(
            nn.ConvTranspose2d(channels*16, channels*8, kernel_size=2, stride=2),
            nn.BatchNorm2d(channels*8),
            nn.LeakyReLU(negative_slope=negative_slope),
        )
        self.d1 = nn.Sequential(
           nn.Conv2d(channels*16, channels*8, kernel_size=3, padding=1),
           nn.BatchNorm2d(channels*8),
           nn.LeakyReLU(negative_slope=negative_slope),
           nn.Conv2d(channels*8, channels*8, kernel_size=3, padding=1),
           nn.BatchNorm2d(channels*8),
           nn.LeakyReLU(negative_slope=negative_slope),
        )
        
        self.upconv2 = nn.Sequential(
            nn.ConvTranspose2d(channels*8, channels*4, kernel_size=2, stride=2),
            nn.BatchNorm2d(channels*4),
            nn.LeakyReLU(negative_slope=negative_slope),
        )
        self.d2 = nn.Sequential(
           nn.Conv2d(channels*8, channels*4, kernel_size=3, padding=1),
           nn.BatchNorm2d(channels*4),
           nn.LeakyReLU(negative_slope=negative_slope),
           nn.Conv2d(channels*4, channels*4, kernel_size=3, padding=1),
           nn.BatchNorm2d(channels*4),
           nn.LeakyReLU(negative_slope=negative_slope),
        )

        self.upconv3 = nn.Sequential(
            nn.ConvTranspose2d(channels*4, channels*2, kernel_size=2, stride=2),
            nn.BatchNorm2d(channels*2),
            nn.LeakyReLU(negative_slope=negative_slope),
        )
        self.d3 = nn.Sequential(
           nn.Conv2d(channels*4, channels*2, kernel_size=3, padding=1),
           nn.BatchNorm2d(channels*2),
           nn.LeakyReLU(negative_slope=negative_slope),
           nn.Conv2d(channels*2, channels*2, kernel_size=3, padding=1),
           nn.BatchNorm2d(channels*2),
           nn.LeakyReLU(negative_slope=negative_slope),
        )

        self.upconv4 = nn.Sequential(
            nn.ConvTranspose2d(channels*2, channels, kernel_size=2, stride=2),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(negative_slope=negative_slope),
        )
        self.d4 = nn.Sequential(
           nn.Conv2d(channels*2, channels, kernel_size=3, padding=1),
           nn.BatchNorm2d(channels),
           nn.LeakyReLU(negative_slope=negative_slope),
           nn.Conv2d(channels, channels, kernel_size=3, padding=1),
           nn.BatchNorm2d(channels),
           nn.LeakyReLU(negative_slope=negative_slope),
        )

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
        dim = x.shape[2] #Keep track of the original dimension of the input to remove padding at the end
        x = self.pad(x)

        #Encoder
        xe1 = self.e1(x)
        x = self.pool1(xe1)

        xe2 = self.e2(x)
        x = self.pool2(xe2)

        xe3 = self.e3(x)
        x = self.pool3(xe3)

        xe4 = self.e4(x)
        x = self.pool4(xe4)

        x = self.e5(x)

        #Decoder
        x = self.upconv1(x)
        x = torch.cat([x, xe4], dim=1)
        x = self.d1(x)

        x = self.upconv2(x)
        x = torch.cat([x, xe3], dim=1)
        x = self.d2(x)

        x = self.upconv3(x)
        x = torch.cat([x, xe2], dim=1)
        x = self.d3(x)

        x = self.upconv4(x)
        x = torch.cat([x, xe1], dim=1)
        x = self.d4(x)

        x = self.out(x)

        x = x[:, :, :dim, :dim]

        return x


### EVALUATION FUNCTIONS ###
def evaluate(y_pred: torch.Tensor, y_true: torch.Tensor, epsilon: float=1e-10) -> tuple: 
    """
    Function to evaluate the performance of a model based on precision, recall and F1 score

    Parameters:
    - y_pred (torch.Tensor): The predicted output matrix
    - y_true (torch.Tensor): The true output matrix
    - epsilon (float): A small number to avoid division by zero. Default is 1e-10.

    Returns:
    tuple: A tuple containing the precision, recall and F1 score
    """
    
    TP = torch.sum(y_true * y_pred) #True positive
    FP = torch.sum((1 - y_true) * y_pred) #False postive
    FN = torch.sum(y_true * (1-y_pred)) #False negative

    precision = TP / (TP + FP + epsilon)
    recall = TP / (TP + FN + epsilon)

    F1 = 2 * (precision*recall)/(precision+recall+epsilon)
    
    
    return precision.item(), recall.item(), F1.item()
