import pickle

from collections import namedtuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset

RNA = namedtuple('RNA', 'input output length family name sequence')

#### LOSS FUNCTIONS AND ERROR METRICS ####
def dice_loss(inputs, targets, smooth=1e-5):
  intersection = torch.sum(targets * inputs, dim=(1,2,3))
  sum_of_squares_pred = torch.sum(torch.square(inputs), dim=(1,2,3))
  sum_of_squares_true = torch.sum(torch.square(targets), dim=(1,2,3))
  dice = (2 * intersection + smooth) / (sum_of_squares_pred + sum_of_squares_true + smooth)
  return 1-dice

def f1_score(inputs, targets, epsilon=1e-7, treshold = 0.5):
    """
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
def adam_optimizer(model, lr, weight_decay = 0):
  return torch.optim.Adam(model.parameters(), lr=lr, weight_decay = weight_decay)

### OTHER FUNCTIONS ###
class ImageToImageDataset(Dataset):
    """

    """
    def __init__(self, file_list):
        self.file_list = file_list

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
      data = pickle.load(open(self.file_list[idx], 'rb'))
      
      input_image = data.input
      output_image = data.output

      return input_image, output_image
    

### MODEL ARCHITECTURES ###
class DynamicPadLayer(nn.Module):
  """
  """
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
    def __init__(self, channels=32, in_channels=8, output_channels=1, negative_slope = 0.01, pooling = MaxPooling):
        """
        args:
        num_channels: length of the one-hot encoding vector
        num_hidden_channels: number of channels in the hidden layers of both encoder and decoder
        """
        super(RNA_Unet, self).__init__()

        self.negative_slope = negative_slope

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
        dim = x.shape[2]
        x = self.pad(x)

        #Encoder
        xe1 = self.e1(x)
        xp1 = self.pool1(xe1)

        xe2 = self.e2(xp1)
        xp2 = self.pool2(xe2)

        xe3 = self.e3(xp2)
        xp3 = self.pool3(xe3)

        xe4 = self.e4(xp3)
        xp4 = self.pool4(xe4)

        xe5 = self.e5(xp4)

        #Decoder
        xu1 = self.upconv1(xe5)
        xu11 = torch.cat([xu1, xe4], dim=1)
        xd1 = self.d1(xu11)

        xu2 = self.upconv2(xd1)
        xu22 = torch.cat([xu2, xe3], dim=1)
        xd2 = self.d2(xu22)

        xu3 = self.upconv3(xd2)
        xu33 = torch.cat([xu3, xe2], dim=1)
        xd3 = self.d3(xu33)

        xu4 = self.upconv4(xd3)
        xu44 = torch.cat([xu4, xe1], dim=1)
        xd4 = self.d4(xu44)

        out = self.out(xd4)

        out = out[:, :, :dim, :dim]

        return out


### EVALUATION FUNCTIONS ###
def evaluate(y_pred, y_true, epsilon: float=1e-10): 
    """
    epsilon is a small number that is added to avoid potential division by 0 
    """
    
    TP = torch.sum(y_true * y_pred) #True positive
    FP = torch.sum((1 - y_true) * y_pred) #False postive
    FN = torch.sum(y_true * (1-y_pred)) #False negative

    precision = TP / (TP + FP + epsilon)
    recall = TP / (TP + FN + epsilon)

    F1 = 2 * (precision*recall)/(precision+recall+epsilon)
    
    
    return precision.item(), recall.item(), F1.item()
