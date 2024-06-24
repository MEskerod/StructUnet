import torch, pickle

import pandas as pd

from collections import namedtuple

import torch.nn as nn
import torch.nn.functional as F


from tqdm import tqdm
from utils.model_and_training import RNA_Unet

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
  
class RNA_Unet_multi_large(nn.Module):
    def __init__(self, channels=32, in_channels=8, output_channels=1, classes = 8, negative_slope = 0.01, pooling = MaxPooling):
        """
        args:
        num_channels: length of the one-hot encoding vector
        num_hidden_channels: number of channels in the hidden layers of both encoder and decoder
        """
        super(RNA_Unet_multi_large, self).__init__()

        self.negative_slope = negative_slope

        self.pad = DynamicPadLayer(2**4)

        # Encoder
        self.bn11 = nn.BatchNorm2d(channels)
        self.e11 = nn.Conv2d(in_channels, channels, kernel_size = 3, padding = 1)
        self.bn12 = nn.BatchNorm2d(channels)
        self.e12 = nn.Conv2d(channels, channels, kernel_size = 3, padding = 1)
        self.pool1 = pooling(channels, channels, kernel_size=2, stride=2)

        self.bn21 = nn.BatchNorm2d(channels * 2)
        self.e21 = nn.Conv2d(channels, channels*2, kernel_size=3, padding=1)
        self.bn22 = nn.BatchNorm2d(channels * 2)
        self.e22 = nn.Conv2d(channels*2, channels*2, kernel_size=3, padding=1)
        self.pool2 = pooling(channels*2, channels*2, kernel_size=2, stride=2)

        self.bn31 = nn.BatchNorm2d(channels*4)
        self.e31 = nn.Conv2d(channels*2, channels*4, kernel_size=3, padding=1)
        self.bn32 = nn.BatchNorm2d(channels*4)
        self.e32 = nn.Conv2d(channels*4, channels*4, kernel_size=3, padding=1)
        self.pool3 = pooling(channels*4, channels*4, kernel_size=2, stride=2)

        self.bn41 = nn.BatchNorm2d(channels*8)
        self.e41 = nn.Conv2d(channels*4, channels*8, kernel_size=3, padding=1)
        self.bn42 = nn.BatchNorm2d(channels*8)
        self.e42 = nn.Conv2d(channels*8, channels*8, kernel_size=3, padding=1)
        self.pool4 = pooling(channels*8, channels*8, kernel_size=2, stride=2)

        self.bn51 = nn.BatchNorm2d(channels*16)
        self.e51 = nn.Conv2d(channels*8, channels*16, kernel_size=3, padding=1)
        self.bn52 = nn.BatchNorm2d(channels*16)
        self.e52 = nn.Conv2d(channels*16, channels*16, kernel_size=3, padding=1)

        #Decoder
        self.bn61 = nn.BatchNorm2d(channels*8)
        self.upconv1 = nn.ConvTranspose2d(channels*16, channels*8, kernel_size=2, stride=2)
        self.bn62 = nn.BatchNorm2d(channels*8)
        self.d11 = nn.Conv2d(channels*16, channels*8, kernel_size=3, padding=1)
        self.bn63 = nn.BatchNorm2d(channels*8)
        self.d12 = nn.Conv2d(channels*8, channels*8, kernel_size=3, padding=1)

        self.bn71 = nn.BatchNorm2d(channels*4)
        self.upconv2 = nn.ConvTranspose2d(channels*8, channels*4, kernel_size=2, stride=2)
        self.bn72 = nn.BatchNorm2d(channels*4)
        self.d21 = nn.Conv2d(channels*8, channels*4, kernel_size=3, padding=1)
        self.bn73 = nn.BatchNorm2d(channels*4)
        self.d22 = nn.Conv2d(channels*4, channels*4, kernel_size=3, padding=1)

        self.bn81 = nn.BatchNorm2d(channels*2)
        self.upconv3 = nn.ConvTranspose2d(channels*4, channels*2, kernel_size=2, stride=2)
        self.bn82 = nn.BatchNorm2d(channels*2)
        self.d31 = nn.Conv2d(channels*4, channels*2, kernel_size=3, padding=1)
        self.bn83 = nn.BatchNorm2d(channels*2)
        self.d32 = nn.Conv2d(channels*2, channels*2, kernel_size=3, padding=1)

        self.bn91 = nn.BatchNorm2d(channels)
        self.upconv4 = nn.ConvTranspose2d(channels*2, channels, kernel_size=2, stride=2)
        self.bn92 = nn.BatchNorm2d(channels)
        self.d41 = nn.Conv2d(channels*2, channels, kernel_size=3, padding=1)
        self.bn93 = nn.BatchNorm2d(channels)
        self.d42 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

        self.out = nn.Sequential(nn.Conv2d(channels, output_channels, kernel_size=1),
                                 nn.Sigmoid())

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels*16, classes),
            nn.Softmax(dim=1))

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
        xe12 = F.leaky_relu(self.bn12(self.e12(xe11)), negative_slope=self.negative_slope)
        xp1 = self.pool1(xe12)

        xe21 = F.leaky_relu(self.bn21(self.e21(xp1)), negative_slope=self.negative_slope)
        xe22 = F.leaky_relu(self.bn22(self.e22(xe21)), negative_slope=self.negative_slope)
        xp2 = self.pool2(xe22)

        xe31 = F.leaky_relu(self.bn31(self.e31(xp2)), negative_slope=self.negative_slope)
        xe32 = F.leaky_relu(self.bn32(self.e32(xe31)), negative_slope=self.negative_slope)
        xp3 = self.pool3(xe32)

        xe41 = F.leaky_relu(self.bn41(self.e41(xp3)), negative_slope=self.negative_slope)
        xe42 = F.leaky_relu(self.bn42(self.e42(xe41)), negative_slope=self.negative_slope)
        xp4 = self.pool4(xe42)

        xe51 = F.leaky_relu(self.bn51(self.e51(xp4)), negative_slope=self.negative_slope)
        xe52 = F.leaky_relu(self.bn52(self.e52(xe51)), negative_slope=self.negative_slope)

        #Decoder
        xu1 = F.leaky_relu(self.bn61(self.upconv1(xe52)), negative_slope=self.negative_slope)
        xu11 = torch.cat([xu1, xe42], dim=1)
        xd11 = F.leaky_relu(self.bn62(self.d11(xu11)), negative_slope=self.negative_slope)
        xd12 = F.leaky_relu(self.bn63(self.d12(xd11)), negative_slope=self.negative_slope)

        xu2 = F.leaky_relu(self.bn71(self.upconv2(xd12)), negative_slope=self.negative_slope)
        xu22 = torch.cat([xu2, xe32], dim=1)
        xd21 = F.leaky_relu(self.bn72(self.d21(xu22)), negative_slope=self.negative_slope)
        xd22 = F.leaky_relu(self.bn73(self.d22(xd21)), negative_slope=self.negative_slope)

        xu3 = F.leaky_relu(self.bn81(self.upconv3(xd22)), negative_slope=self.negative_slope)
        xu33 = torch.cat([xu3, xe22], dim=1)
        xd31 = F.leaky_relu(self.bn81(self.d31(xu33)), negative_slope=self.negative_slope)
        xd32 = F.leaky_relu(self.bn83(self.d32(xd31)), negative_slope=self.negative_slope)

        xu4 = F.leaky_relu(self.bn91(self.upconv4(xd32)), negative_slope=self.negative_slope)
        xu44 = torch.cat([xu4, xe12], dim=1)
        xd41 = F.leaky_relu(self.bn92(self.d41(xu44)), negative_slope=self.negative_slope)
        xd42 = F.leaky_relu(self.bn93(self.d42(xd41)), negative_slope=self.negative_slope)

        out = self.out(xd42)

        out = out[:, :, :dim, :dim]

        classification_output = self.classifier(xe52)

        return out, classification_output

def find_max_size(files): 
    max_length = 0
    max_file = ''

    progress = tqdm(total=len(files), desc='Finding max size of test set', unit='files')
    for file in files:
       length = pickle.load(open(file, 'rb')).length

       if length > max_length:
          max_length = length
          max_file = file
       progress.update()
    progress.close()
    
    tensor = pickle.load(open(max_file, 'rb')).input

    return tensor.numel() * tensor.element_size() / 1024 / 1024

def count_parameters(model: torch.nn.Module) -> int:
    """
    Counts the number of trainable parameters in a model.

    Parameters:
    - model (torch.nn.Module): The model to count the parameters of.

    Returns:
    - int: The number of trainable parameters in the model.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def model_memory_usage(model):
    total_memory = 0

    # Iterate over all parameters and buffers
    for param in model.parameters():
        total_memory += param.numel() * param.element_size()
        
    for buffer in model.buffers():
        total_memory += buffer.numel() * buffer.element_size()

    return total_memory / 1024 / 1024

if __name__ == "__main__":

    RNA = namedtuple('RNA', 'input output length family name sequence')

    print("Counting parameters... ")

    baseline_param = count_parameters(RNA_Unet(in_channels=8, channels=32))

    models = ['baseline', 'channels64', 'input16', 'input17', 'multitask']
    parameters = [count_parameters(RNA_Unet(in_channels=8, channels=32)),
                  count_parameters(RNA_Unet(in_channels=8, channels=64)),
                  count_parameters(RNA_Unet(in_channels=16, channels=32)),
                  count_parameters(RNA_Unet(in_channels=17, channels=32)),
                  count_parameters(RNA_Unet_multi_large(in_channels=8, channels=32))]
    increase = [params-baseline_param for params in parameters]
    percentage = [params/baseline_param*100 for params in parameters]
    size = [model_memory_usage(RNA_Unet(in_channels=8, channels=32)),
            model_memory_usage(RNA_Unet(in_channels=8, channels=64)),
            model_memory_usage(RNA_Unet(in_channels=16, channels=32)),
            model_memory_usage(RNA_Unet(in_channels=17, channels=32)),
            model_memory_usage(RNA_Unet_multi_large(in_channels=8, channels=32))]
    
    df = pd.DataFrame({'model': models, 'parameters': parameters, 'parameter increase':increase, 'percentage of baseline': percentage, 'size (MB)': size})
    df.to_csv('results/parameters.csv')

    print(df, '\n')
    
    files = pickle.load(open('data/test.pkl', 'rb'))
    print('Max size of test set: ', find_max_size(files), 'MB\n')
    
