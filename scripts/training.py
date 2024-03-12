import torch, os, pickle, logging
from collections import namedtuple

from torch.utils.data import DataLoader

def fit_model(): 
    return

if __name__ == "__main__":
    train = pickle.load(open('data/train.pkl', 'rb'))
    valid = pickle.load(open('data/val.pkl', 'rb'))
    
    pass