import pickle, time, os, torch, datetime

import torch
import torch.optim as optim
from torch.utils import data

import numpy as np

from itertools import product

from Network import U_Net as FCNNet

from ufold.utils import *
from ufold.config import process_config
import time
from ufold.data_generator import RNASSDataGenerator_input
from ufold.data_generator import Dataset_Cut_concat_new as Dataset_FCN
from ufold.postprocess import postprocess_new as postprocess
import collections
from collections import defaultdict

import warnings
warnings.filterwarnings("ignore", message="FutureWarning: elementwise comparison failed")


absolute_path = os.path.dirname(os.path.realpath(__file__))
args = get_args(absolute_path=absolute_path)
perm = list(product(np.arange(4), np.arange(4)))

def format_time(seconds):
    """
    Format a time duration in seconds to hh:mm:ss format.
    
    Parameters:
    seconds: Time duration in seconds.
    
    Returns:
    Formatted time string in hh:mm:ss format.
    """
    time_delta = datetime.timedelta(seconds=seconds)
    hours, remainder = divmod(time_delta.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return '{:02d}:{:02d}:{:02d}'.format(hours, minutes, seconds)


def model_eval_all_test(contact_net,test_generator):
    device = torch.device("cpu")
    contact_net.train()

    batch_n = 0

    for seq_embeddings, seq_lens, seq_ori, seq_name in test_generator:
        if batch_n%100==0:
            print('Prediction number: ', batch_n)

        batch_n += 1

        #Prepare for prediction
        seq_embedding_batch = torch.Tensor(seq_embeddings.float()).to(device)
        seq_ori = torch.Tensor(seq_ori.float()).to(device)
        
        #Predict
        with torch.no_grad():
            pred_contacts = contact_net(seq_embedding_batch)

        # only post-processing without learning
        u_no_train = postprocess(pred_contacts,
            seq_ori, 0.01, 0.1, 100, 1.6, True,1.5)
        
        #Make output binary

        map_no_train = (u_no_train > 0.5).float()[:,:seq_lens[0],:seq_lens[0]].squeeze(0).cpu().numpy().astype("float32")
        for i in range(map_no_train.shape[0]):
            #Check if row is all zeros
            if np.all(map_no_train[i] == 0):
                map_no_train[i,i] = 1


        pickle.dump(map_no_train, open('results_Ufold/' + seq_name[0], 'wb'))
    
def get_cut_len(data_len,set_len):
    l = data_len
    if l <= set_len:
        l = set_len
    else:
        l = (((l - 1) // 16) + 1) * 16
    return l

def creatmat(data, device=None):
    if device==None:
        device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    with torch.no_grad():
        data = ''.join(['AUCG'[list(d).index(1)] for d in data])
        paired = defaultdict(int, {'AU':2, 'UA':2, 'GC':3, 'CG':3, 'UG':0.8, 'GU':0.8})

        mat = torch.tensor([[paired[x+y] for y in data] for x in data]).to(device)
        n = len(data)

        i, j = torch.meshgrid(torch.arange(n).to(device), torch.arange(n).to(device), indexing='ij')
        t = torch.arange(30).to(device)
        m1 = torch.where((i[:, :, None] - t >= 0) & (j[:, :, None] + t < n), mat[torch.clamp(i[:,:,None]-t, 0, n-1), torch.clamp(j[:,:,None]+t, 0, n-1)], 0)
        m1 *= torch.exp(-0.5*t*t)

        m1_0pad = torch.nn.functional.pad(m1, (0, 1))
        first0 = torch.argmax((m1_0pad==0).to(int), dim=2)
        to0indices = t[None,None,:]>first0[:,:,None]
        m1[to0indices] = 0
        m1 = m1.sum(dim=2)

        t = torch.arange(1, 30).to(device)
        m2 = torch.where((i[:, :, None] + t < n) & (j[:, :, None] - t >= 0), mat[torch.clamp(i[:,:,None]+t, 0, n-1), torch.clamp(j[:,:,None]-t, 0, n-1)], 0)
        m2 *= torch.exp(-0.5*t*t)

        m2_0pad = torch.nn.functional.pad(m2, (0, 1))
        first0 = torch.argmax((m2_0pad==0).to(int), dim=2)
        to0indices = torch.arange(29).to(device)[None,None,:]>first0[:,:,None]
        m2[to0indices] = 0
        m2 = m2.sum(dim=2)
        m2[m1==0] = 0

        return (m1+m2).to(torch.device('cpu'))

class Dataset_FCN(data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, file):
        'Initialization'
        
        with open(file, 'r') as f:
            data = [file.strip() for file in f.readlines()]
        self.files = data

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.files)
  
  def one_hot_600(self,seq_item):
        RNN_seq = seq_item
        BASES = 'AUCG'
        bases = np.array([base for base in BASES])
        feat = np.concatenate(
                [[(bases == base.upper()).astype(int)] if str(base).upper() in BASES else np.array([[-1] * len(BASES)]) for base
                in RNN_seq])
        one_hot_matrix_600 = np.zeros((600,4))
        one_hot_matrix_600[:len(seq_item),] = feat
        return one_hot_matrix_600

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        RNA = collections.namedtuple('RNA', 'input output length family name sequence')
        data = pickle.load(open(self.files[index], 'rb'))

        sequence = data.sequence.replace('N', 'A')

        data_len = len(sequence)
        data_seq = self.one_hot_600(sequence)

        l = get_cut_len(data_len,80)
        data_fcn = np.zeros((16, l, l))
        if l >= 500:
            seq_adj = np.zeros((l, 4))
            seq_adj[:data_len] = data_seq[:data_len]
            data_seq = seq_adj
        for n, cord in enumerate(perm):
            i, j = cord
            data_fcn[n, :data_len, :data_len] = np.matmul(data_seq[:data_len, i].reshape(-1, 1), data_seq[:data_len, j].reshape(1, -1))
        data_fcn_1 = np.zeros((1,l,l))
        data_fcn_1[0,:data_len,:data_len] = creatmat(data_seq[:data_len,])
        data_fcn_2 = np.concatenate((data_fcn,data_fcn_1),axis=0)
        
        return data_fcn_2, data_len, data_seq[:l], os.path.basename(self.files[index])

def main():
    torch.multiprocessing.set_sharing_strategy('file_system')

    print('Welcome using UFold prediction tool!!!')

    os.makedirs('results_Ufold', exist_ok=True)

    config_file = args.config

    config = process_config(config_file)
    
    d = config.u_net_d
    BATCH_SIZE = config.batch_size_stage_1
    OUT_STEP = config.OUT_STEP
    LOAD_MODEL = config.LOAD_MODEL
    data_type = config.data_type
    model_type = config.model_type
    epoches_first = config.epoches_first

    script_dir = os.path.dirname(os.path.realpath(__file__))
    MODEL_SAVED = os.path.join(script_dir, 'models/ufold_train.pt')

    device = "cpu"

    seed_torch()
    
    params = {'batch_size': BATCH_SIZE,
              'shuffle': True,
              'num_workers': 1,
              'drop_last': True}

    test_set = Dataset_FCN('input.txt')
    #test_set = Dataset_FCN(test_data)s
    test_generator = data.DataLoader(test_set, **params)
    contact_net = FCNNet(img_ch=17)

    print('==========Start Loading Pretrained Model==========')
    contact_net.load_state_dict(torch.load(MODEL_SAVED,map_location='cpu'))
    print('==========Finish Loading Pretrained Model==========')
    print("TOTAL NUMBER OF SEQUENCES: ", len(test_set))
    contact_net.to(device)
    time_start = time.time()
    model_eval_all_test(contact_net,test_generator)
    total_time = time.time() - time_start
    print('==========Done!!! Please check results folder for the predictions!==========')
    print(f'Total time: {format_time(total_time)}. Average time per sequence: {total_time/len(test_set):.5f} seconds.')

    
if __name__ == '__main__':
    """
    See module-level docstring for a description of the script.
    """
    RNA_SS_data = collections.namedtuple('RNA_SS_data','seq ss_label length name pairs')
    RNA = collections.namedtuple('RNA', 'input output length family name sequence')
    main()





