import torch, pickle, os

import networkx as nx
import numpy as np

from torch.utils.data import Dataset, DataLoader

from collections import namedtuple

def keep_topk(mat, k=3):
    v, idx = torch.topk(mat, k, dim=-1)
    n = idx.size(-2)
    new_mat = torch.zeros_like(mat)
    for i in range(idx.size(0)):
        for j in range(k):
            new_mat[i, 0, range(n), idx[i, 0, :, j]] = mat[i, 0, range(n), idx[i, 0, :, j]]
    return new_mat.float()


def blossom(mat):
    A = mat.clone()
    n = A.size(-1)
    mask = torch.eye(n) * 2
    big_a = torch.zeros((2*n, 2*n))
    big_a[:n, :n] = A
    big_a[n:, n:] = A
    big_a[:n, n:] = A*mask
    big_a[n:, :n] = A*mask
    G = nx.convert_matrix.from_numpy_array(np.array(big_a.cpu().data))
    pairings = nx.matching.max_weight_matching(G)
    y_out = torch.zeros_like(A)
    for (i, j) in pairings:
        if i>n and j>n:
            continue
        y_out[i%n, j%n] = 1
        y_out[j%n, i%n] = 1
    return y_out

def binarize(y_hat, threshold=0, use_blossom=False):
    """
    First term is necessary to ignore paddings for example
    y_hat in shape [B, 1, N, N]
    """
    
    if not use_blossom:
        return (y_hat>threshold).float()*(y_hat == y_hat.max(dim=-1, keepdim=True)[0]).float()
    else:
        y_hat = keep_topk(y_hat, 3)
        new_y_hat = torch.zeros_like(y_hat[:, 0])
        for i, sample in enumerate(y_hat[:, 0]):
            out = blossom(sample.squeeze())
            new_y_hat[i] = out
        return new_y_hat.unsqueeze(1) # [B, 1, N, N]

def create_matrix(sequence, onehot=True, min_dist=3):
    """
    Create input matrix which is 16xNxN or 1xNxN according to the onehot value
    At the moment works faster than the previous one (matrix multiplication vs normal loop)
    
    We have 16 different pairing types w.r.t [A, U, G, C]
        0, 5, 10, 15 are self_loops (unpair) --> 1
        1, 4, 6, 9, 11, 14 are pairings --> 6
        others are invalid --> 1
        = 8 modes (channels)
    """
    n = len(sequence)
    invalid = []
    seq = []
    for i, s in enumerate(sequence):
        if s not in bases:
            invalid.append(i)
            seq.append(0)
        else:
            seq.append(bases.index(s))

    seq = torch.tensor(seq)
    if onehot:
        mat = torch.zeros((17, n, n))
    else:
        mat = torch.zeros((1, n, n))


    q2 = seq.repeat(n, 1)
    q1 = q2.transpose(1, 0)    
    t = torch.stack(((torch.abs(q1-q2)==1).long(), torch.eye(n).long()))
    mask = torch.max(t, 0)[0]
    flat_mat = ((q1*4+q2+1) * mask)
    
    for i in range(1, min_dist+1):
        flat_mat[range(n-i), range(i, n)] = 0
        flat_mat[range(i, n), range(n-i)] = 0
    
#     flat_mat[invalid] = 0
#     flat_mat[:, invalid] = 0
    flat_mat = flat_mat.unsqueeze(0)
    
    if onehot:
        idx2 = torch.arange(n).repeat(n, 1)
        idx1 = idx2.transpose(1, 0).reshape(-1)
        idx2 = idx2.reshape(-1)
        mat[flat_mat.reshape(-1), idx1, idx2] = 1
#         mat[q2.reshape(-1), idx1, idx2] = 1
#         mat[4+q1.reshape(-1), idx1, idx2] = 1
        mat = mat[1:]
        mat8 = mat[[1, 4, 6, 9, 11, 14]]
        # mat8 = mat[[1, 6, 11]]
        # mat8 = mat8 + mat8.transpose(-1, -2)

        mat8 = torch.cat((mat8, torch.sum(mat[[0, 5, 10, 15]], 0).unsqueeze(0)), 0)
        mat8 = torch.cat((mat8, 1-torch.sum(mat8, 0).unsqueeze(0)), 0)
        return mat8
    return flat_mat

class ImageToImageDataset(Dataset):
    """
    Dataset class for image to image translation
    For each sample, the dataset returns a tuple of input and output images
    Is initialized with a list of file paths to pickle files containing the data
    """
    def __init__(self, file_list_path: list) -> None:
        self.file_list = pickle.load(open(file_list_path, 'rb'))

    def __len__(self) -> int:
        return len(self.file_list)

    def __getitem__(self, idx: int) -> tuple:
      data = pickle.load(open(self.file_list[idx], 'rb'))

      input = create_matrix(data.sequence, onehot=True)
      length = len(data.sequence)
      
      return input, os.path.basename(self.file_list[idx])




def main(dataset):
    """
    """

    os.makedirs('results_CNNfold', exist_ok=True)

    absolute_path = os.path.abspath(os.path.dirname(__file__))

    print('==========Start Loading Pretrained Model==========')
    model_address = os.path.join(absolute_path, 'models/cnnfold600.mdl') # CNNFold-600 (for shorters)
    model_address2 = os.path.join(absolute_path, 'models/cnnfold.mdl') # CNNFold 
    cnn1 = torch.load(model_address, map_location='cpu')
    cnn1.train()
    cnn2 = torch.load(model_address2, map_location='cpu')
    cnn2.train()
    print('==========Finish Loading Pretrained Model==========')

    test_set = DataLoader(dataset, batch_size=1)
    print("TOTAL NUMBER OF SEQUENCES: ", len(test_set))

    print('==========Start Predicting==========')
    
    for i, (input, name) in enumerate(test_set):
        if i % 100 == 0:
            print('Predicting sequence number: ', i)

        #Predict
        with torch.no_grad():
            if input.size(-1) <= 600:
                predicted = cnn1(input, test = True)
            else: 
                predicted = cnn2(input, test = True)

        #Post-process
        predicted = binarize(predicted, threshold, use_blossom=True)
        predicted = predicted.squeeze().cpu().numpy()

        #Save
        pickle.dump(predicted, open('results_CNNfold/' + name[0], 'wb'))

    print('==========Finish Predicting==========')



if __name__ == '__main__':
    bases = ['A', 'U', 'G', 'C']
    topk_k = 5
    threshold = .0
    RNA = namedtuple('RNA', 'input output length family name sequence')
    dataset = ImageToImageDataset('data/test.pkl')
    main(dataset)
