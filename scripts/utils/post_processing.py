import torch, math

import torch.nn.functional as F

import numpy as np
import networkx as nx

from utils import blossom
from utils.Mfold1 import Mfold as Mfold_param
from utils.Mfold2 import Mfold as Mfold_constrain
from utils.hotknots import hotknots

def pairs(x: str, y:str) -> bool:
    if x == 'A' and y == 'U':
        return True
    if x == 'U' and y == 'A':
        return True
    if x == 'C' and y == 'G':
        return True
    if x == 'G' and y == 'C':
        return True
    if x == 'G' and y == 'U':
        return True
    if x == 'U' and y == 'G':
        return True
    if x == 'N' or y == 'N':
        return True
    return False

def prepare_input(matrix: torch.Tensor, sequence: str, device: str) -> torch.Tensor:
    """
    Takes a sequence and returns a mask tensor with 1s in the positions where a base can pair with another base and for unpaired bases.
    """
    
    N = len(sequence)
    
    m = torch.eye(N, device=device)

    #Make mask to ensure only allowed base pairs and that no sharp turns are present
    for i in range(N):
        for j in range(N):
            if abs(i-j) > 3:
                if pairs(sequence[i], sequence[j]):
                    m[i, j] = 1
    
    #Make symmetric and apply mask
    matrix = (matrix + matrix.T) / 2 * m
    
    return matrix

def argmax_postprocessing(matrix: torch.Tensor, sequence: str, device: str) -> torch.Tensor:
    """
    Postprocessing function that takes a matrix and returns a matrix with 1s in the position of the maximum value in each row.
    The functions has sequence as input, but does not use it. It is provided to make the function compatible with other postprocessing functions.

    Parameters:
    - matrix (torch.Tensor): The matrix to postprocess.
    - sequence (str): The sequence that the matrix was generated from.

    Returns:
    - torch.Tensor: The postprocessed matrix.
    """ 
    y_out = torch.zeros_like(matrix)
    indices = torch.argmax(matrix, dim=1)
    y_out.scatter_(1, indices.unsqueeze(1), 1)

    zero_rows = ~torch.any(y_out != 0, dim=1)

    y_out[zero_rows, zero_rows] = 1

    return y_out


def nx_blossum_postprocessing(matrix: torch.Tensor, sequence: str, device: str) -> torch.Tensor: 
    """
    Postprocessing function that takes a matrix and returns a matrix.
    The function uses NetworkX to find the maximum weight matching in the graph representation of the matrix, with copying of the matrix to allow for self-pairing.
    The functions has sequence as input, but does not use it. It is provided to make the function compatible with other postprocessing functions.

    Parameters:
    - matrix (torch.Tensor): The matrix to postprocess.
    - sequence (str): The sequence that the matrix was generated from.

    Returns:
    - torch.Tensor: The postprocessed matrix.
    """    
    n = matrix.shape[0]

    mask = torch.eye(n, device=device)*2

    A = torch.zeros((2*n, 2*n), device=device)
    A[:n, :n] = matrix
    A[n:, n:] = matrix
    A[:n, n:] = matrix*mask
    A[n:, :n] = matrix*mask

    G = nx.convert_matrix.from_numpy_array(A.numpy())
    pairing = nx.max_weight_matching(G)

    y_out = torch.zeros_like(matrix, device=device)

    for (i, j) in pairing:
        if i>n and j>n:
            continue
        y_out[i%n, j%n] = 1
        y_out[j%n, i%n] = 1
    
    return y_out

def blossom_postprocessing(matrix: torch.Tensor, sequence: str, device: str) -> torch.Tensor: 
    """
    Postprocessing function that takes a matrix and returns a matrix.
    The function uses the blossom algorithm to find the maximum weight matching in the graph representation of the matrix, with copying of the matrix to allow for self-pairing.
    The functions used are modified version of NetworkX functions, and are implemented in the blossom.py file.
    The functions has sequence as input, but does not use it. It is provided to make the function compatible with other postprocessing functions.

    Parameters:
    - matrix (torch.Tensor): The matrix to postprocess.
    - sequence (str): The sequence that the matrix was generated from.

    Returns:
    - torch.Tensor: The postprocessed matrix.
    """
    n = matrix.shape[0]

    mask = torch.eye(n, device=device)*2

    A = torch.zeros((2*n, 2*n), device=device)
    A[:n, :n] = matrix
    A[n:, n:] = matrix
    A[:n, n:] = matrix*mask
    A[n:, :n] = matrix*mask

    pairing = blossom.max_weight_matching_matrix(A)

    y_out = torch.zeros_like(matrix, device=device)

    for (i, j) in pairing:
        if i>n and j>n:
            continue
        y_out[i%n, j%n] = 1
        y_out[j%n, i%n] = 1
    
    return y_out

def blossom_weak(matrix: torch.Tensor, sequence: str, device: str, treshold: float = 0.75) -> torch.Tensor: 
    """
    Postprocessing function that takes a matrix and returns a matrix.
    Uses the blossom algorithm to find the maximum weight matching in the graph representation of the matrix.
    Since the blossom algorithm does not allow for self-pairing (i.e. pairing a base with itself), the matrix is first thresholded to remove weak pairings.
    This allows for bases with no strong pairings to be left out of the pairing.
    The function has sequence as input, but does not use it. It is provided to make the function compatible with other postprocessing functions.

    Parameters:
    - matrix (torch.Tensor): The matrix to postprocess.
    - sequence (str): The sequence that the matrix was generated from.
    - treshold (float): The treshold to use for the matrix.

    Returns:
    - torch.Tensor: The postprocessed matrix.
    """
    matrix = matrix.clone()
    matrix[matrix < treshold] = 0

    pairs = blossom.max_weight_matching_matrix(matrix)

    y_out = torch.zeros_like(matrix, device=device)

    for (i, j) in pairs:
        y_out[i, j] = y_out[j, i] = 1

    # Find rows where all elements are zero
    zero_rows = ~torch.any(y_out != 0, dim=1)

    # Set diagonal elements to 1 in rows where all other elements are zero
    y_out[zero_rows, torch.arange(y_out.shape[0], device=device)[zero_rows]] = 1

    return y_out

def Mfold_param_postprocessing(matrix: torch.Tensor, sequence: str, device: str) -> torch.Tensor:
    """
    Postprocessing function that takes a matrix and returns a matrix.
    Uses the Mfold algorithm to find the maximum weight matching in the graph representation of the matrix.
    The Mfold alorthm uses the matrix from the network as parameters for base pairing.

    Parameters:
    - matrix (torch.Tensor): The matrix to postprocess.
    - sequence (str): The sequence that the matrix was generated from.

    Returns:
    - torch.Tensor: The postprocessed matrix.
    """
    M = -matrix.clone()
    M[M == 0] = torch.inf

    pairs = Mfold_param(sequence, M, device)

    y_out = torch.zeros_like(matrix, device=device)

    for i, j in enumerate(pairs):
        y_out[i, j] = 1
    
    return y_out


#TODO - Remember to ad application of mask to the evaluation script (and timing of everything)(and GPU usage)
#TODO - Change evaluate to use tensors (and check what scripts uses evaluate and change them to tensors if possible)


def Mfold_constrain_postprocessing(matrix: torch.Tensor, sequence: str, device: str, treshold: float = 0.5) -> torch.Tensor: 
    """
    Postprocessing function that takes a matrix and returns a matrix.
    Uses the Mfold algorithm to find the maximum weight matching in the graph representation of the matrix.
    The Mfold alorthm uses the matrix from the network as constraints for which bases can pair.

    Parameters:
    - matrix (torch.Tensor): The matrix to postprocess.
    - sequence (str): The sequence that the matrix was generated from.
    - device (str): The device to use for the matrix.
    - treshold (float): The treshold to use for the matrix. Bases with a value below the treshold are not allowed to pair.

    Returns:
    - torch.Tensor: The postprocessed matrix.
    """
    matrix = matrix.clone()
    matrix[matrix < treshold] = 0
    
    pairs = Mfold_constrain(sequence, matrix, device)
    
    y_out = torch.zeros_like(matrix, device=device)

    for i, j in enumerate(pairs):
        y_out[i, j]  = 1
    
    return y_out

def hotknots_postprocessing(matrix: torch.Tensor, sequence: str, device: str, k=3, gap_penalty = 0.5, treshold_prop = 0.8) -> torch.Tensor:  #TODO - Change "fixed" parameters
    """
    Postprocessing function that takes a matrix and returns a matrix.
    Uses the HotKnots algorithm, which is a heuristic able to find structures with pseudoknot.
    The HotKnots algorithm uses the matrix from the network as parameters for base pairing.

    Parameters:
    - matrix (torch.Tensor): The matrix to postprocess.
    - sequence (str): The sequence that the matrix was generated from.
    - device (str): The device to use for the matrix.
    - k (int): The maximum number of children for each node.
    - gap_penalty (float): The penalty for gaps in the structure.
    - treshold_prop (float): The porportion of the naive structure to use as treshold for adding new hotspots.

    Returns:
    - torh.Tensor: The postprocessed matrix.
    """
    pairs = hotknots(matrix, sequence, k=k, gap_penalty=gap_penalty, treshold_prop=treshold_prop)

    y_out = torch.zeros_like(matrix, device=device)
    
    for (i, j) in pairs:
        y_out[i, j] = y_out[j, i] = 1
    
    for i in range(matrix.shape[0]): 
        if not torch.any(y_out[i, :], device=device):    
            y_out[i, i] = 1
    
    return y_out