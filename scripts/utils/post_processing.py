import numpy as np
import networkx as nx

from utils import blossom
from utils.Mfold1 import Mfold as Mfold_param
from utils.Mfold2 import Mfold as Mfold_constrain
from utils.hotknots import hotknots

def argmax_postprocessing(matrix: np.ndarray, sequence: str) -> np.ndarray:
    """
    Postprocessing function that takes a matrix and returns a matrix with 1s in the position of the maximum value in each row.
    The functions has sequence as input, but does not use it. It is provided to make the function compatible with other postprocessing functions.

    Parameters:
    - matrix (np.ndarray): The matrix to postprocess.
    - sequence (str): The sequence that the matrix was generated from.

    Returns:
    - np.ndarray: The postprocessed matrix.
    """ 
    N = matrix.shape[0]
    
    #Make symmetric
    matrix = (matrix + matrix.T) / 2

    y_out = np.zeros_like(matrix)
    
    indices =  np.argmax(matrix, axis=1)
    y_out[np.arange(N), indices] = 1
    
    for i in range(N): 
        if not np.any(y_out[i, :]) and not np.any(y_out[:, i]): 
            y_out[i, i] = 1
    
    return y_out

def nx_blossum_postprocessing(matrix: np.ndarray, sequence: str) -> np.ndarray: 
    """
    Postprocessing function that takes a matrix and returns a matrix.
    The function uses NetworkX to find the maximum weight matching in the graph representation of the matrix, with copying of the matrix to allow for self-pairing.
    The functions has sequence as input, but does not use it. It is provided to make the function compatible with other postprocessing functions.

    Parameters:
    - matrix (np.ndarray): The matrix to postprocess.
    - sequence (str): The sequence that the matrix was generated from.

    Returns:
    - np.ndarray: The postprocessed matrix.
    """
    n = matrix.shape[0]

    mask = np.eye(n)*2

    A = np.zeros((2*n, 2*n))
    A[:n, :n] = matrix
    A[n:, n:] = matrix
    A[:n, n:] = matrix*mask
    A[n:, :n] = matrix*mask

    G = nx.convert_matrix.from_numpy_array(A)
    pairing = nx.max_weight_matching(G)

    y_out = np.zeros_like(matrix)

    for (i, j) in pairing:
        if i>n and j>n:
            continue
        y_out[i%n, j%n] = 1
        y_out[j%n, i%n] = 1
    
    return y_out

def blossom_postprocessing(matrix: np.ndarray, sequence: str) -> np.ndarray: 
    """
    Postprocessing function that takes a matrix and returns a matrix.
    The function uses the blossom algorithm to find the maximum weight matching in the graph representation of the matrix, with copying of the matrix to allow for self-pairing.
    The functions used are modified version of NetworkX functions, and are implemented in the blossom.py file.
    The functions has sequence as input, but does not use it. It is provided to make the function compatible with other postprocessing functions.

    Parameters:
    - matrix (np.ndarray): The matrix to postprocess.
    - sequence (str): The sequence that the matrix was generated from.

    Returns:
    - np.ndarray: The postprocessed matrix.
    """
    n = matrix.shape[0]

    mask = np.eye(n)*2

    A = np.zeros((2*n, 2*n))
    A[:n, :n] = matrix
    A[n:, n:] = matrix
    A[:n, n:] = matrix*mask
    A[n:, :n] = matrix*mask

    pairing = blossom.max_weight_matching_matrix(A)

    y_out = np.zeros_like(matrix)

    for (i, j) in pairing:
        if i>n and j>n:
            continue
        y_out[i%n, j%n] = 1
        y_out[j%n, i%n] = 1
    
    return y_out

def blossom_weak(matrix: np.ndarray, sequence: str, treshold: float = 0.5) -> np.ndarray: 
    """
    Postprocessing function that takes a matrix and returns a matrix.
    Uses the blossom algorithm to find the maximum weight matching in the graph representation of the matrix.
    Since the blossom algorithm does not allow for self-pairing (i.e. pairing a base with itself), the matrix is first thresholded to remove weak pairings.
    This allows for bases with no strong pairings to be left out of the pairing.
    The function has sequence as input, but does not use it. It is provided to make the function compatible with other postprocessing functions.

    Parameters:
    - matrix (np.ndarray): The matrix to postprocess.
    - sequence (str): The sequence that the matrix was generated from.
    - treshold (float): The treshold to use for the matrix.

    Returns:
    - np.ndarray: The postprocessed matrix.
    """
    matrix[matrix < treshold] = 0

    pairs = blossom.max_weight_matching_matrix(matrix)

    y_out = np.zeros_like(matrix)

    for (i, j) in pairs:
        y_out[i, j] = y_out[j, i] = 1
    
    for i in range(matrix.shape[0]): 
        if not np.any(y_out[i, :]): 
            y_out[i, i] = 1
    
    return y_out

def Mfold_param_postprocessing(matrix: np.ndarray, sequence: str) -> np.ndarray:
    """
    Postprocessing function that takes a matrix and returns a matrix.
    Uses the Mfold algorithm to find the maximum weight matching in the graph representation of the matrix.
    The Mfold alorthm uses the matrix from the network as parameters for base pairing.

    Parameters:
    - matrix (np.ndarray): The matrix to postprocess.
    - sequence (str): The sequence that the matrix was generated from.

    Returns:
    - np.ndarray: The postprocessed matrix.
    """
    M = np.copy(-matrix)
    M[M == 0] = np.inf

    pairs = Mfold_param(sequence, M)

    y_out = np.zeros_like(matrix)

    for (i, j) in pairs:
        y_out[i, j] = y_out[j, i] = 1
    
    for i in range(matrix.shape[0]): 
        if not np.any(y_out[i, :]): 
            y_out[i, i] = 1
    
    return y_out

def Mfold_constrain_postprocessing(matrix: np.ndarray, sequence: str, treshold: float = 0.01) -> np.ndarray: 
    """
    Postprocessing function that takes a matrix and returns a matrix.
    Uses the Mfold algorithm to find the maximum weight matching in the graph representation of the matrix.
    The Mfold alorthm uses the matrix from the network as constraints for which bases can pair.

    Parameters:
    - matrix (np.ndarray): The matrix to postprocess.
    - sequence (str): The sequence that the matrix was generated from.
    - treshold (float): The treshold to use for the matrix. Bases with a value below the treshold are not allowed to pair.

    Returns:
    - np.ndarray: The postprocessed matrix.
    """
    matrix[matrix < treshold] = 0
    
    pairs = Mfold_constrain(sequence, matrix)
    
    y_out = np.zeros_like(matrix)

    for (i, j) in pairs:
        y_out[i, j] = y_out[j, i] = 1
    
    for i in range(matrix.shape[0]): 
        if not np.any(y_out[i, :]): 
            y_out[i, i] = 1
    
    return y_out

def hotknots_postprocessing(matrix: np.ndarray, sequence: str, k=15, gap_penalty = 0.5, treshold_prop = 1) -> np.ndarray:  #TODO - Change "fixed" parameters
    """
    Postprocessing function that takes a matrix and returns a matrix.
    Uses the HotKnots algorithm, which is a heuristic able to find structures with pseudoknot.
    The HotKnots algorithm uses the matrix from the network as parameters for base pairing.

    Parameters:
    - matrix (np.ndarray): The matrix to postprocess.
    - sequence (str): The sequence that the matrix was generated from.

    Returns:
    - np.ndarray: The postprocessed matrix.
    """
    pairs = hotknots(matrix, sequence, k=k, gap_penalty=gap_penalty, treshold_prop=treshold_prop)

    y_out = np.zeros_like(matrix)
    
    for (i, j) in pairs:
        y_out[i, j] = y_out[j, i] = 1
    
    for i in range(matrix.shape[0]): 
        if not np.any(y_out[i, :]): 
            y_out[i, i] = 1
    
    return y_out