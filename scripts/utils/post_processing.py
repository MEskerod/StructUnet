import numpy as np
import networkx as nx

def argmax_postprocessing(matrix: np.array): 
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

def blossum_postprocessing(matrix: np.array): 
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

def blossom_linear(matrix: np.array):
    return