import os, torch, math, pickle, sys
import numpy as np
import random as rd
from fnmatch import fnmatch
from collections import defaultdict, namedtuple

def read_ct(file: str) -> tuple:
    """
    Takes a .ct file and returns the sequence as a string and a list pairing state of each base (0 = unpaired)

    Parameters:
    - file (str): The path to the .ct file.

    Returns:
    - tuple: A tuple containing the sequence as a string and a list of integers representing the pairing state of each base.
    """
    sequence = ""
    pairs = []

    with open(file, 'r') as f:
        lines = [line.split() for line in f.readlines()]

    #Remove header - if any
    header_lines = 0
    for line in lines:
        if line[0] == '1':
                break
        else:
            header_lines += 1

    lines = lines[header_lines:]

    for line in lines:
        sequence += line[1].upper() #Make sure the sequence is in upper case
        pairs.append(int(line[4])-1) if line[4] != '0' else pairs.append(None)

    # Make sure the sequence is the same length as the pairs list
    assert len(sequence) == len(pairs)

    return sequence, pairs

def list_all_files(root: str, pattern = '*.ct') -> list:
    """
    List all files in a directory and its subdirectories that match a specific pattern.

    Parameters:
    - root (str): The path to the root directory.
    - pattern (str, optional): The pattern to match. Default is '*.ct'.

    Returns:
    - list: A list of file paths.
    """
    ct_files = []

    for path, subdirs, files  in os.walk(root):
        for file in files:
            if fnmatch(file, pattern):
                ct_files.append(os.path.join(path, file))
    return ct_files

def getLength(ct_file: str) -> int:
    '''
    Opens a ct_file and returns the length of the sequence as it appears in the first line.

    Parameters:
    - ct_file (str): The path to the .ct file.

    Returns:
    - int: The length of the sequence.
    '''
    with open(ct_file, "r") as f:
      for line in f:
        length = int(line.split()[0])
        break
    return length


def sequence_onehot(sequence: str) -> np.ndarray:
    """
    Convert a sequence to a onehot representation

    Parameters:
    - sequence (str): The sequence to convert.

    Returns:
    - np.array: A 3D numpy array with shape (4, len(sequence), 1).
    
    """
    seq_dict = {
        'A':np.array([1,0,0,0], dtype=np.float32),
        'U':np.array([0,1,0,0], dtype=np.float32),
        'C':np.array([0,0,1,0], dtype=np.float32),
        'G':np.array([0,0,0,1], dtype=np.float32),
        'N':np.array([0,0,0,0], dtype=np.float32),
        'M':np.array([1,0,1,0], dtype=np.float32),
        'Y':np.array([0,1,1,0], dtype=np.float32),
        'W':np.array([1,0,0,0], dtype=np.float32),
        'V':np.array([1,0,1,1], dtype=np.float32),
        'K':np.array([0,1,0,1], dtype=np.float32),
        'R':np.array([1,0,0,1], dtype=np.float32),
        'I':np.array([0,0,0,0], dtype=np.float32),
        'X':np.array([0,0,0,0], dtype=np.float32),
        'S':np.array([0,0,1,1], dtype=np.float32),
        'D':np.array([1,1,0,1], dtype=np.float32),
        'P':np.array([0,0,0,0], dtype=np.float32),
        'B':np.array([0,1,1,1], dtype=np.float32),
        'H':np.array([1,1,1,0], dtype=np.float32)}
    
    onehot = np.array([seq_dict[base] for base in sequence])
    return onehot

def input_representation(sequence: str) -> np.ndarray:
    """
    Returns a NxNx16 matrix with the onehot representation of the sequence and all possible base pairs

    Parameters:
    - sequence (str): The sequence to convert.

    Returns:
    - np.array: A 3D numpy array with shape (len(sequence), len(sequence), 16).
    """
    x = sequence_onehot(sequence)
    return np.kron(x, x).reshape((len(sequence), len(sequence), 16))

def Gaussian(x: int, t2 = 1) -> float: 
    """
    Calculate the Gaussian function for a given x and t2

    Parameters:
    - x (int): The x value.
    - t2 (float, optional): The t2 value. Default is 1.

    Returns:
    - float: The value of the Gaussian function.
    """
    return math.exp(-(x*x)/(2*t2))

def P(pair: str, x: float = 0.8):
    """
    Calculate the score for pairing of a given base pair

    Parameters:
    - pair (str): The base pair.
    - x (float, optional): The x value, which is assigned to GU and UG base pairs. Default is 0.8.

    Returns:
    - float: The score for base pairing.
    """ 
    pairs = {'AU': 2, 'UA':2,
             'GC': 3, 'CG': 3,
             'GU': x, 'UG': x}
    
    if pair in pairs: 
        return pairs[pair]
    else: 
        return 0


def calculate_W(sequence: str, i: int, j: int) -> float: 
    """
    Calculate the value for W[i, j] in the W matrix
    Based on method used in Ufold

    Parameters:
    - sequence (str): The sequence.
    - i (int): The i value.
    - j (int): The j value.

    Returns:
    - float: The value for W[i, j].
    """
    pairs = {'AU', 'UA', 'GU', 'UG', 'GC', 'CG'}
    
    ij =  0
    
    if abs(i-j) < 4: #Only calculate for bases that are at least 3 positions apart
        return ij
    
    for alpha in range(30): 
        if i - alpha >= 0 and j + alpha < len(sequence) and abs(i-j) >= 4: 
            score = P(sequence[i - alpha] + sequence[j + alpha])
            if score == 0: 
                break
            ij += Gaussian(alpha)*score
        else: 
            break

    if ij > 0: 
        for beta in range(1, 30): 
            if i + beta < len(sequence) and j - beta >= 0 and abs(i-j) >= 4: 
                score = P(sequence[i + beta] + sequence[j-beta]) 
                if score == 0: 
                    break
                ij += Gaussian(beta)*score
            else: 
                break
        
    return ij

def calculate_score_matrix(sequence: str) -> np.ndarray: 
    """
    Calculate the score matrix for a given sequence
    Is performed based on the method used in Ufold

    Parameters:
    - sequence (str): The sequence.

    Returns:
    - np.array: A 3D numpy array with shape (len(sequence), len(sequence), 1).
    """
    N = len(sequence)
    S = np.zeros((N, N), dtype=np.float32)

    for i, j in np.ndindex(N, N):
        S[i, j] = calculate_W(sequence, i, j) 
    
    return S.reshape(N, N, 1)

def make_matrix_from_sequence_16(sequence: str) -> torch.Tensor:
    """
    A sequence is converted to a matrix containing all the possible base pairs resulting in a 3D matrix with 16 layers
    Each pair in encoded as a onehot vector.

    Parameters:
    - sequence (str): The sequence to convert.

    Returns:
    - torch.Tensor: A 3D tensor with shape (16, len(sequence), len(sequence)).
    """
    return torch.from_numpy(input_representation(sequence)).permute(2, 0, 1)

def make_matrix_from_sequence_17(sequence: str) -> torch.Tensor:
    """
    A sequence is converted to a matrix containing all the possible base pairs and the score matrix resulting in a 3D matrix with 17 layers
    Each pair in encoded as a onehot vector.
    Unpaired are the bases on the diagonal, representing the unpaired/unfolded sequence

    Parameters:
    - sequence (str): The sequence to convert.

    Returns:
    - torch.Tensor: A 3D tensor with shape (17, len(sequence), len(sequence)).
    """
    return torch.from_numpy(np.concatenate((input_representation(sequence), calculate_score_matrix(sequence)), axis = -1)).permute(2, 0, 1)


def make_matrix_from_sequence_9(sequence: str) -> torch.Tensor:
    """
    A sequence is converted to a matrix containing all the possible base pairs and the score matrix resulting in a 3D matrix with 9 layers
    Each pair in encoded as a onehot vector.
    Unpaired are the bases on the diagonal, representing the unpaired/unfolded sequence

    Parameters:
    - sequence (str): The sequence to convert.

    Returns:
    - torch.Tensor: A 3D tensor with shape (9, len(sequence), len(sequence)).
    """

    return torch.cat((make_matrix_from_sequence_8(sequence), torch.from_numpy(calculate_score_matrix(sequence)).permute(2, 0, 1)), dim=0)



def make_matrix_from_sequence_8(sequence: str, device: str = 'cpu') -> torch.Tensor:
    """
    A sequence is converted to a matrix containing all the possible base pairs
    Each pair in encoded as a onehot vector.
    Unpaired are the bases on the diagonal, representing the unpaired/unfolded sequence

    Parameters:
    - sequence (str): The sequence to convert.

    Returns:
    - torch.Tensor: A 3D tensor with shape (8, len(sequence), len(sequence)).
    """
    coding = np.array([
        [1, 0, 0, 0, 0, 0, 0, 0],  # invalid pairing
        [0, 1, 0, 0, 0, 0, 0, 0],  # unpaired
        [0, 0, 1, 0, 0, 0, 0, 0],  # GC
        [0, 0, 0, 1, 0, 0, 0, 0],  # CG
        [0, 0, 0, 0, 1, 0, 0, 0],  # UG
        [0, 0, 0, 0, 0, 1, 0, 0],  # GU
        [0, 0, 0, 0, 0, 0, 1, 0],  # UA
        [0, 0, 0, 0, 0, 0, 0, 1],  # AU
    ], dtype=np.float32)

    basepairs = ["GC", "CG", "UG", "GU", "UA", "AU"]

    N = len(sequence)

    # Create an array filled with "invalid pairing" vectors
    matrix = np.tile(coding[0], (N, N, 1))

    # Update the diagonal with "unpaired" vectors
    matrix[np.arange(N), np.arange(N), :] = coding[1]
    
    # Update base pair positions directly
    for i, j in np.ndindex(N, N):
        pair = sequence[i] + sequence[j]
        if pair in basepairs and abs(i-j) >=4:
            matrix[i, j, :] = coding[basepairs.index(pair)+2]

    return torch.from_numpy(matrix.transpose(2, 0, 1))


def make_matrix_from_basepairs(pairs: list, unpaired: bool = True) -> torch.Tensor:
    """
    Takes a list of all which base each position in the sequence is paired with. If a base is unpaired pairs[i] = 0.
    From the list a 2D matrix is made, with each cell coresponding to a base pair encoded as 1 and unpaired bases encoded as 1 at the diagonal

    Parameters:
    - pairs (list): A list of integers representing the pairing state of each base.
    - unpaired (bool, optional): If True the unpaired bases are encoded as 1 at the diagonal. Default is True.

    Returns:
    - torch.Tensor: A 2D tensor with shape (len(pairs), len(pairs)).
    """

    N = len(pairs)
    matrix = torch.zeros((N,N), dtype=torch.float32)

    for i, j in enumerate(pairs):
        if isinstance(j, int):
            matrix[i, j] = 1
        elif unpaired:
            matrix[i, i] = 1

    return matrix

def make_pairs_from_list(pairs: list) -> list: 
    """
    Takes a list of the pairing state at each position in the sequence and converts it into a list with all base pairs in tuples

    Parameters:
    - pairs (list): A list of integers representing the pairing state of each base.

    Returns:
    - list: A list of tuples representing the base pairs.
    """
    pairs = [tuple([index, p]) for index, p in enumerate(pairs) if p and index < p]

    return pairs



RNA_data = namedtuple('RNA_data', 'input output length family name pairs') #Used for experiments
RNA = namedtuple('RNA', 'input output length family name sequence') #Used for complete data set

def update_progress_bar(current_index: int, total_indices: int) -> None:
    """
    Function to update a progress bar in the terminal

    Parameters:
    - current_index (int): The current index.
    - total_indices (int): The total number of indices.

    Returns:
    - None
    """
    progress = (current_index + 1) / total_indices
    bar_length = 50
    filled_length = int(bar_length * progress)
    bar = '=' * filled_length + '.' * (bar_length - filled_length)
    sys.stdout.write(f'\r[{bar}] {int(progress * 100)}%')
    sys.stdout.flush()

def file_length(file: str) -> int:
  """
  Returns the length of a sequence based on the value stored in the pickle file

  Parameters:
  - file (str): The path to the pickle file.

  Returns:
  - int: The length of the sequence.
  """
  return int(pickle.load(open(file, 'rb')).length)

def get_indices(ratios: tuple) -> tuple:
  """
  Returns indices for splitting 10 elements into three groups based on the ratios provided

  Parameters:
  - ratios (tuple): A tuple of three floats representing the ratios for the three groups.

  Returns:
  - tuple: A tuple of three lists of indices.
  """ 
  rd.seed(42)
  
  numbers = list(range(10))
  rd.shuffle(numbers)

  count1 = int(10 * ratios[0])
  count2 = int(10 * ratios[1])

  group1 = numbers[:count1]
  group2 = numbers[count1: count1+count2]
  group3 = numbers[count1+count2:]

  return group1, group2, group3

def split_data(file_list: list, train_ratio: float = 0.8, validation_ratio: float = 0.1, test_ratio: float = 0.1) -> tuple:
  """
  Split a list of files into three groups based on the provided ratios
  Returns three lists of files

  Parameters:
  - file_list (list): A list of file paths.
  - train_ratio (float, optional): The ratio for the training set. Default is 0.8.
  - validation_ratio (float, optional): The ratio for the validation set. Default is 0.1.
  - test_ratio (float, optional): The ratio for the test set. Default is 0.1.

  Returns:
  - tuple: A tuple of three lists of file paths in the order (train, valid, test).
  """
  #Test that the ratios sum to 1
  assert int((train_ratio + validation_ratio + test_ratio)) == 1
  
  #Get indices for splitting the data
  train_indices, valid_indices, test_indices = get_indices([train_ratio, validation_ratio, test_ratio])
  
  train, valid, test = [], [], []

  #Make a dictionary with the family as key and a list of files as value
  family_data = defaultdict(list)
  for file in file_list:
    family = pickle.load(open(file, 'rb')).family
    family_data[family].append((file))

  #Sort files within each family based on length
  for family, files in family_data.items():
    N = len(files)
    files.sort(key=file_length)

    #Used the optained indices to split batches of 10 files into train, validation and test
    for i in range(0, N, 10): 
      train_idx = [n+i for n in train_indices]
      valid_idx = [n+i for n in valid_indices]
      test_idx = [n+i for n in test_indices]

      train.extend([files[i] for i in train_idx if i < N])
      valid.extend(files[i] for i in valid_idx if i < N)
      test.extend(files[i] for i in test_idx if i < N)

  return train, valid, test

def make_family_map(file_list: list) -> dict: 
    """
    Create a dictionary mapping each family to a onehot vector

    Parameters:
    - file_list (list): A list of file paths.

    Returns:
    - dict: A dictionary with the family as key and a onehot vector as value.
    """
    families = []
    for index, file in enumerate(file_list): 
        update_progress_bar(index, len(file_list))
        families.append(pickle.load(open(file, 'rb')).family)
    
    families = set(families)
    
    family_map = {family: torch.from_numpy(np.eye(len(families))[i]) for i, family in enumerate(families)}
    
    return family_map
