import os, torch, math, pickle, sys
import numpy as np
import random as rd
from fnmatch import fnmatch
from collections import defaultdict, namedtuple

from torch.utils.data import Dataset

def read_ct(file: str) -> tuple():
    """
    Takes a .ct file and returns the sequence as a string and a list pairing state of each base (0 = unpaired)
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
        sequence += line[1].upper()
        pairs.append(int(line[4])-1) if line[4] != '0' else pairs.append(0)

    assert len(sequence) == len(pairs)

    return sequence, pairs

def list_all_files(root, pattern = '*.ct'):
    ct_files = []

    for path, subdirs, files  in os.walk(root):
        for file in files:
            if fnmatch(file, pattern):
                ct_files.append(os.path.join(path, file))
    return ct_files

def getLength(ct_file: str):
    '''
    Opens a ct_file and returns the length of the sequence as it appears in the first line.
    '''
    with open(ct_file, "r") as f:
      for line in f:
        length = int(line.split()[0])
        break
    return length

def getFamily(file_name):
  '''
  '''
  return ''.join(file_name.split(os.sep)[4].split('_')[:-1])


def sequence_onehot(sequence):
    """
    
    """
    seq_dict = {
        'A':np.array([1,0,0,0]),
        'U':np.array([0,1,0,0]),
        'C':np.array([0,0,1,0]),
        'G':np.array([0,0,0,1]),
        'N':np.array([0,0,0,0]),
        'M':np.array([1,0,1,0]),
        'Y':np.array([0,1,1,0]),
        'W':np.array([1,0,0,0]),
        'V':np.array([1,0,1,1]),
        'K':np.array([0,1,0,1]),
        'R':np.array([1,0,0,1]),
        'I':np.array([0,0,0,0]),
        'X':np.array([0,0,0,0]),
        'S':np.array([0,0,1,1]),
        'D':np.array([1,1,0,1]),
        'P':np.array([0,0,0,0]),
        'B':np.array([0,1,1,1]),
        'H':np.array([1,1,1,0])}
    
    onehot = np.array([seq_dict[base] for base in sequence])
    return onehot

def input_representation(sequence):
    """
    
    """
    x = sequence_onehot(sequence)
    return np.kron(x, x).reshape((len(sequence), len(sequence), 16))

def Gaussian(x, t2 = 1): 
    """
    
    """
    return math.exp(-(x*x)/(2*t2))

def P(pair, x = 0.8): 
    pairs = {'AU': 2, 'UA':2,
             'GC': 3, 'CG': 3,
             'GU': x, 'UG': x}
    
    if pair in pairs: 
        return pairs[pair]
    else: 
        return 0


def calculate_W(sequence, i, j): 
    """
    
    """
    pairs = {'AU', 'UA', 'GU', 'UG', 'GC', 'CG'}
    
    ij =  0
    
    if abs(i-j) < 4:
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

def calculate_score_matrix(sequence): 
    """
    
    """
    N = len(sequence)
    S = np.zeros((N, N))

    for i, j in np.ndindex(N, N):
        S[i, j] = calculate_W(sequence, i, j) 
    
    return S.reshape(N, N, 1)

def make_matrix_from_sequence_17(sequence: str) -> np.array:
    """

    """
    return torch.from_numpy(np.concatenate((input_representation(sequence), calculate_score_matrix(sequence)), axis = -1)).permute(2, 0, 1)




def make_matrix_from_sequence_8(sequence: str) -> np.array:
    """
    A sequence is converted to a matrix containing all the possible base pairs

    Each pair in encoded as a onehot vector.

    Unpaired are the bases on the diagonal, representing the unpaired/unfolded sequence
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

    return torch.from_numpy(matrix.transpose((2, 0, 1)))


def make_matrix_from_basepairs(pairs: list) -> np.array:
    """
    Takes a list of all the base pairs.
    From the list a 2D matrix is made, with each cell coresponding to a base pair encoded as 1 and unpaired bases encoded as 1 at the diagonal
    """

    N = len(pairs)
    matrix = np.full((N,N), 0, dtype="float32")

    pairs = [p if p else index for index, p in enumerate(pairs)]

    for i in range(N):
        matrix[i, pairs[i]] = 1 

    return torch.from_numpy(matrix)

def make_pairs_from_list(pairs): 
    """
    Takes a list of the pairing state at each position in the sequence and converts it into a list with all base pairs in tuples
    """
    pairs = [tuple([index, p]) for index, p in enumerate(pairs) if p and index < p]

    return pairs


RNA_data = namedtuple('RNA_data', 'input output length family name pairs')

def update_progress_bar(current_index, total_indices):
    progress = (current_index + 1) / total_indices
    bar_length = 50
    filled_length = int(bar_length * progress)
    bar = '=' * filled_length + '.' * (bar_length - filled_length)
    sys.stdout.write(f'\r[{bar}] {int(progress * 100)}%')
    sys.stdout.flush()

def process_and_save(file_list: list, output_folder: str, matrix_type: str = '8'):
  """
  Takes a list of ct files and converts them into a namedtuple type element containing: 
  - input
  - output
  - sequence length 
  - RNAfamily 
  - file name 
  - pairs in structure
  All elements are saved into output_folder as pickle files. 
  """
  converted = 0

  os.makedirs(output_folder, exist_ok=True)
  
  for i, file in enumerate(file_list):

    length = getLength(file)
    family = getFamily(file)

    sequence, pairs = read_ct(file)

    try:
        if (i + 1) % 100 == 0:
            update_progress_bar(i, len(file_list))

        if matrix_type == '8':
            input_matrix = make_matrix_from_sequence_8(sequence)
        elif matrix_type == '17':  
            input_matrix = make_matrix_from_sequence_17(sequence)
        else: 
            raise ValueError("Wrong matrix type")
        
        output_matrix = make_matrix_from_basepairs(pairs)

        sample = RNA_data(input = input_matrix,
                                     output = output_matrix,
                                     length = length,
                                     family = family,
                                     name = file, 
                                     pairs = make_pairs_from_list(pairs))
        
        pickle.dump(sample, open(os.path.join(output_folder, os.path.splitext(os.path.basename(file))[0] + '.pkl'), 'wb'))
        converted += 1

    except Exception as e:
        # Skip this file if an unexpected error occurs during processing
        print(f"Skipping {file} due to unexpected error: {e}", file=sys.stderr)
  
  print(f"\n\n{converted} files converted", file=sys.stdout)

def file_length(file):
  return int(pickle.load(open(file, 'rb')).length)

def get_indices(ratios): 
  rd.seed(42)
  
  numbers = list(range(10))
  rd.shuffle(numbers)

  count1 = int(10 * ratios[0])
  count2 = int(10 * ratios[1])

  group1 = numbers[:count1]
  group2 = numbers[count1: count1+count2]
  group3 = numbers[count1+count2:]

  return group1, group2, group3

def split_data(file_list, train_ratio = 0.8, validation_ratio = 0.1, test_ratio = 0.1, input_path = "input", output_path = "output"):
  train_indices, valid_indices, test_indices = get_indices([train_ratio, validation_ratio, test_ratio])
  
  train, valid, test = [], [], []

  family_data = defaultdict(list)
  for file in file_list:
    family = pickle.load(open(file, 'rb')).family
    family_data[family].append((file))

  for family, files in family_data.items():
    N = len(files)
    files.sort(key=file_length)

    for i in range(0, N, 10): 
      train_idx = [n+i for n in train_indices]
      valid_idx = [n+i for n in valid_indices]
      test_idx = [n+i for n in test_indices]

      train.extend([files[i] for i in train_idx if i < N])
      valid.extend(files[i] for i in valid_idx if i < N)
      test.extend(files[i] for i in test_idx if i < N)

  return train, valid, test

def make_family_map(file_list): 
    """
    """
    families = []
    for index, file in enumerate(file_list): 
        update_progress_bar(index, len(file_list))
        families.append(pickle.load(open(file, 'rb')).family)
    
    families = set(families)
    
    family_map = {family: torch.from_numpy(np.eye(len(families))[i]) for i, family in enumerate(families)}
    
    return family_map

class ImageToImageDataset(Dataset):
    """

    """
    def __init__(self, file_list, family_map):
        self.file_list = family_map

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
      data = pickle.load(open(self.file_list[idx], 'rb'))
      
      input_image = data.input
      output_image = data.output

      family = data.family
      label = self.family_map[family]

      return input_image, output_image, label