import sys, argparse, os, time

from tqdm import tqdm

from Bio import SeqIO

import torch
import torch.nn as nn

import blossom


#### MODEL ####
class DynamicPadLayer(nn.Module):
  """
  Layer for dynamic padding
  For the RNA Unet, the input size must be divisible by a number of times equal to the stride product (stride*stride*....*stride = stride_product)
  Adds zero padding at bottom and right of the input tensor to make the input a compatible size
  """
  def __init__(self, stride_product):
    super(DynamicPadLayer, self).__init__()
    self.stride_product = stride_product

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    input_size = x.shape[2]
    padding = self.calculate_padding(input_size, self.stride_product)
    return nn.functional.pad(x, padding)

  def calculate_padding(self, input_size: int, stride_product: int) -> tuple:
    p = stride_product - input_size % stride_product
    return (0, p, 0, p)

class MaxPooling(nn.Module):
  """
  Wrapper for the max pooling layer
  The wrapper is needed to make the pooling layer have the same inputs as convolutional layers
  """
  def __init__(self, in_channels, out_channels, kernel_size=2, stride=2, padding=0):
    super(MaxPooling, self).__init__()
    self.max_pool = nn.MaxPool2d(kernel_size = kernel_size, stride = stride)

  def forward(self, x):
    return self.max_pool(x)
  

class RNA_Unet(nn.Module):
    def __init__(self, channels=32, in_channels=8, output_channels=1, negative_slope = 0.01, pooling = MaxPooling):
        """
        Pytorch implementation of a Unet for RNA secondary structure prediction

        Parameters:
        - channels (int): number of channels in the first hidden layer.
        - in_channels (int): number of channels in the input layer
        - output_channels (int): number of channels in the output layer
        - negative_slope (float): negative slope for the LeakyReLU activation function
        - pooling (nn.Module): the pooling layer to use
        """
        super(RNA_Unet, self).__init__()

        self.negative_slope = negative_slope

        #Add padding layer to make input size compatible with the Unet
        self.pad = DynamicPadLayer(2**4)

        # Encoder
        self.e1 = nn.Sequential(
           nn.Conv2d(in_channels, channels, kernel_size=3, padding=1),
           nn.BatchNorm2d(channels, affine=False),
           nn.LeakyReLU(negative_slope=negative_slope),
           nn.Conv2d(channels, channels, kernel_size=3, padding=1),
           nn.BatchNorm2d(channels, affine=False),
           nn.LeakyReLU(negative_slope=negative_slope),
        )
        self.pool1 = pooling(channels, channels, kernel_size=2, stride=2)

        self.e2 = nn.Sequential(
            nn.Conv2d(channels, channels*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels*2, affine=False),
            nn.LeakyReLU(negative_slope=negative_slope),
            nn.Conv2d(channels*2, channels*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels*2, affine=False),
            nn.LeakyReLU(negative_slope=negative_slope),
        )
        self.pool2 = pooling(channels*2, channels*2, kernel_size=2, stride=2)

        self.e3 = nn.Sequential(
            nn.Conv2d(channels*2, channels*4, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels*4, affine=False),
            nn.LeakyReLU(negative_slope=negative_slope),
            nn.Conv2d(channels*4, channels*4, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels*4, affine=False),
            nn.LeakyReLU(negative_slope=negative_slope),
        )
        self.pool3 = pooling(channels*4, channels*4, kernel_size=2, stride=2)

        self.e4 = nn.Sequential(
            nn.Conv2d(channels*4, channels*8, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels*8, affine=False),
            nn.LeakyReLU(negative_slope=negative_slope),
            nn.Conv2d(channels*8, channels*8, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels*8, affine=False),
            nn.LeakyReLU(negative_slope=negative_slope),
        )
        self.pool4 = pooling(channels*8, channels*8, kernel_size=2, stride=2)

        self.e5 = nn.Sequential(
            nn.Conv2d(channels*8, channels*16, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels*16, affine=False),
            nn.LeakyReLU(negative_slope=negative_slope),
            nn.Conv2d(channels*16, channels*16, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels*16, affine=False),
            nn.LeakyReLU(negative_slope=negative_slope),
        )

        #Decoder
        self.upconv1 = nn.Sequential(
            nn.ConvTranspose2d(channels*16, channels*8, kernel_size=2, stride=2),
            nn.BatchNorm2d(channels*8, affine=False),
            nn.LeakyReLU(negative_slope=negative_slope),
        )
        self.d1 = nn.Sequential(
           nn.Conv2d(channels*16, channels*8, kernel_size=3, padding=1),
           nn.BatchNorm2d(channels*8, affine=False),
           nn.LeakyReLU(negative_slope=negative_slope),
           nn.Conv2d(channels*8, channels*8, kernel_size=3, padding=1),
           nn.BatchNorm2d(channels*8, affine=False),
           nn.LeakyReLU(negative_slope=negative_slope),
        )
        
        self.upconv2 = nn.Sequential(
            nn.ConvTranspose2d(channels*8, channels*4, kernel_size=2, stride=2),
            nn.BatchNorm2d(channels*4, affine=False),
            nn.LeakyReLU(negative_slope=negative_slope),
        )
        self.d2 = nn.Sequential(
           nn.Conv2d(channels*8, channels*4, kernel_size=3, padding=1),
           nn.BatchNorm2d(channels*4, affine=False),
           nn.LeakyReLU(negative_slope=negative_slope),
           nn.Conv2d(channels*4, channels*4, kernel_size=3, padding=1),
           nn.BatchNorm2d(channels*4, affine=False),
           nn.LeakyReLU(negative_slope=negative_slope),
        )

        self.upconv3 = nn.Sequential(
            nn.ConvTranspose2d(channels*4, channels*2, kernel_size=2, stride=2),
            nn.BatchNorm2d(channels*2, affine=False),
            nn.LeakyReLU(negative_slope=negative_slope),
        )
        self.d3 = nn.Sequential(
           nn.Conv2d(channels*4, channels*2, kernel_size=3, padding=1),
           nn.BatchNorm2d(channels*2, affine=False),
           nn.LeakyReLU(negative_slope=negative_slope),
           nn.Conv2d(channels*2, channels*2, kernel_size=3, padding=1),
           nn.BatchNorm2d(channels*2, affine=False),
           nn.LeakyReLU(negative_slope=negative_slope),
        )

        self.upconv4 = nn.Sequential(
            nn.ConvTranspose2d(channels*2, channels, kernel_size=2, stride=2),
            nn.BatchNorm2d(channels, affine=False),
            nn.LeakyReLU(negative_slope=negative_slope),
        )
        self.d4 = nn.Sequential(
           nn.Conv2d(channels*2, channels, kernel_size=3, padding=1),
           nn.BatchNorm2d(channels, affine=False),
           nn.LeakyReLU(negative_slope=negative_slope),
           nn.Conv2d(channels, channels, kernel_size=3, padding=1),
           nn.BatchNorm2d(channels, affine=False),
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
        #elif isinstance(layer, nn.BatchNorm2d):
        #  nn.init.constant_(layer.weight, 1)
        #  nn.init.constant_(layer.bias, 0)


    def forward(self, x):
        dim = x.shape[2] #Keep track of the original dimension of the input to remove padding at the end
        x = self.pad(x)

        #Encoder
        xe1 = self.e1(x)
        x = self.pool1(xe1)

        xe2 = self.e2(x)
        x = self.pool2(xe2)

        xe3 = self.e3(x)
        x = self.pool3(xe3)

        xe4 = self.e4(x)
        x = self.pool4(xe4)

        x = self.e5(x)

        #Decoder
        x = self.upconv1(x)
        x = torch.cat([x, xe4], dim=1)
        x = self.d1(x)

        x = self.upconv2(x)
        x = torch.cat([x, xe3], dim=1)
        x = self.d2(x)

        x = self.upconv3(x)
        x = torch.cat([x, xe2], dim=1)
        x = self.d3(x)

        x = self.upconv4(x)
        x = torch.cat([x, xe1], dim=1)
        x = self.d4(x)

        x = self.out(x)

        x = x[:, :, :dim, :dim]

        return x



#### PRE- AND POST-PROCESSING ####
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
    coding = torch.tensor([
        [1, 0, 0, 0, 0, 0, 0, 0],  # invalid pairing
        [0, 1, 0, 0, 0, 0, 0, 0],  # unpaired
        [0, 0, 1, 0, 0, 0, 0, 0],  # GC
        [0, 0, 0, 1, 0, 0, 0, 0],  # CG
        [0, 0, 0, 0, 1, 0, 0, 0],  # UG
        [0, 0, 0, 0, 0, 1, 0, 0],  # GU
        [0, 0, 0, 0, 0, 0, 1, 0],  # UA
        [0, 0, 0, 0, 0, 0, 0, 1],  # AU
    ], dtype=torch.float32, device=device)

    basepairs = ["GC", "CG", "UG", "GU", "UA", "AU"]

    N = len(sequence)

    # Create an array filled with "invalid pairing" vectors
    matrix = coding[0].repeat(N, N, 1)

    # Update the diagonal with "unpaired" vectors
    matrix[torch.arange(N), torch.arange(N)] = coding[1]
    
    # Update base pair positions directly
    for i in range(N):
        for j in range(N):
            pair = sequence[i] + sequence[j]
            if pair in basepairs and abs(i-j) >=4:
                matrix[i, j, :] = coding[basepairs.index(pair)+2]

    return matrix.permute(2, 0, 1).unsqueeze(0)

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

def mask(matrix: torch.Tensor, sequence: str, device: str) -> torch.Tensor:
    """
    Takes a sequence and returns a mask tensor with 1s in the positions where a base can pair with another base and for unpaired bases.

    Parameters:
    - matrix (torch.Tensor): The matrix to prepare.
    - sequence (str): The sequence that the matrix was generated from.
    - device (str): The device to use for the matrix.

    Returns:
    - torch.Tensor: The prepared matrix.
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

def blossom_postprocessing(matrix: torch.Tensor,  device: str) -> torch.Tensor: 
    """
    Postprocessing function that takes a matrix and returns a matrix.
    The function uses the blossom algorithm to find the maximum weight matching in the graph representation of the matrix, with copying of the matrix to allow for self-pairing.
    The functions used are modified version of NetworkX functions, and are implemented in the blossom.py file.
    The functions has sequence as input, but does not use it. It is provided to make the function compatible with other postprocessing functions.

    Parameters:
    - matrix (torch.Tensor): The matrix to postprocess.

    Returns:
    - torch.Tensor: The postprocessed matrix.
    """
    n = matrix.shape[0]

    mask = torch.eye(n, device=device)

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


### HANDLING FILES AND COMMAND LINE INPUT ###
def prepare_sequence(sequence: str) -> str: 
    """
    Function that prepares a sequence for input to the model.
    The function converts the sequence to uppercase and replaces all Ts with Us.
    The function also checks that the sequence only contains valid characters.

    Parameters:
    - sequence (str): The sequence to prepare.

    Returns:
    - str: The prepared sequence.
    """
    valid_characters = ['A', 'U', 'C', 'G', 'N', 'M', 'Y', 'W', 'V', 'K', 'R', 'I', 'X', 'S', 'D', 'P', 'B', 'H']
    sequence = sequence.upper().replace('T', 'U')

    for char in sequence:
        if char not in valid_characters:
            raise ValueError(f"Invalid character '{char}' found in the sequence.")

    return sequence

def read_fasta(input: str) -> str:
    """
    Reads in a FASTA-file and returns the sequence
    If there is more than one sequence in the FASTA file an error is raised 

    Args: 
        input: path to input sequence
    """
    records = list(SeqIO.parse(input, 'fasta'))
    
    if len(records) > 1: 
        raise ValueError("FASTA file contains more than one sequence")

    return str(records[0].seq), records[0].id

def write_ct(outputfile: str, sequence: str, output: torch.Tensor, seq_name: str) -> None:
   """
   Writes the output to a ct file.

   Parameters:
    - outputfile (str): The output file to write to.
    - sequence (str): The sequence that the output was generated from.
    - output (torch.Tensor): The structure in matrix format.
    - seq_name (str): The name of the sequence.

    Returns:
    - None
   """
   pairs = torch.nonzero(output)

   ct = [[str(i+1), sequence[i], str(i), str(i+2), str(0), str(i+1)] for i in range(len(sequence))]
   
   for row, col in pairs:
       if not row == col:
        ct[row][4] = str(col.item()+1)

   with open(outputfile, 'w') as f:
       f.write(f'{len(sequence)}\tENERGY =\t?\t{seq_name}\n')
       f.write('\n'.join(['\t'.join(line) for line in ct]))
       f.write('\n')



def write_bpseq(outputfile: str, sequence: str, output: torch.Tensor, seq_name: str) -> None:
    """
    Writes the output to a bpseq file.

    Parameters:
    - outputfile (str): The output file to write to.
    - sequence (str): The sequence that the output was generated from.
    - output (torch.Tensor): The structure in matrix format.
    - seq_name (str): The name of the sequence (not used).

    Returns:
    - None
    """
    pairs = torch.nonzero(output)

    bpseq = [[str(i+1), sequence[i], str(0)] for i in range(len(sequence))]
    
    for row, col in pairs:
        if not row == col:
            bpseq[row][2] = str(col.item()+1)
    
    with open(outputfile, 'w') as f:
        f.write(f'Filename: {outputfile}\nOrganism: Unknown\nAccession Number: 000000\nCitation and related information available at?\n')
        f.write('\n'.join([' '.join(line) for line in bpseq]))
        f.write('\n')


def write_to_stdout(outputfile: str, sequence: str, output: torch.Tensor, seq_name: str) -> None:
    """
    Writes the output to stdout.

    Parameters:
    - outputfile (str): The output file to write to (not used).
    - sequence (str): The sequence that the output was generated from.
    - output (torch.Tensor): The structure in matrix format.
    - seq_name (str): The name of the sequence.

    Returns:
    - None
    """
    pairs = torch.nonzero(output)

    bpseq = [[str(i+1), sequence[i], str(0)] for i in range(len(sequence))]
    
    for row, col in pairs:
        if not row == col:
            bpseq[row][2] = str(col.item()+1)

    print(f">{seq_name}\n{sequence}\n")
    print('\n'.join([' '.join(line) for line in bpseq]))


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(prog="StructUnet",
                                        description="""
                                        This program predicts the RNA secondary structure of a given sequence.
                                        The program takes a sequence as input and outputs the secondary structure in dot-bracket notation.
                                        Implemented by Maria Eskerod, master thesis at Aarhus University, spring 2024""")
    argparser.add_argument('-i', '--input', metavar='', type=str, help='Input sequence provided in command line')
    argparser.add_argument('-f', '--file', metavar='', type=argparse.FileType('r'), help='Fasta file containing sequence')
    argparser.add_argument('-m', '--multifile', metavar='', type=argparse.FileType('r'), help='Fasta file containing multiple sequences. Output will be written to multiple bpseq files.')
    argparser.add_argument('-o', '--output', metavar='', default=sys.stdout, help='Output file for the secondary structure. Default is stdout. Valid file formats are .dbn, .ct and .bpseq')

    args = argparser.parse_args()

    if args.input: 
        sequence = prepare_sequence(args.input)
        name = "User inputted sequence"

    if args.file:
       sequence, name = read_fasta(args.file)
       sequence = prepare_sequence(sequence)

    if args.output != sys.stdout:
        output_map = {'.ct': write_ct, '.bpseq': write_bpseq}
        file_type = os.path.splitext(args.output)[1]
        if file_type not in output_map:
            raise ValueError(f"Invalid output file format. Valid file formats are .dbn, .ct and .bpseq. {file_type} is not valid")
        to_outputfile = output_map[file_type]
    else:
        to_outputfile = write_to_stdout


    #Load model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('-- Loading model --')
    model = RNA_Unet()
    model.load_state_dict(torch.load('RNA_Unet.pth', map_location=torch.device(device)))
    model.to(device)

    if args.multifile: 
        print('--Predicting--')
        os.makedirs('StructUnet_predictions', exist_ok=True)
        progress_bar = tqdm(total=len(list(SeqIO.parse(args.multifile, 'fasta'))), unit='seq', file=sys.stdout)
        for record in SeqIO.parse(args.multifile, 'fasta'):
            sequence = prepare_sequence(str(record.seq))
            name = record.id
            input = make_matrix_from_sequence_8(sequence, device=device).to(device)
            output = model(input).squeeze(0).squeeze(0).detach()
            output = mask(output, sequence, device)
            output = blossom_postprocessing(output, device)
            write_bpseq(f'StructUnet_predictions/{name}.bpseq', sequence, output, name)
            progress_bar.update(1)
        progress_bar.close()

    else:
        print('-- Predicting --')
        start_time = time.time()
        input = make_matrix_from_sequence_8(sequence, device=device).to(device)
        output = model(input).squeeze(0).squeeze(0).detach()
        output = mask(output, sequence, device)
        output = blossom_postprocessing(output, device)
        total_time = time.time() - start_time

        to_outputfile(args.output, sequence, output, name)
        print(f'-- Prediction done in {total_time:.2f} seconds --')