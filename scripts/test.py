import utils.prepare_data as prep
import scripts.utils.model_and_training as train
import utils.post_processing as post_process

import numpy as np
import torch, pytest, os

### FUNCTIONS USED IN TESTS ###
@pytest.fixture
def make_ct_file_1(tmpdir): 
    """
    """
    file_content = """31
    1 C 0 0 0
    2 G 0 0 27
    3 U 0 0 26
    4 G 0 0 25
    5 U 0 0 24
    6 C 0 0 0
    7 A 0 0 0
    8 G 0 0 0
    9 G 0 0 0
    10 U 0 0 19
    11 C 0 0 18
    12 C 0 0 17
    13 G 0 0 0
    14 G 0 0 0
    15 A 0 0 0
    16 A 0 0 0
    17 G 0 0 12
    18 G 0 0 11
    19 A 0 0 10
    20 A 0 0 0
    21 G 0 0 0
    22 C 0 0 0
    23 A 0 0 0
    24 G 0 0 5
    25 C 0 0 4
    26 A 0 0 3
    27 C 0 0 2
    28 U 0 0 0
    29 A 0 0 0
    30 A 0 0 0
    31 C 0 0 0"""
    
    ct_file_path = tmpdir.join("test.ct")
    with open(ct_file_path, "w") as f: 
        f.write(file_content)
    return str(ct_file_path)



### FUNCTIONS FOR PREPARING/HANDLING DATA
def test_read_ct(make_ct_file_1):
    sequence = 'CGUGUCAGGUCCGGAAGGAAGCAGCACUAAC'
    pairs = [None, 26, 25, 24, 23, None, None, None, None, 18, 17, 16, None, None, None, None, 11, 10, 9, None, None, None, None, 4, 3, 2, 1, None, None, None, None]

    assert (sequence, pairs) == prep.read_ct(make_ct_file_1)

def test_get_length(make_ct_file_1): 
    assert 31 == prep.getLength(make_ct_file_1)

def test_score_matrix(): 
    sequence = 'GCGGUUAUAG'
    matrix = np.round(prep.calculate_score_matrix(sequence), 2)
    true = np.array([[0, 0, 0, 0, 2.62, 0.8, 0, 0.8, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 3.0],
                     [0, 0, 0, 0, 0, 0, 0, 0.8, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 2.01, 0, 0],
                     [2.62, 0, 0, 0, 0, 0, 0, 0, 2.0, 2.31],
                     [0.8, 0, 0, 0, 0, 0, 0, 0, 0, 0.8],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0.8, 0, 0.8, 2.01, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 2.0, 0, 0, 0, 0, 0],
                     [0, 3.0, 0, 0, 2.31, 0.8, 0, 0, 0, 0]], dtype='float32')    
    assert np.allclose(matrix, true.reshape(10, 10, 1))

def test_matrix_8():
    sequence = 'CGUGUCAGGUCCGGAAGGAAGCAGCACUAAC'
    matrix = prep.make_matrix_from_sequence_8(sequence)

    assert torch.all(torch.eq(matrix[:, 0, 0], torch.tensor([0, 1, 0, 0, 0, 0, 0, 0], dtype=torch.float32)))
    assert torch.all(torch.eq(matrix[:, 1, 0], torch.tensor([1, 0, 0, 0, 0, 0, 0, 0], dtype=torch.float32)))
    assert torch.all(torch.eq(matrix[:, 1, -1], torch.tensor([0, 0, 1, 0, 0, 0, 0, 0], dtype=torch.float32)))
    assert torch.all(torch.eq(matrix[:, 0, -8], torch.tensor([0, 0, 0, 1, 0, 0, 0, 0], dtype=torch.float32)))
    assert torch.all(torch.eq(matrix[:, 2, -8], torch.tensor([0, 0, 0, 0, 1, 0, 0, 0], dtype=torch.float32)))
    assert torch.all(torch.eq(matrix[:, 1, -4], torch.tensor([0, 0, 0, 0, 0, 1, 0, 0], dtype=torch.float32)))
    assert torch.all(torch.eq(matrix[:, 2, -2], torch.tensor([0, 0, 0, 0, 0, 0, 1, 0], dtype=torch.float32)))
    assert torch.all(torch.eq(matrix[:, 6, -4], torch.tensor([0, 0, 0, 0, 0, 0, 0, 1], dtype=torch.float32)))

def test_matrix_17(): 
    sequence = 'AUCGNMYWVKRIXSDPBH'

    onehot  = prep.sequence_onehot(sequence)

    seq_dict = [
        np.array([1,0,0,0], dtype=np.float32),
        np.array([0,1,0,0], dtype=np.float32),
        np.array([0,0,1,0], dtype=np.float32),
        np.array([0,0,0,1], dtype=np.float32),
        np.array([0,0,0,0], dtype=np.float32),
        np.array([1,0,1,0], dtype=np.float32),
        np.array([0,1,1,0], dtype=np.float32),
        np.array([1,0,0,0], dtype=np.float32),
        np.array([1,0,1,1], dtype=np.float32),
        np.array([0,1,0,1], dtype=np.float32),
        np.array([1,0,0,1], dtype=np.float32),
        np.array([0,0,0,0], dtype=np.float32),
        np.array([0,0,0,0], dtype=np.float32),
        np.array([0,0,1,1], dtype=np.float32),
        np.array([1,1,0,1], dtype=np.float32),
        np.array([0,0,0,0], dtype=np.float32),
        np.array([0,1,1,1], dtype=np.float32),
        np.array([1,1,1,0], dtype=np.float32)]
    
    for i, _ in enumerate(sequence): 
        assert np.all(onehot[i] == seq_dict[i])
    
    matrix = prep.input_representation(sequence)
    assert np.array_equal(matrix[0, 0, :], np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32))
    assert np.array_equal(matrix[17, 0, :], np.array([1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32))
    assert np.array_equal(matrix[17, 17, :], np.array([1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0], dtype=np.float32))

def test_output_matrix(): 
    output1 = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0 ],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0 ],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0 ],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0 ],
                        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ],
                        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ],
                        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ],
                        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0 ],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0 ],
                        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ],
                        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ],
                        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ],
                        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0 ],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0 ],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0 ],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 ]], dtype='float32')
    assert np.all(output1 == prep.make_matrix_from_basepairs([None, 26, 25, 24, 23, None, None, None, None, 18, 17, 16, None, None, None, None, 11, 10, 9, None, None, None, None, 4, 3, 2, 1, None, None, None, None]).numpy())

### ERROR METRICS ###

def test_dice(): 
    #Test on binary
    inputs = torch.tensor([[[[1, 1, 1, 1]]]])
    targets = torch.tensor([[[[1, 0, 1, 0]]]])
    assert round(train.dice_loss(inputs, targets).item(), 2) == 0.33

    #Test zero-case
    inputs = torch.tensor([[[[0, 0, 0, 0]]]])
    targets = torch.tensor([[[[1, 1, 1, 1]]]])
    assert round(train.dice_loss(inputs, targets).item()) == 1

    #Test perfect case
    inputs = torch.tensor([[[[1, 1, 1, 1]]]])
    targets = torch.tensor([[[[1, 1, 1, 1]]]])
    assert round(train.dice_loss(inputs, targets).item()) == 0

    #Test on non-binary
    inputs = torch.tensor([[[[0.2, 0.4, 0.9, 0.1]]]])
    targets = torch.tensor([[[[0, 0, 1, 0]]]])
    assert round(train.dice_loss(inputs, targets).item(), 2) == 0.11

    #Test on more compex
    inputs = torch.tensor([[[[0.2, 0.4, 0.9, 0.1],
                             [0.1, 0.4, 0.66, 0.0],
                             [0.99, 0.3, 0.86, 0.1],
                             [0.87, 0.67, 0.99, 0.1]]]])
    targets = torch.tensor([[[[0, 0, 1, 0],
                              [0, 0, 1, 0],
                              [1, 0, 0, 0],
                              [1, 0, 1, 0]]]])
    assert round(train.dice_loss(inputs, targets).item(), 2) == 0.17

def test_f1(): 
    # Test functionality
    y_pred = torch.tensor([1, 1, 1, 1])
    y_true = torch.tensor([1, 0, 1, 0])
    assert round(train.f1_score(y_pred, y_true).item(), 2) == 0.67

    #Test zero-case
    y_pred = torch.tensor([0, 0, 0, 0])
    y_true = torch.tensor([1, 1, 1, 1])
    assert round(train.f1_score(y_pred, y_true).item()) == 0

    #Test perfect case
    y_pred = torch.tensor([1, 1, 1, 1])
    y_true = torch.tensor([1, 1, 1, 1])
    assert round(train.f1_score(y_pred, y_true).item()) == 1

    #Test more complex case
    y_pred = torch.tensor([[1, 0, 1, 0],
                          [1, 0, 1, 0],
                          [1, 0, 1, 0],
                          [1, 0, 1, 0]])
    y_true = torch.tensor([[1, 1, 1, 1],
                           [0, 0, 0, 0], 
                           [1, 1, 1, 1],
                           [0, 0, 0, 0]])
    assert round(train.f1_score(y_pred, y_true).item(), 2) == 0.5


### POST PROCESSING ###

def test_all_paired():
    
    matrix1 = np.random.random((50, 50))
    matrix2 = np.eye(50)
    sequence = 'AUCGAUCGAUCGAUCGAUCGAUCGAUCGAUCGAUCGAUCGAUCGAUCGAU'

    #Row wise
    assert np.all(np.any(post_process.argmax_postprocessing(matrix1, 'sequence'), axis=1))
    assert np.all(np.any(post_process.blossom_postprocessing(matrix1, 'sequence'), axis=1))
    assert np.all(np.any(post_process.blossom_weak(matrix1, 'sequence'), axis=1)) 
    assert np.all(np.any(post_process.Mfold_constrain_postprocessing(matrix1, sequence), axis=1)) 
    assert np.all(np.any(post_process.Mfold_param_postprocessing(matrix1, sequence), axis=1)) 

    assert np.all(np.any(post_process.argmax_postprocessing(matrix2, 'sequence'), axis=1))
    assert np.all(np.any(post_process.blossom_postprocessing(matrix2, 'sequence'), axis=1))
    assert np.all(np.any(post_process.blossom_weak(matrix2, 'sequence'), axis=1)) 
    assert np.all(np.any(post_process.Mfold_constrain_postprocessing(matrix2, sequence), axis=1))
    assert np.all(np.any(post_process.Mfold_param_postprocessing(matrix2, sequence), axis=1))

    #Column wise (without argmax)
    assert np.all(np.any(post_process.blossom_postprocessing(matrix1, 'sequence'), axis=0))
    assert np.all(np.any(post_process.blossom_weak(matrix1, 'sequence'), axis=0)) 
    assert np.all(np.any(post_process.Mfold_constrain_postprocessing(matrix1, sequence), axis=0)) 
    assert np.all(np.any(post_process.Mfold_param_postprocessing(matrix1, sequence), axis=0)) 

    assert np.all(np.any(post_process.blossom_postprocessing(matrix2, 'sequence'), axis=0)) 
    assert np.all(np.any(post_process.blossom_weak(matrix2, 'sequence'), axis=0)) 
    assert np.all(np.any(post_process.Mfold_constrain_postprocessing(matrix2, sequence), axis=0))
    assert np.all(np.any(post_process.Mfold_param_postprocessing(matrix2, sequence), axis=0))

def test_argmax(): 
    matrix1 = np.random.random((50, 50))

    assert isinstance(post_process.argmax_postprocessing(matrix1, 'sequence'), np.ndarray) #Test that matrix is returned
    assert len(np.nonzero(post_process.argmax_postprocessing(matrix1, 'sequence'))[0]) == 50 #Test correct number of values is in the matrix

    # Test that the functionality is correct
    result = post_process.argmax_postprocessing(matrix1, 'sequence')
    max_indices = np.argmax((matrix1+matrix1.T)/2, axis=1)
    for i, idx in enumerate(max_indices):
        assert result[i, idx] == 1

def test_blossum():
    matrix1 = np.random.random((50, 50))

    result1 = post_process.blossom_postprocessing(matrix1, 'sequence')
    result2 = post_process.blossom_weak(matrix1, 'sequence')
    assert np.allclose(result1, result1.T)
    assert np.allclose(result2, result2.T)

    matrix = np.zeros((3, 3))
    result = post_process.blossom_weak(matrix, 'sequence')
    assert np.array_equal(result, np.eye(3))

def test_Mfold(): 
    sequence = 'CGUGUCAGGUCCGGAAGGAAGCAGCACUAAC'
    pairs = [0, 26, 25, 24, 23, 0, 0, 0, 0, 18, 17, 16, 0, 0, 0, 0, 11, 10, 9, 0, 0, 0, 0, 4, 3, 2, 1, 0, 0, 0, 0]

    matrix = np.zeros((len(pairs), len(pairs)))
    for i in range(len(pairs)):
        if pairs[i] != 0:
            matrix[i, pairs[i]] = 1
        else: 
            matrix[i, i] = 1
    
    result = post_process.Mfold_param_postprocessing(matrix, sequence)

    assert np.array_equal(matrix, result)


### EVALUATION ###
def test_evaluation(): 
    # Test functionality
    y_pred = np.array([1, 1, 1, 1])
    y_true = np.array([1, 0, 1, 0])

    precision, recall, F1 = train.evaluate(y_pred, y_true, include_unpaired=True)

    assert round(precision, 1) == 0.5
    assert round(recall, 1) == 1
    assert round(F1, 2) == 0.67

    #Test zero-case
    y_pred = np.array([0, 0, 0, 0])
    y_true = np.array([1, 1, 1, 1])

    precision, recall, F1 = train.evaluate(y_pred, y_true, include_unpaired=True)

    assert round(precision) == 0
    assert round(recall) == 0
    assert round(F1) == 0

    #Test perfect case
    y_pred = torch.tensor([1, 1, 1, 1])
    y_true = torch.tensor([1, 1, 1, 1])

    precision, recall, F1 = train.evaluate(y_pred, y_true)
    assert round(precision) == 1
    assert round(recall) == 1
    assert round(F1) == 1

    #Test more complex case
    y_pred = torch.tensor([[1, 0, 1, 0],
                          [1, 0, 1, 0],
                          [1, 0, 1, 0],
                          [1, 0, 1, 0]])
    y_true = torch.tensor([[1, 1, 1, 1],
                           [0, 0, 0, 0], 
                           [1, 1, 1, 1],
                           [0, 0, 0, 0]])
    precision, recall, F1 = train.evaluate(y_pred, y_true, include_unpaired=True)
    assert round(precision, 2) == 0.5
    assert round(recall, 2) == 0.5
    assert round(F1, 2) == 0.5
    

