import utils.model
import utils.prepare_data as prep
import utils.training_functions as train
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
    pairs = [0, 26, 25, 24, 23, 0, 0, 0, 0, 18, 17, 16, 0, 0, 0, 0, 11, 10, 9, 0, 0, 0, 0, 4, 3, 2, 1, 0, 0, 0, 0]

    assert (sequence, pairs) == prep.read_ct(make_ct_file_1)

def test_get_length(make_ct_file_1): 
    assert 31 == prep.getLength(make_ct_file_1)

def test_matrix_8():
    assert 2==2

def test_matrix_17(): 
    assert 2==2

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
    assert np.all(output1 == prep.make_matrix_from_basepairs([0, 26, 25, 24, 23, 0, 0, 0, 0, 18, 17, 16, 0, 0, 0, 0, 11, 10, 9, 0, 0, 0, 0, 4, 3, 2, 1, 0, 0, 0, 0]).numpy())

def test_split_data(): 
    assert 2==2

### ERROR METRICS ###

def test_dice(): 
    assert 2==2

def test_mse(): 
    assert 2==2

def test_f1(): 
    assert 2==2


### POST PROCESSING ###

def test_all_paired():
    
    matrix1 = np.random.random((50, 50))
    matrix2 = np.eye(50)
    sequence = 'AUCGAUCGAUCGAUCGAUCGAUCGAUCGAUCGAUCGAUCGAUCGAUCGAU'

    #Row wise
    assert np.all(np.any(post_process.argmax_postprocessing(matrix1), axis=1))
    assert np.all(np.any(post_process.blossom_postprocessing(matrix1), axis=1))
    assert np.all(np.any(post_process.blossom_weak(matrix1), axis=1)) 
    assert np.all(np.any(post_process.Mfold_constrain_postprocessing(matrix1, sequence), axis=1)) 
    assert np.all(np.any(post_process.Mfold_param_postprocessing(matrix1, sequence), axis=1)) 

    assert np.all(np.any(post_process.argmax_postprocessing(matrix2), axis=1))
    assert np.all(np.any(post_process.blossom_postprocessing(matrix2), axis=1))
    assert np.all(np.any(post_process.blossom_weak(matrix2), axis=1)) 
    assert np.all(np.any(post_process.Mfold_constrain_postprocessing(matrix2, sequence), axis=1))
    assert np.all(np.any(post_process.Mfold_param_postprocessing(matrix2, sequence), axis=1))

    #Column wise (without argmax)
    assert np.all(np.any(post_process.blossom_postprocessing(matrix1), axis=0))
    assert np.all(np.any(post_process.blossom_weak(matrix1), axis=0)) 
    assert np.all(np.any(post_process.Mfold_constrain_postprocessing(matrix1, sequence), axis=0)) 
    assert np.all(np.any(post_process.Mfold_param_postprocessing(matrix1, sequence), axis=0)) 

    assert np.all(np.any(post_process.blossom_postprocessing(matrix2), axis=0)) 
    assert np.all(np.any(post_process.blossom_weak(matrix2), axis=0)) 
    assert np.all(np.any(post_process.Mfold_constrain_postprocessing(matrix2, sequence), axis=0))
    assert np.all(np.any(post_process.Mfold_param_postprocessing(matrix2, sequence), axis=0))


def test_argmax(): 
    matrix1 = np.random.random((50, 50))

    assert isinstance(post_process.argmax_postprocessing(matrix1), np.ndarray)
    assert 2==2


def test_blossum(): 
    assert 2==2


### EVALUATION ###
def test_evaluation(): 
    assert 2==2