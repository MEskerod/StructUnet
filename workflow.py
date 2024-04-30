
import os, pickle
from gwf import Workflow, AnonymousTarget

### EXPERIMENTS ###
def make_experiment_data(matrix_type): 
    """
    Make data for experiments with either 8, 9 or 17 channels. 
    Test sets contains a sequences under 500 and no more than 5000 sequences from each family.
    """
    inputs = [os.path.join('data', 'RNAStralign.tar.gz')]
    outputs = [os.path.join('data', f'experiment{matrix_type}.tar.gz')]
    options = {"memory":"16gb", "walltime":"03:00:00", "account":"RNA_Unet"}
    spec = """echo "Job ID: $SLURM_JOB_ID\n"
    python3 scripts/experiment_files.py {matrix_type}
    tar -czf data/experiment{matrix_type}.tar.gz data/experiment{matrix_type}
    rm -r data/experiment{matrix_type}""".format(matrix_type = matrix_type)
    return AnonymousTarget(inputs=inputs, outputs=outputs, options=options, spec=spec)

def postprocess_time(): 
    """
    Time postprocessing methods
    """
    inputs = []
    outputs = [os.path.join('results', 'postprocess_time.csv'),
               os.path.join('figures', 'postprocess_time.png')]
    options = {"memory": "16gb", "walltime": "36:00:00", "account":"RNA_Unet", "cores": 5}
    spec = """echo "Job ID: $SLURM_JOB_ID\n"
    python3 scripts/time_postprocessing.py"""
    return AnonymousTarget(inputs=inputs, outputs=outputs, options=options, spec=spec)

def convert_time():
    """
    Time matrix conversion with 8, 9 and 17 channels    
    """ 
    inputs = []
    outputs = [os.path.join('results', 'convert_time.csv'),
               os.path.join('figures', 'convert_time.png')]
    options = {"memory": "16gb", "walltime": "24:00:00", "account":"RNA_Unet"}
    spec = """echo "Job ID: $SLURM_JOB_ID\n"
    python3 scripts/time_matrix_conversion.py"""
    return AnonymousTarget(inputs=inputs, outputs=outputs, options=options, spec=spec)


### TRAINING ###
def make_complete_set(): 
    """
    Convert all data to matrices and save namedtuple as pickle files
    """
    inputs = [os.path.join('data', 'RNAStralign.tar.gz')]
    outputs = [os.path.join('data', 'train.pkl'),
               os.path.join('data', 'valid.pkl'),
               os.path.join('data', 'test.pkl'),
               os.path.join('figures', 'length_distribution.png'),
               os.path.join('figures', 'family_distribution.png')]
    options = {"memory":"16gb", "walltime":"6:00:00", "account":"RNA_Unet", "cores":4}
    spec = """echo "Job ID: $SLURM_JOB_ID\n"
    python3 scripts/complete_dataset.py
    tar -czf data/test_files.tar.gz data/test_files"""
    return AnonymousTarget(inputs=inputs, outputs=outputs, options=options, spec=spec)    

def train_model_small(files): 
    """
    Train the model on the entire data set
    """
    inputs = ['data/complete_set.tar.gz']
    outputs = ['RNA_Unet.pth']
    options = {"memory":"8gb", "walltime":"168:00:00", "account":"RNA_Unet", "gres":"gpu:1", "queue":"gpu"} 
    spec = """CONDA_BASE=$(conda info --base)
    source $CONDA_BASE/etc/profile.d/conda.sh
    conda activate RNA_Unet

    echo "Job ID: $SLURM_JOB_ID\n"
    nvidia-smi -L
    export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
    nvcc --version
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    echo "Training neural network"
    python3 scripts/training.py"""
    return AnonymousTarget(inputs=inputs, outputs=outputs, options=options, spec=spec)

def test_model(files): 
    """
    Test the model of the test set and time it
    """
    inputs = ['RNA_Unet.pth'] + files
    outputs = ['results/test_scores.csv'] + [file.replace('data/test_files', 'steps/RNA_Unet') for file in files] #TODO - Add paths for plots and change path to csv
    options = {"memory":"16gb", "walltime":"24:00:00", "account":"RNA_Unet"} #NOTE - Think about memory and walltime
    spec = """echo "Job ID: $SLURM_JOB_ID\n"
    python3 scripts/predict_test.py"""
    return AnonymousTarget(inputs=inputs, outputs=outputs, options=options, spec=spec)

### EVALUATION ###

def evaluate_hotknots():
    """
    Evaluate the hotknots post-processing with different hyper-parameters
    """
    inputs = [os.path.join('data', 'test_RNA_sample', file) for file in os.listdir('data/test_RNA_sample')]
    outputs = ['results/F1_hotknots.csv', 
               'figures/F1_hotknots.png', 
               'figures/time_hotknots.png', 
               'results/time_hotknots.csv']
    options = {"memory":"8gb", "walltime":"48:00:00", "account":"RNA_Unet", "cores":1}
    spec = """echo "Job ID: $SLURM_JOB_ID\n"
    python3 scripts/evaluate_hotknot.py"""
    return AnonymousTarget(inputs=inputs, outputs=outputs, options=options, spec=spec)

def predict_hotknots(files): 
    """
    Predict structure with hotknots 
    """
    inputs = [files]
    outputs = [os.path.join('steps', 'hotknots', os.path.basename(file)) for file in files]
    options = {"memory":"64gb", "walltime":"32:00:00", "account":"RNA_Unet", "cores":4} 
    spec = """echo "Job ID: $SLURM_JOB_ID\n"
    python3 ../HotKnots/hotknots.py"""
    return AnonymousTarget(inputs=inputs, outputs=outputs, options=options, spec=spec)

def predict_ufold(files): 
    """
    Predict structure with Ufold
    """
    inputs = [files]
    outputs = [file.replace('data/test_files', 'steps/Ufold') for file in files]
    options = {"memory":"8gb", "walltime":"3:00:00", "account":"RNA_Unet"} 
    spec = """echo "Job ID: $SLURM_JOB_ID\n"
    echo "{files}" > input.txt

    CONDA_BASE=$(conda info --base)
    source $CONDA_BASE/etc/profile.d/conda.sh
    conda activate UFold

    python3 ../UFOLD/ufold_predict.py
    mkdir steps/Ufold
    mv results_Ufold/* steps/Ufold/
    rm -r results_Ufold
    rm input.txt""".format(files = '\n'.join(files))
    return AnonymousTarget(inputs=inputs, outputs=outputs, options=options, spec=spec)

def predict_cnnfold(files): 
    """
    Predict structure with CNNfold
    """
    inputs = [file for file in files]
    outputs = [file.replace('data/test_files', 'steps/CNNfold') for file in files]
    options = {"memory":"16gb", "walltime":"18:00:00", "account":"RNA_Unet"}
    spec = """echo "Job ID: $SLURM_JOB_ID\n"

    python3 ../CNNfold/cnnfold_predict.py
    mv results_CNNfold/* steps/CNNfold/
    rm -r results_CNNfold"""
    return AnonymousTarget(inputs=inputs, outputs=outputs, options=options, spec=spec)

def predict_vienna(files): 
    """
    Predict structure with viennaRNA
    """
    inputs = [file for file in files]
    outputs = [file.replace('data/test_files', 'steps/vienna_mfold') for file in files]
    options = {"memory":"8gb", "walltime":"2:00:00", "account":"RNA_Unet"}
    spec = """echo "Job ID: $SLURM_JOB_ID\n"
    
    python3 other_methods/vienna_mfold.py"""
    return AnonymousTarget(inputs=inputs, outputs=outputs, options=options, spec=spec)

def predict_nussinov(files):
    """
    Predict structure with Nussinov algorithm
    """
    inputs = [file for file in files]
    outputs = [file.replace('data/test_files', 'steps/nussinov') for file in files] + ['results/times_nussinov.csv', 'figures/times_nussinov.png']
    options = {"memory":"8gb", "walltime":"12:00:00", "account":"RNA_Unet"}
    spec = """echo "Job ID: $SLURM_JOB_ID\n"
    python3 other_methods/nussinov.py"""
    return AnonymousTarget(inputs=inputs, outputs=outputs, options=options, spec=spec)


def evaluate_postprocessing(files): 
    """
    Evaluate all the implemented post-processing methods and compare them
    """
    inputs = [os.path.join('RNA_Unet.pth')] + files
    outputs = [os.path.join('results', 'average_scores_postprocess.csv'), 
               os.path.join('figures', 'evaluation_postprocess.png'),
               os.path.join('results', 'evalutation_postprocess.csv')]
    options = {"memory":"8gb", "walltime":"72:00:00", "account":"RNA_Unet", "cores":12} 
    spec = """echo "Job ID: $SLURM_JOB_ID\n"
    python3 scripts/evaluate_postprocessing.py"""
    return AnonymousTarget(inputs=inputs, outputs=outputs, options=options, spec=spec) 

def compare_methods(methods_under600, files_under600, methods, files):
    """
    Compare the different previous methods with the RNAUnet
    """
    inputs = [file.replace('data/test_files', f'steps/{method}') for file in files_under600 for method in methods_under600] + [file.replace('data/test_files', f'steps/{method}') for file in files for method in methods]
    outputs = ['results/test_scores_under600.csv',
               'results/test_scores.csv',
               'results/f1_pseudoknots.csv',
               'results/average_scores_methods.csv',
               'results/RNAUnet_family_scores.csv'] #TODO - Add paths for plots
    options = {"memory":"16gb", "walltime":"24:00:00", "account":"RNA_Unet"} #NOTE - Think about memory and walltime
    spec = """echo "Job ID: $SLURM_JOB_ID\n
    python3 scripts/compare_methods.py"""
    return AnonymousTarget(inputs=inputs, outputs=outputs, options=options, spec=spec) 


### WORKFLOW ###
gwf = Workflow()

#Make data for experiments
for matrix_type in [8, 9, 17]:
    gwf.target_from_template(f'experiment_data_{matrix_type}', make_experiment_data(matrix_type))

#Make experiment of post processing time 
gwf.target_from_template('time_postprocess', postprocess_time())

#Make experiment of conversion time
gwf.target_from_template('time_convert', convert_time())


## FOR TRAINING THE ON THE ENTIRE DATA SET
gwf.target_from_template('convert_data', make_complete_set())

gwf.target_from_template('train_RNAUnet', train_model_small(files = pickle.load(open('data/train.pkl', 'rb')) + pickle.load(open('data/valid.pkl', 'rb'))))

gwf.target_from_template('evaluate_hotknots', evaluate_hotknots())



#Predicting with other methods for comparisons
under_600 = pickle.load(open('data/test_under_600.pkl', 'rb'))
test_files = pickle.load(open('data/test.pkl', 'rb'))


files = [test_files[i] for i in under_600]
gwf.target_from_template('predict_hotknots', predict_hotknots(files))
gwf.target_from_template('predict_ufold', predict_ufold(files))

gwf.target_from_template('predict_cnnfold', predict_cnnfold(test_files))
gwf.target_from_template('predict_vienna', predict_vienna(test_files))
gwf.target_from_template('predict_nussinov', predict_nussinov(test_files))


methods_under600 = ['hotknots', 'Ufold']
methods = ['CNNfold', 'vienna_mfold', 'RNAUnet']

gwf.target_from_template('compare_postprocessing', evaluate_postprocessing(pickle.load(open('data/valid.pkl', 'rb'))))
#gwf.target_from_template('evaluate_RNAUnet', test_model(test_files))
#gwf.target_from_template('compare_methods', compare_methods(methods_under600, files, methods, test_files))


