
import os, pickle
from gwf import Workflow, AnonymousTarget

### EXPERIMENTS ###
def make_experiment_data(matrix_type): 
    """
    Make data for experiments with either 8, 9 or 17 channels. 
    Test sets contains a sequences under 500 and no more than 5000 sequences from each family.
    """
    inputs = [os.path.join('data', 'RNAStralign.tar.gz')]
    outputs = [os.path.join('data', f'experiment{matrix_type[0]}{'_wo_unpaired' if not matrix_type[1] else ''}.tar.gz')]
    options = {"memory":"16gb", "walltime":"03:00:00", "account":"RNA_Unet"}
    spec = """echo "Job ID: $SLURM_JOB_ID"
    echo "Output folder: data/experiment{matrix_type}{unpaired_str}\n"
    python3 scripts/experiment_files.py {matrix_type} {unpaired}
    tar -czf data/experiment{matrix_type}{unpaired_str}.tar.gz data/experiment{matrix_type}{unpaired_str}
    rm -r data/experiment{matrix_type}{unpaired_str}""".format(matrix_type = matrix_type[0], unpaired = matrix_type[1], unpaired_str = '_wo_unpaired' if not matrix_type[1] else '')
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
    inputs = ['data/complete_set.tar.gz'] #TODO - Change to the correct inputs
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

### OTHER METHODS ###
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

def predict_cnnfold(files, data_set = 'RNAStrAlign', input_dir = 'data/test_files', output_dir = ''): 
    """
    Predict structure with CNNfold
    """
    inputs = [file for file in files]
    outputs = [file.replace(input_dir, f'steps/CNNfold{output_dir}') for file in files]
    options = {"memory":"16gb", "walltime":"18:00:00", "account":"RNA_Unet"}
    spec = """echo "Job ID: $SLURM_JOB_ID\n"

    python3 ../CNNfold/cnnfold_predict.py {data_set}
    mkdir {output_dir}
    mv results_CNNfold/* {output_dir}/
    rm -r results_CNNfold""".format(data_set = data_set, output_dir = 'steps/CNNfold' if data_set == 'RNAStrAlign' else 'steps/CNNfold_archive')
    return AnonymousTarget(inputs=inputs, outputs=outputs, options=options, spec=spec)

def predict_vienna(files, data_set = 'RNAStrAlign', input_dir = 'data/test_files', output_dir = ''): 
    """
    Predict structure with viennaRNA
    """
    inputs = [file for file in files]
    outputs = [file.replace(input_dir, f'steps/viennaRNA{output_dir}') for file in files]
    options = {"memory":"8gb", "walltime":"2:00:00", "account":"RNA_Unet"}
    spec = """echo "Job ID: $SLURM_JOB_ID\n"
    
    python3 other_methods/vienna_mfold.py {data_set}""".format(data_set = data_set)
    return AnonymousTarget(inputs=inputs, outputs=outputs, options=options, spec=spec)

def predict_nussinov(files, data_set = 'RNAStrAlign', input_dir = 'data/test_files', output_dir = ''):
    """
    Predict structure with Nussinov algorithm
    """
    inputs = [file for file in files]
    outputs = [file.replace(input_dir, f'steps/nussinov{output_dir}') for file in files]
    options = {"memory":"8gb", "walltime":"12:00:00", "account":"RNA_Unet"}
    spec = """echo "Job ID: $SLURM_JOB_ID\n"
    python3 other_methods/nussinov.py {data_set}""".format(data_set = data_set)
    return AnonymousTarget(inputs=inputs, outputs=outputs, options=options, spec=spec)

def predict_contrafold(files, data_set = 'RNAStrAlign', input_dir = 'data/test_files', output_dir = ''):
    """
    Predict structure with Contrafold
    """
    inputs = [file for file in files]
    outputs = [file.replace(input_dir, f'steps/contrafold{output_dir}') for file in files]
    options = {"memory":"8gb", "walltime":"6:00:00", "account":"RNA_Unet"} 
    spec = """echo "Job ID: $SLURM_JOB_ID\n"
    python3 other_methods/contrafold.py {data_set}""".format(data_set = data_set)
    return AnonymousTarget(inputs=inputs, outputs=outputs, options=options, spec=spec)


### EVALUATION ###

def evaluate_hotknots():
    """
    Evaluate the hotknots post-processing with different hyper-parameters
    """
    inputs = [os.path.join('data', 'test_RNA_sample', file) for file in os.listdir('data/test_RNA_sample')]
    outputs = ['results/F1_hotknots.csv', 
               'figures/F1_hotknots.png', 
               'results/time_hotknots.csv']
    options = {"memory":"8gb", "walltime":"48:00:00", "account":"RNA_Unet", "cores":1}
    spec = """echo "Job ID: $SLURM_JOB_ID\n"
    python3 scripts/evaluate_hotknot.py"""
    return AnonymousTarget(inputs=inputs, outputs=outputs, options=options, spec=spec)

def evaluate_postprocessing_under600(files): 
    """
    Evaluate all the implemented post-processing methods and compare them
    """
    inputs = [os.path.join('RNA_Unet.pth')] + files
    outputs = [os.path.join('results', 'average_scores_postprocess_under600.csv'), 
               os.path.join('figures', 'evaluation_postprocess_under600.png'),
               os.path.join('results', 'evaluation_postprocess_under600.csv')]
    options = {"memory":"16gb", "walltime":"24:00:00", "account":"RNA_Unet","cores":15} 
    spec = """echo "Job ID: $SLURM_JOB_ID\n"
    python3 scripts/evaluate_postprocessing_under600.py"""
    return AnonymousTarget(inputs=inputs, outputs=outputs, options=options, spec=spec) 

def test_model_cpu(files): 
    """
    Test the model of the test set and time it
    """
    inputs = ['RNA_Unet.pth'] + files
    outputs = [file.replace('data/test_files', 'steps/RNA_Unet') for file in files] 
    options = {"memory":"16gb", "walltime":"12:00:00", "account":"RNA_Unet"}
    spec = """echo "Job ID: $SLURM_JOB_ID\n"
    python3 scripts/predict_test.py"""
    return AnonymousTarget(inputs=inputs, outputs=outputs, options=options, spec=spec)

def test_model_gpu(files): 
    """
    Test the model of the test set and time it
    """
    inputs = ['RNA_Unet.pth'] + files
    outputs = ['results/times_final_cuda.csv', 'figures/time_final_cuda.png'] 
    options = {"memory":"16gb", "walltime":"72:00:00", "account":"RNA_Unet", "gres":"gpu:1", "queue":"gpu"} #NOTE - Think about memory and walltime
    spec = """CONDA_BASE=$(conda info --base)
    source $CONDA_BASE/etc/profile.d/conda.sh
    conda activate RNA_Unet
    
    echo "Job ID: $SLURM_JOB_ID\n"
    nvidia-smi -L
    export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
    nvcc --version
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    python3 scripts/predict_test.py"""
    return AnonymousTarget(inputs=inputs, outputs=outputs, options=options, spec=spec)

def time_final_model(files):
    """
    Find the average time of prediction using the RNA_Unet model
    """
    inputs = ['RNA_Unet.pth'] + files
    outputs = ['results/time_final_cpu.csv', 'figures/time_final_cpu.png']
    options = {"memory":"16gb", "walltime":"52:00:00", "account":"RNA_Unet"} #NOTE - Think about memory and walltime
    spec = """echo "Job ID: $SLURM_JOB_ID\n"
    python3 scripts/time_final.py"""
    return AnonymousTarget(inputs=inputs, outputs=outputs, options=options, spec=spec)

def compare_methods_under600(methods, files):
    """
    Compare the different previous methods with the RNAUnet
    """
    inputs = [file.replace('data/test_files', f'steps/{method}') for file in files for method in methods]
    outputs = ['results/testscores_under600.csv',
               'results/pseudoknot_F1.csv',
               'results/average_scores.csv',
               'figures/evaluation_predictions_under600.png'] 
    options = {"memory":"8gb", "walltime":"1:00:00", "account":"RNA_Unet"} 
    spec = """echo "Job ID: $SLURM_JOB_ID\n"
    python3 scripts/compare_predictions_under600.py"""
    return AnonymousTarget(inputs=inputs, outputs=outputs, options=options, spec=spec) 

def compare_methods_over600(methods, files):
    """
    Compare the different previous methods with the RNAUnet
    """
    inputs = [file.replace('data/test_files', f'steps/{method}') for file in files for method in methods] + ['results/pseudoknot_F1.csv', 'results/average_scores.csv', 'results/testscores_under600.csv']
    outputs = ['results/testscores_over600.csv',
               'results/family_scores.csv',
               'figures/evaluation_predictions_all.png',
               'figures/per_sequence_F1.png'] 
    options = {"memory":"8gb", "walltime":"2:00:00", "account":"RNA_Unet"} 
    spec = """echo "Job ID: $SLURM_JOB_ID\n"
    python3 scripts/compare_predictions_over600.py"""
    return AnonymousTarget(inputs=inputs, outputs=outputs, options=options, spec=spec) 

### EVALUATION ON ARCHIVE II ###
def convert_archiveII():
    """
    Convert the archive II data to matrices
    """
    inputs = [os.path.join('data', 'archiveii.csv')]
    outputs = ['data/archiveii.pkl',
               'figures/families_archiveii.png',
               'figures/lengths_archiveii.png']
    options = {"memory":"16gb", "walltime":"6:00:00", "account":"RNA_Unet"}
    spec = """echo "Job ID: $SLURM_JOB_ID\n"
    python3 scripts/prepare_archiveii.py"""
    return AnonymousTarget(inputs=inputs, outputs=outputs, options=options, spec=spec)

def predict_RNAUnet(input_file, output_dir, input_dir): 
    """
    Predict structure with RNAUnet
    """
    files = pickle.load(open(input_file, 'rb'))
    inputs = [file for file in files]
    outputs = [file.replace(input_dir, output_dir) for file in files]
    options = {"memory":"64gb", "walltime":"18:00:00", "account":"RNA_Unet"}
    spec = """echo "Job ID: $SLURM_JOB_ID\n"

    python3 scripts/predict_from_file.py {file}
    mkdir {output_dir}
    mv results_RNAUnet/* {output_dir}/
    rm -r results_RNAUnet""".format(file = input_file, output_dir = output_dir) 
    return AnonymousTarget(inputs=inputs, outputs=outputs, options=options, spec=spec)

def compare_archiveII(methods, files):
    """
    Compare the different previous methods with the RNAUnet
    """
    inputs = [file.replace('data/archiveii', f'steps/{method}_archive') for file in files for method in methods]
    outputs = ['results/testscores_archive.csv',
               'results/family_scores_archive.csv',
               'results/pseudoknot_F1_archive.csv',
               'figures/evaluation_predictions_all_archive.png',
               'figures/per_sequence_F1_archive.png'] 
    options = {"memory":"8gb", "walltime":"5:00:00", "account":"RNA_Unet"} 
    spec = """echo "Job ID: $SLURM_JOB_ID\n"
    python3 scripts/compare_predictions_archive.py"""
    return AnonymousTarget(inputs=inputs, outputs=outputs, options=options, spec=spec)


#ADDITIONAL ANALYSIS
def evaluate_random_predictions(files): 
    """
    Evaluate random predictions
    """
    inputs = ['RNA_Unet.pth'] + files
    outputs = ['results/testscores_random.csv',
               'results/pseudoknot_F1_random.csv',
               'results/average_scores_random.csv',
               'results/family_scores_random.csv',
               'results/family_scores_random_archive.csv',
               'results/family_scores_random_align.csv']
    options = {"memory":"16gb", "walltime":"12:00:00", "account":"RNA_Unet", "cores":15}
    spec = """echo "Job ID: $SLURM_JOB_ID\n"
    python3 scripts/random_predictions.py"""
    return AnonymousTarget(inputs=inputs, outputs=outputs, options=options, spec=spec)



### WORKFLOW ###
gwf = Workflow()

#Make data for experiments
for matrix_type in [(8, True), (9, True), (17, True), (8, False)]:
    gwf.target_from_template(f'experiment_data_{matrix_type[0]}_unpaired_{matrix_type[1]}', make_experiment_data(matrix_type))

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


files_under600 = [test_files[i] for i in under_600]
gwf.target_from_template('predict_hotknots', predict_hotknots(files_under600))
gwf.target_from_template('predict_ufold', predict_ufold(files_under600))


datasets = [(test_files, 'RNAStrAlign', 'data/test_files', ''), (pickle.load(open('data/archiveii.pkl', 'rb')), 'ArchiveII', 'data/archiveii', '_archive')]

for dataset in datasets: 
    gwf.target_from_template(f'predict_vienna_{dataset[1]}', predict_vienna(dataset[0], dataset[1], dataset[2], dataset[3]))
    gwf.target_from_template(f'predict_nussinov_{dataset[1]}', predict_nussinov(dataset[0], dataset[1], dataset[2], dataset[3]))
    gwf.target_from_template(f'predict_contrafold_{dataset[1]}', predict_contrafold(dataset[0], dataset[1], dataset[2], dataset[3]))
    gwf.target_from_template(f'predict_cnnfold_{dataset[1]}', predict_cnnfold(dataset[0], dataset[1], dataset[2], dataset[3]))


gwf.target_from_template('compare_postprocessing_under600', evaluate_postprocessing_under600(pickle.load(open('data/valid_under_600.pkl', 'rb'))))

#Evaluate on test set
gwf.target_from_template('evaluate_RNAUnet_cpu', test_model_cpu(test_files))
gwf.target_from_template('evaluate_RNAUnet_gpu', test_model_gpu(test_files))

gwf.target_from_template('time_final_model', time_final_model(test_files))

methods_under600 = ['hotknots', 'Ufold']
methods = ['CNNfold', 'viennaRNA', 'RNA_Unet', 'nussinov', 'contrafold']
files_over600 = [test_files[i] for i in range(len(test_files)) if i not in under_600]

gwf.target_from_template('compare_methods_under600', compare_methods_under600(methods_under600 + methods, files_under600))
gwf.target_from_template('compare_methods_over600', compare_methods_over600(methods, files_over600))


#Evaluate on archive II
gwf.target_from_template('convert_archiveII', convert_archiveII())
gwf.target_from_template('predict_RNAUnet_archiveII', predict_RNAUnet('data/archiveii.pkl', 'steps/RNAUnet_archive', 'data/archiveii'))

methods = ['CNNfold', 'viennaRNA', 'RNAUnet', 'nussinov', 'contrafold']
gwf.target_from_template('compare_archiveII', compare_archiveII(methods, pickle.load(open('data/archiveii.pkl', 'rb'))))

#Additional analysis
gwf.target_from_template('evaluate_random_predictions', evaluate_random_predictions(pickle.load(open('data/test.pkl', 'rb'))+pickle.load(open('data/archiveii.pkl', 'rb'))))