
import os
from gwf import Workflow, AnonymousTarget

### EXPERIMENTS ###
def make_experiment_data(matrix_type): 
    inputs = [os.path.join('data', 'RNAStralign.tar.gz')]
    outputs = [os.path.join('data', f'experiment{matrix_type}.tar.gz')]
    options = {"memory":"16gb", "walltime":"03:00:00"}
    spec = """python3 scripts/experiment_files.py {matrix_type}
    tar -czf data/experiment{matrix_type}.tar.gz data/experiment{matrix_type}
    rm -r data/experiment{matrix_type}""".format(matrix_type = matrix_type)
    return AnonymousTarget(inputs=inputs, outputs=outputs, options=options, spec=spec)

def postprocess_time(): 
    inputs = []
    outputs = [os.path.join('results', 'postprocess_time.csv'),
               os.path.join('figures', 'postprocess_time.png')]
    options = {"memory": "16gb", "walltime": "36:00:00"}
    spec = """python3 scripts/time_postprocessing.py"""
    return AnonymousTarget(inputs=inputs, outputs=outputs, options=options, spec=spec)


### TRAINING ###
def make_complete_set(): 
    inputs = [os.path.join('data', 'RNAStralign.tar.gz')]
    outputs = [os.path.join('data', 'complete_set.tar.gz'),
               os.path.join('data', 'test_files.tar.gz'),
               os.path.join('data', 'train.pkl'),
               os.path.join('data', 'valid.pkl'),
               os.path.join('data', 'test.pkl'),
               os.path.join('data', 'familymap.pkl'),
               os.path.join('figures', 'length_distribution.png'),
               os.path.join('figures', 'family_distribution.png')]
    options = {"memory":"64gb", "walltime":"24:00:00"}
    spec = """python3 scripts/complete_dataset.py
    tar -czf data/test_files.tar.gz data/test_files
    rm -r data/test_files
    tar -czf data/complete_set.tar.gz data/complete_set
    rm -r data/complete_set"""
    return AnonymousTarget(inputs=inputs, outputs=outputs, options=options, spec=spec)    



### WORKFLOW ###
gwf = Workflow()

#Make data for experiments
for matrix_type in [8, 9, 17]:
    gwf.target_from_template(f'experiment_data_{matrix_type}', make_experiment_data(matrix_type))

#Convert entire data set
gwf.target_from_template('convert_data', make_complete_set())

#Make experiment of post processing time 
gwf.target_from_template('time_postprocess', postprocess_time())