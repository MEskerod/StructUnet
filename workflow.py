
import os
from gwf import Workflow, AnonymousTarget

### EXPERIMENTS ###
def make_experiment_data(matrix_type): 
    inputs = [os.path.join('data', 'RNAStralign.tar.gz')]
    outputs = [os.path.join('data', f'experiment{matrix_type}.pkl')]
    options = {"memory":"64gb", "walltime":"03:00:00"}
    spec = """python3 scripts/experiment_files.py {matrix_type}
    gzip data/experiment{matrix_type}.pkl""".format(matrix_type = matrix_type)
    return AnonymousTarget(inputs=inputs, outputs=outputs, options=options, spec=spec)


### TRAINING ###



### WORKFLOW ###
gwf = Workflow()

for matrix_type in [8, 17]:
    gwf.target_from_template(f'experiment_data_{matrix_type}', make_experiment_data(matrix_type))