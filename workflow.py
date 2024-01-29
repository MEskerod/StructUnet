
import os
from gwf import Workflow, AnonymousTarget

### EXPERIMENTS ###
def make_experiment_data(): 
    inputs = [os.path.join('data', 'RNAStralign.tar.gz')]
    outputs = [os.path.join('data', 'experiment.pkl')]
    options = {"memory":"64gb", "walltime":"03:00:00"}
    spec = """python3 scripts/experiment_files.py"""
    return AnonymousTarget(inputs=inputs, outputs=outputs, options=options, spec=spec)


### TRAINING ###



### WORKFLOW ###
gwf = Workflow()

gwf.target_from_template('experiment_data', make_experiment_data())