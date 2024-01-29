
from gwf import Workflow, AnonymousTarget

### EXPERIMENTS ###
def make_experiment_data(): 
    inputs = ['data/RNAStralign.tar.gz']
    outputs = ['data/experiment.pkl']
    options = {"memory":"32gb", "walltime":"03:00:00"}
    spec = """python3 scripts/experiment_files.py"""
    return AnonymousTarget(inputs=inputs, outputs=outputs, options=options, spec=spec)




### TRAINING ###



### WORKFLOW ###
gwf = Workflow()

gwf.target_from_template('experiment_data', make_experiment_data())