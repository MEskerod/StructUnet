
import os, pickle
from gwf import Workflow, AnonymousTarget

### EXPERIMENTS ###
def make_experiment_data(matrix_type): 
    inputs = [os.path.join('data', 'RNAStralign.tar.gz')]
    outputs = [os.path.join('data', f'experiment{matrix_type}.tar.gz')]
    options = {"memory":"16gb", "walltime":"03:00:00", "account":"RNA_Unet"}
    spec = """python3 scripts/experiment_files.py {matrix_type}
    tar -czf data/experiment{matrix_type}.tar.gz data/experiment{matrix_type}
    rm -r data/experiment{matrix_type}""".format(matrix_type = matrix_type)
    return AnonymousTarget(inputs=inputs, outputs=outputs, options=options, spec=spec)

def postprocess_time(): 
    inputs = []
    outputs = [os.path.join('results', 'postprocess_time.csv'),
               os.path.join('figures', 'postprocess_time.png')]
    options = {"memory": "16gb", "walltime": "36:00:00", "account":"RNA_Unet", "cores": 4}
    spec = """python3 scripts/time_postprocessing.py"""
    return AnonymousTarget(inputs=inputs, outputs=outputs, options=options, spec=spec)

def convert_time(): 
    inputs = []
    outputs = [os.path.join('results', 'convert_time.csv'),
               os.path.join('figures', 'convert_time.png')]
    options = {"memory": "16gb", "walltime": "24:00:00", "account":"RNA_Unet"}
    spec = """python3 scripts/time_matrix_conversion.py"""
    return AnonymousTarget(inputs=inputs, outputs=outputs, options=options, spec=spec)


### TRAINING ###
def make_complete_set(): 
    inputs = [os.path.join('data', 'RNAStralign.tar.gz')]
    outputs = [os.path.join('data', 'train.pkl'),
               os.path.join('data', 'valid.pkl'),
               os.path.join('data', 'test.pkl'),
               os.path.join('figures', 'length_distribution.png'),
               os.path.join('figures', 'family_distribution.png')]
    options = {"memory":"16gb", "walltime":"6:00:00", "account":"RNA_Unet", "cores":4}
    spec = """python3 scripts/complete_dataset.py
    tar -czf data/test_files.tar.gz data/test_files"""
    return AnonymousTarget(inputs=inputs, outputs=outputs, options=options, spec=spec)    

def train_model(): 
    inputs = []
    outputs = ['RNA_Unet.pth']
    options = {"memory":"16gb", "walltime":"5:00:00", "account":"RNA_Unet", "gres":"gpu:1", "queue":"gpu"} #NOTE - Think about memory and walltime and test GPU
    spec = """python3 scripts/training.py"""
    return AnonymousTarget(inputs=inputs, outputs=outputs, options=options, spec=spec)

### EVALUATION ###

def predict_hotknots(file): 
    inputs = [file]
    outputs = [os.path.join('steps', 'hotknots', os.path.basename(file))]
    options = {"memory":"512gb", "walltime":"8:00:00", "account":"RNA_Unet"} 
    spec = """python3 ../HotKnots/hotknots.py "{file}" "steps/hotknots/{output}" """.format(file = file, output = os.path.basename(file))
    return AnonymousTarget(inputs=inputs, outputs=outputs, options=options, spec=spec)

def evaluate_nn(): 
    inputs = [os.path.join('data', 'test_files.tar.gz')] #FIXME - Add path to model
    outputs = [os.path.join('results', 'evaluation_nn.csv'),
               os.path.join('figures', 'evaluation_nn.png')]
    options = {"memory":"16gb", "walltime":"24:00:00", "account":"RNA_Unet"} #NOTE - Think about memory and walltime
    spec = """
    python3 scripts/evaluate_nn.py"""
    return AnonymousTarget(inputs=inputs, outputs=outputs, options=options, spec=spec) #TODO - Add some commands!


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

#gwf.target_from_template('train_model', train_model())


#Predicting with other methods for comparison
test_files = pickle.load(open('data/test_small.pkl', 'rb'))
for i, file in enumerate(test_files): 
    gwf.target_from_template(f'predict_hotknots_file_no_{i}', predict_hotknots(file))



def remove(file): 
    inputs = [file]
    outputs = []
    options = {} 
    spec = """rm {file}""".format(file = file)
    return AnonymousTarget(inputs=inputs, outputs=outputs, options=options, spec=spec) #TODO - Add some commands!

test_files = pickle.load(open('data/test.pkl', 'rb'))
excluded = ([1282, 1287, 1288, 1289, 1290, 1291, 1292, 1294, 1295, 1297, 1298, 1299, 1300, 1302, 1303, 1304, 1305, 1306, 1307, 1308, 1309] + list(range(2036, 2186)) + list(range(2020, 2035)) +
list(range(1964, 2019)) + list(range(1940, 1963)) + list(range(1917, 1939)) + list(range(1914, 1915)) + list(range(1908,  1913)) + list(range(1901, 1907)) + list(range(1870, 1900)) + 
[1868, 1865, 1866] + list(range(1855,  1864)) + list(range(1831,  1854)) + list(range(1742, 1830)) + list(range(1721, 1741)) + list(range(1718, 1720)) + list(range(1710,  1717)) +
list(range(1683,  1709)) + list(range(1601,  1682)) + list(range(1555,  1600)) + list(range(1532,  1554)) + list(range(1502,  1530)) + list(range(1417,  1501)) + list(range(1385,  1416)) +
list(range(1327,  1382)) + list(range(1317, 1326)) + list(range(1311, 1316)))



for i in excluded: 
    file = os.path.join('steps', 'hotknots', os.path.basename(test_files[i]))
    #gwf.target_from_template(f'remove_{i}', remove(file))



