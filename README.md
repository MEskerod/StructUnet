# RNA UNET
**WORK IN PROGRESS!**

Made as part of a master thesis in Bioinformatics 

## Folder structure
- data --> Files are too big for gitHub
- experiments   
    - utils.py --> functions used in experiments on google CoLab  
    - RNAUnet_experiment... --> Notebooks containing the three experiments
    - experiment_familymap.pkl, -train.pkl, -valid.pkl --> pickle files with map of family to one-hot encoding and lists of files in each set   
- figures --> Contains all figures used in report. Both showing results and digrams to explain methods   
- results   
- scripts
    - utils --> Folder containing the functions used in various scripts
    - complete_dataset.py --> script for converting entire dataset using 8-channel input
    - experiment_files.py --> script for converting sequences in RNAStralign used for experiments to matrices
    - test.py --> contains pytests for functions
    - time_matrix_conversion.py --> script for timing conversion to different types of input matrices 
    - time_postprocessing.py --> script for timing use of different post-processing methods
- workflow.py --> GWF workflow used to run some scripts on cluster


