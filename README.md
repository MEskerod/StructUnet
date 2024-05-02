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
- other_methods --> Scripts used to process test set using other methods
- results   
- scripts
    - utils --> Folder containing the functions used in various scripts
    - compare_methods.py --> script that evaluates the predicted structure by different methods
    - complete_dataset.py --> script for converting entire dataset using 8-channel input
    - evaluate_hotknot.py --> script that uses different k with hotknots on a very small subset of the data to evaluate its performance
    - evaluate_postprocessing.py --> script that uses the final model to evaluate all available post-processing methods
    - experiment_files.py --> script for converting sequences in RNAStralign used for experiments to matrices. Can convert inputs to 8, 9 or 17-channel input
    - make_test_under_600.py --> script that writes the index of all the files in the test set with sequence lengths below 600 to pickle file
    - predict_test.py --> script that uses the final model to predict and post-process the files in the test set
    - test.py --> contains pytests for functions
    - time_matrix_conversion.py --> script for timing conversion to different types of input matrices 
    - time_postprocessing.py --> script for timing use of different post-processing methods
    - traning.py --> script used for training the model on the entire data set using the device available
- workflow.py --> GWF workflow used to run some scripts on cluster
- environment1.yml --> File containing *RNAUnet* conda environment
- environment2.yml --> File containing *RNA_Unet* conda environment (used when using GPU)


