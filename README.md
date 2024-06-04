# StructUnet
**WORK IN PROGRESS!**

Made as part of a master thesis in Bioinformatics 

Files that are too big can be located at: https://drive.google.com/drive/folders/15VAdY8AYT4Z6OosgDE6-HZ-c1UZ5YQeW?usp=sharing
## Folder structure
- data --> Most files are too big for GitHub (or scripts exists for converting from other files)
- experiments   
    - utils.py --> functions used in experiments on google CoLab  
    - RNAUnet_experiment... --> Notebooks containing the three experiments
    - experiment_familymap.pkl, -train.pkl, -valid.pkl --> pickle files with map of family to one-hot encoding and lists of files in each set   
- figures --> Contains all figures used in report. Both showing results and digrams to explain methods   
- other_methods --> Scripts used to process test set using other methods
- results   
- scripts
    - utils --> Folder containing the functions used in various scripts
    - compare_predictions_over600.py --> script that evaluates the predictions made of the method able to predict on sequences longer than 600 nucleotides
    - compare_precitions_under600.py --> script that evaluates the predictions made of all methods on sequences below 600
    - compare_methods.py --> script that evaluates the predicted structure by different methods
    - complete_dataset.py --> script for converting entire dataset using 8-channel input
    - count_loops.py --> script used to count all hairpin loops in all sequences in RNAStralign
    - count_noncanoncial_pairs.py --> script used to count the different base pair types in RNAStralign
    - evaluate_hotknot.py --> script that uses different k with hotknots on a very small subset of the data to evaluate its performance
    - evaluate_postprocessing_under600.py --> script that uses the final model to evaluate all available post-processing methods on all sequences in the validation set below 600 nucleotides
    - experiment_files.py --> script for converting sequences in RNAStralign used for experiments to matrices. Can convert inputs to 8, 9 or 17-channel input
    - make_test_under_600.py --> script that writes the index of all the files in the test set with sequence lengths below 600 to pickle file
    - predict_test.py --> script that uses the final model to predict and post-process the files in the test set
    - test.py --> contains pytests for functions
    - time_matrix_conversion.py --> script for timing conversion to different types of input matrices 
    - time_postprocessing.py --> script for timing use of different post-processing methods
    - traning.py --> script used for training the model on the entire data set using the device available
- workflow.py --> GWF workflow used to run some scripts on cluster
- environment1.yml --> file containing *RNAUnet* conda environment
- environment2.yml --> file containing *RNA_Unet* conda environment (used when using GPU)
- blossom.py --> script containing modified version of NetworkX maximum weight matching 
- predict.py --> script that can be used to predict RNA secondary structure using StructUnet. Input it either sequence inputted directly or in fasta file. Output can be .ct or .bpseq file


