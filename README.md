# FslQNLP
This repo contains the code and results used for the thesis "A Few Shot Learning Scheme for Quantum Natural Language Processing", by Juan Pablo Rubio Perez, for the MSc Quantum Tehcnologies at UCL in 2024. This repo also contains the tex source files.
## Overview
This contains the resources and the code to run the meaning comparison task in NLP on a quantum circuit using FSL. The MC_exe.py runs with the presets for various initial seeds and averages. A couple of of things need to be prepared before being able to run any of the file.

## Preparations
The requirement.txt contains all the modules and their versions. Python 3.11.9 is used and encouraged, since the same version of the packages may not exist for other versions of python. The GloVe embeddings or any other embeddings need to be downloaded and saved in the appropriate resource folder. resources/embeddings.

## CLI

For running the code from the CLI, the following structure must be used:
For MC_exe.py call:

```console
~$python3 MC_exe.py $N_qubits $N_layers $TESTING $ANSATZ
~$python3 MC_exe.py 3 1 False alpha
```   

For nn_all_used_words.py call:
```console
~$ python3 nn_all_used_words.py $N_qubits $N_layers
~$ python3 nn_all_used_words.py 3 1
```
The ansätze keys are: 

    alpha: Sim15Ansatz

    beta: FslBase 

    gamma: FslNN

    eta: IQP

## Keys for file schema
For NN models:

    (name of training set)_(number of qubits)_(number of parameters)

    AUW - All used words (i.e. training space is all but only the words used, they can be used during training or testing)

## Some notes
Running a single model could take up to a week without any gpus, and depending on the ansatz and circuit size may take up to 60 gb of RAM.

## File structure

The outputs for the runs that tasted the behaviour (i.e. for low dataset sizes) are on the folder behavioural_runs. The outputs for the bigger datasets, as well as the outputs for the training of the nerual network are on runs/Proper and runs/NN_outputs respectviely. The definition of the new ansätze are on the utils folder. The folder resources/embeddings/common_crawl is empty. It should be populated with the datasets downloaded from GloVe.