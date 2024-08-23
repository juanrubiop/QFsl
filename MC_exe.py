"""
To run this on terminal execute: python3 MC_exe.py $N$ $n_layers$ $is_testing$ $ansatz$ $SEED$

SEED is optional, if not present the code will run all seeds in a hardocded array

where ansatz is an element of ['alpha','beta','gamma'] and

alpha="Sim15Ansatz"
beta="FslBase"
gamma="FslNN"
eta="IQP"

Example: python3 MC_exe.py 2 1 True alpha 450
"""
from icecream import ic
ic('init')

from lambeq import BobcatParser, AtomicType, SpacyTokeniser, Rewriter
import numpy as np

from lambeq import TketModel, QuantumTrainer, SPSAOptimizer,remove_cups

import matplotlib.pyplot as plt

from lambeq import AtomicType,BinaryCrossEntropyLoss, Dataset

from lambeq import NumpyModel

from lambeq import IQPAnsatz,Sim15Ansatz

import datetime

from utils.FslAnsatz import FslBaseAnsatz, FslNN

import pickle

import torch.multiprocessing as mp

from torch.nn.parallel import DistributedDataParallel as DDPP

import warnings
warnings.filterwarnings("ignore")

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import pathlib

import torch

from torch import nn

import sys

args=sys.argv


ic('parser and tokeniser')
parser = BobcatParser(verbose='text')
tokeniser = SpacyTokeniser()

# Loading classical embeddings from the resources folder
def load_data():
    preq_embeddings={}
    with open("resources/embeddings/common_crawl/glove.42B.300d.txt", 'r', encoding="utf-8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], "float32")
            preq_embeddings[word] = vector
    return preq_embeddings



ic('importing embeddings')
preq_embeddings=load_data()

n_layers=int(args[2])
ic('Number of layers:',n_layers)


def read_data(filename):
    labels, sentences = [], []
    with open(filename) as f:
        for line in f:
            t = int(line[0])
            labels.append([t, 1-t])
            sentences.append(line[1:].strip())
    return labels, sentences

def get_unique_words(sentences):
    unique_words = set()
    
    for sentence in sentences:
        words = sentence.split()
        for word in words:
            # Remove punctuation and convert to lower case
            clean_word = ''.join(char for char in word if char.isalnum()).lower()
            if clean_word:
                unique_words.add(clean_word)
    
    return unique_words


def generate_diagrams(train_data,dev_data,test_data,OOV_test_data,redundant_test_data):
    raw_train_tokens = tokeniser.tokenise_sentences(train_data)
    raw_train_tokens = [tokens[:-1] for tokens in raw_train_tokens]

    raw_dev_tokens = tokeniser.tokenise_sentences(dev_data)
    raw_dev_tokens = [tokens[:-1] for tokens in raw_dev_tokens]


    raw_test_tokens = tokeniser.tokenise_sentences(test_data)
    raw_test_tokens =  [tokens[:-1] for tokens in raw_test_tokens]

    raw_OOV_test_tokens = tokeniser.tokenise_sentences(OOV_test_data)
    raw_OOV_test_tokens = [tokens[:-1] for tokens in raw_OOV_test_tokens]

    raw_redundancy_test_tokens = tokeniser.tokenise_sentences(redundant_test_data)
    raw_redundancy_test_tokens = [tokens[:-1] for tokens in raw_redundancy_test_tokens]

    train_diagrams = parser.sentences2diagrams(raw_train_tokens,tokenised=True)
    dev_diagrams = parser.sentences2diagrams(raw_dev_tokens,tokenised=True)
    test_diagrams = parser.sentences2diagrams(raw_test_tokens,tokenised=True)
    OOV_test_diagrams = parser.sentences2diagrams(raw_OOV_test_tokens,tokenised=True)
    redundancy_test_diagrams = parser.sentences2diagrams(raw_redundancy_test_tokens,tokenised=True)
    
    return train_diagrams, dev_diagrams, test_diagrams,OOV_test_diagrams,redundancy_test_diagrams

def get_NN(N_Qubits):
    N_PARAMS=3*N_Qubits-1
    ic(N_Qubits,N_PARAMS,n_layers)
    class PreQ(nn.Module):
        def __init__(self):
            super(PreQ,self).__init__()
            self.flatten = nn.Flatten(start_dim=0)
            self.linear_relu_stack = nn.Sequential(
                nn.Linear(300, 200),
                nn.ReLU(),
                nn.Linear(200, 100),
                nn.ReLU(),
                nn.Linear(100, 50),
                nn.ReLU(),
                nn.Linear(50,N_PARAMS)
            )
            self.double()

        def forward(self, x):
            #x = self.flatten(x)
            logits1=self.linear_relu_stack(x)


            return logits1

        def get_quantum_state(self,parameters):

            first_layer=torch.stack([self.recursiveRx(i,parameters,N_Qubits-1) for i in range(parameters.shape[0])])
            second_layer=torch.stack([self.recursiveRy(i,parameters,2*N_Qubits-1) for i in range(parameters.shape[0])])

            rotation_layers=[first_layer,second_layer]

            Cx_layers=[  kron(Id(i),kron(CRx(parameters[:,2*N_Qubits+i]),Id(N_Qubits-i-2))) for i in range(N_Qubits-1) ]

            all_layers=rotation_layers+Cx_layers
            
            output=self.compose(all_layers)        

            return output

        def recursiveRx(self, i, parameters,counter):        
            if counter==1:
                return kron(Rx(parameters[:,counter])[i],Rx(parameters[:,1])[i])
            else:
                return kron(self.recursiveRx(i,parameters,counter-1),Rx(parameters[:,counter])[i])

        def recursiveRy(self, i, parameters,counter):        
            if counter==N_Qubits+1:
                return kron(Ry(parameters[:,counter-1])[i],Ry(parameters[:,counter])[i])
            else:
                return kron(self.recursiveRy(i,parameters,counter-1),Ry(parameters[:,counter])[i])

        def compose(self,layers):

            if len(layers)==2:
                return bmm(layers[0],layers[1])
            
            else:
                last_element=layers.pop()
                return bmm(self.compose(layers),last_element)

    saved_model=PreQ()

    PATH=f'resources/embeddings/NN/AUW_{N_Qubits}_{N_PARAMS}_{n_layers}/Models/best_model'
    ic(PATH)
    saved_dict=torch.load(PATH)
    
    saved_model.load_state_dict(saved_dict)
    saved_model.eval()
    return saved_model
    

ic('reading data')
train_labels, train_data = read_data('resources/dataset/new_mc_train_data.txt')
dev_labels, dev_data = read_data('resources/dataset/new_mc_dev_data.txt')
test_labels, test_data = read_data('resources/dataset/new_mc_test_data_seen.txt')

OOV_test_labels, OOV_test_data = read_data('resources/dataset/new_mc_test_data_OOV.txt')
redundant_test_labels, redundant_test_data = read_data('resources/dataset/new_mc_test_data_redundancy.txt')

all_sentences=train_data+dev_data+test_data+OOV_test_data+redundant_test_data
unique_words=get_unique_words(all_sentences)

try:
    unique_words.remove('1')
except Exception as e:
    ic(e)


unique_words=list(unique_words)
values=[preq_embeddings.get(key) for key in unique_words]
ic(len(values))
unique_words_values=torch.tensor(values,dtype=torch.double)

NN_embeddings={}
N_Qubits=int(args[1])
if args[4]=="gamma":
    for i in range(2,7):
        model=get_NN(i)
        outputs=model(unique_words_values).tolist()
        temp_dict={word:output for word,output in zip(unique_words,outputs)}
        NN_embeddings[i]=temp_dict


TESTING=(True if args[3]=='True' else False)
ic(TESTING)

if TESTING:
    train_labels, train_data = train_labels[:2], train_data[:2]
    dev_labels, dev_data = dev_labels[:2], dev_data[:2]
    test_labels, test_data = test_labels[:2], test_data[:2]
    OOV_test_labels, OOV_test_data = OOV_test_labels[:2], OOV_test_data[:2]
    redundant_test_labels, redundant_test_data = redundant_test_labels[:2], redundant_test_data[:2]
    EPOCHS = 1


ic('generating diagrams')
train_diagrams, dev_diagrams, test_diagrams,OOV_test_diagrams,redundancy_test_diagrams=generate_diagrams(train_data=train_data,dev_data=dev_data,test_data=test_data,OOV_test_data=OOV_test_data,redundant_test_data=redundant_test_data)

def create_circuits(map,n_layers,ansatz_string,preq_embeddings):
    ansatz = Sim15Ansatz(map,n_layers=n_layers, n_single_qubit_params=3)
    match ansatz_string:
        case "FslBase":
            ansatz = FslBaseAnsatz(preq_embeddings,map, n_layers=n_layers)
        case "Sim15":
            ansatz = Sim15Ansatz(map,n_layers=n_layers, n_single_qubit_params=3)  
        case "FslNN":
            ansatz = FslNN(preq_embeddings=NN_embeddings,ob_map=map,n_layers=n_layers)
        case "IQP":
            ansatz = IQPAnsatz(map,n_layers=n_layers, n_single_qubit_params=3)  

    train_circuits = [ansatz(diagram) for diagram in train_diagrams]
    print("Train circuits done")
    dev_circuits =  [ansatz(diagram) for diagram in dev_diagrams]
    print("Dev circuits done")
    test_circuits = [ansatz(diagram) for diagram in test_diagrams]
    print("Test circuits done")
    OOV_test_circuits = [ansatz(diagram) for diagram in OOV_test_diagrams]
    print("OOV circuits done")
    redundancy_test_circuits = [ansatz(diagram) for diagram in redundancy_test_diagrams]
    print("Redundant circuits done")

    return train_circuits, dev_circuits, test_circuits, OOV_test_circuits, redundancy_test_circuits

def set_model(model_string,checkpoint,logdir=''):
    ic('setting model')
    match model_string:
        case "Numpy":
            if checkpoint:
                    model = NumpyModel.from_checkpoint(logdir+'/model.lt')
            else:
                    model = NumpyModel.from_diagrams(all_circuits, use_jit=False)
        case "Tket":
            backend = AerBackend()
            backend_config = {
                'backend': backend,
                'compilation': backend.default_compilation_pass(2),
                'shots': 8192
            }
            model = TketModel.from_diagrams(all_circuits, backend_config=backend_config)
    
    return model

def save_everything(logdir,loss_function,acc_function,a,c,A,model,trainer,test_acc):
    print("Saving everything")
    acc = lambda y_hat, y: np.sum(np.round(y_hat) == y) / len(y) / 2  # half due to double-counting

    fig, ((ax_tl, ax_tr), (ax_bl, ax_br)) = plt.subplots(2, 2, sharex=True, sharey='row', figsize=(10, 6))
    ax_tl.set_title('Training set')
    ax_tr.set_title('Development set')
    ax_bl.set_xlabel('Iterations')
    ax_br.set_xlabel('Iterations')
    ax_bl.set_ylabel('Accuracy')
    ax_tl.set_ylabel('Loss')

    colours = iter(plt.rcParams['axes.prop_cycle'].by_key()['color'])
    range_ = np.arange(1, trainer.epochs + 1)
    ax_tl.plot(range_, trainer.train_epoch_costs, color=next(colours))
    ax_bl.plot(range_, trainer.train_eval_results['acc'], color=next(colours))
    ax_tr.plot(range_, trainer.val_costs, color=next(colours))
    ax_br.plot(range_, trainer.val_eval_results['acc'], color=next(colours))
    plt.savefig(logdir+'/plot.png')


    best_model=NumpyModel.from_checkpoint(logdir+'/best_model.lt')
    best_model_test_acc = acc(best_model(test_circuits), test_labels)
    model=NumpyModel.from_checkpoint(logdir+'/model.lt')
    test_acc = acc(model(test_circuits), test_labels)

    bm_OOV_test_acc= acc(best_model(OOV_test_circuits), OOV_test_labels)
    OOV_test_acc= acc(model(OOV_test_circuits), OOV_test_labels)

    bm_redundant_test_acc= acc(best_model(redundancy_test_circuits), redundant_test_labels)
    redundant_test_acc= acc(model(redundancy_test_circuits), redundant_test_labels)

    file_path = f"{logdir}/info_file.txt"
    with open(file_path, 'w') as file:
        # Write the input string to the file
        input_string=f"""Task: Meaning classification
    Classical Embeddings: GloVe 50-d
    Parsing: True
    Rewritign: Remove Cups
    Ansatz: {ansatz_string}
    Layers: {n_layers}
    Map: [N:{map[N]}, S:{map[S]}]
    Model: Numpy
    Backend: None
    Trainer: Quantum Trainer
    Loss function: {loss_function}
    Accuracy function: {acc_function}
    Optimizer: SPSA optimizer
    Epochs: {EPOCHS}
    Batch size: {BATCH_SIZE}
    Seed: {SEED}
    Hyperparams: [a:{a},c:{c},A:{A}]
    Test accuracy: {test_acc}
    Test accuracy best model: {best_model_test_acc}
    OOV test accuracy: {OOV_test_acc}
    OOV test accuracy best model: {bm_OOV_test_acc}
    Redundant test accuarcy: {redundant_test_acc}
    Redundant test accuracy best model: {bm_redundant_test_acc}"""
        file.write(input_string)


def main(EPOCHS, SEED, BATCH_SIZE,MODEL):
    # Using the builtin binary cross-entropy error from lambeq
    acc = lambda y_hat, y: np.sum(np.round(y_hat) == y) / len(y) / 2  # half due to double-counting
    bce = BinaryCrossEntropyLoss(use_jax=True)
    loss_function="BindaryCrosEntropyLoss"
    acc_function="lambda y_hat, y: np.sum(np.round(y_hat) == y) / len(y) / 2"

    a=0.05
    c=0.06
    A="0.1*Epochs"
    path = pathlib.Path(logdir)
    path.mkdir(parents=True, exist_ok=True)
    print('Initialize trainer')

    trainer = QuantumTrainer(
        model=MODEL,
        loss_function=bce,
        epochs=EPOCHS,
        optimizer=SPSAOptimizer,
        optim_hyperparams={'a': a, 'c': 0.06, 'A':0.01*EPOCHS},
        evaluate_functions={'acc': acc},
        evaluate_on_train=True,
        verbose = 'text',
        seed=SEED,
        from_checkpoint=checkpoint,
        log_dir=logdir
    )

    train_dataset = Dataset(
                train_circuits,
                train_labels,
                batch_size=BATCH_SIZE)

    val_dataset = Dataset(dev_circuits, dev_labels, shuffle=False)

    now = datetime.datetime.now()
    t = now.strftime("%Y-%m-%d_%H_%M_%S")
    print(t)
    print('Starting fit')
    trainer.fit(train_dataset, val_dataset, log_interval=10)
    test_acc = 'acc(model(test_circuits), test_labels)'

    save_everything(logdir=logdir,loss_function=loss_function,acc_function=acc_function,a=a,c=c,A=A,model=MODEL,trainer=trainer,test_acc=test_acc)

ic('Finished importing embeddings')
# Define atomic types
N = AtomicType.NOUN
S = AtomicType.SENTENCE
map={N:int(args[1]),S:1}

alpha="Sim15Ansatz"
beta="FslBase"
gamma="FslNN"
eta="IQP"

if args[4]=="eta":
    ansatz_string=eta
else:
    ansatz_string={args[4]=='alpha': alpha, args[4]=='beta': beta}.get(True, gamma)

ic(ansatz_string)


ic("Turning sentences to circuits")
ic(ansatz_string)
ic(map)
train_circuits, dev_circuits, test_circuits,OOV_test_circuits, redundancy_test_circuits=create_circuits(map=map,n_layers=n_layers,ansatz_string=ansatz_string,preq_embeddings=preq_embeddings)
ic("Circuit Processing finished")
all_circuits = train_circuits+dev_circuits+test_circuits+OOV_test_circuits+redundancy_test_circuits

checkpoint=False

ic("Setting model")
model=set_model(model_string="Numpy",checkpoint=checkpoint)

if len(args)==6:
    seed_arr=[int(args[5])]
else:
    seed_arr = [0, 10, 50, 77, 100, 111, 150, 169, 200, 234, 250, 300, 350, 400, 450]


if TESTING:
    seed_arr = [100, 111]
B_sizes = [700]
epochs_arr = [1500]

for SEED in seed_arr:
    for BATCH_SIZE in B_sizes:
        for EPOCHS in epochs_arr:
            ic(EPOCHS, SEED, BATCH_SIZE)
            ic(main(EPOCHS, SEED, BATCH_SIZE,MODEL=model))
            model=set_model(model_string="Numpy",checkpoint=checkpoint)

now = datetime.datetime.now()
t = now.strftime("%Y-%m-%d_%H_%M_%S")
ic(t)

