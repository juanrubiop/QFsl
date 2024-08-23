# -*- coding: utf-8 -*-
"""nn_all_used_words.ipynb
This trains a Neural Network to get the parameters for a circuit given only all the words that will be used, in both training and testing.
to call this file call on cli 

python3 nn_all_used_words.py N_Qubits N_Layers

where N_Qubits is the number of qubits you want to input and N_Layers the number of layers
"""

import os
import pathlib
import sys
import torch
from torch import nn,matmul,kron,bmm
from torch.utils.data import DataLoader
import numpy as np
from torchvision import datasets, transforms
import torchvision.transforms as transforms
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

from icecream import ic


from numpy import dot
from numpy.linalg import norm

###D

import torch.multiprocessing as mp

from torch.nn.parallel import DistributedDataParallel as DDPP



####

#from google.colab import drive
#drive.mount('/content/drive')

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
torch.cuda.empty_cache()

try:
    ic(torch.cuda.get_device_name())
except Exception as e:
    ic(e)

ic(f"Using {device} device")

letters=['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']

from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self,labels,embeddings):
        self.labels=labels
        self.embeddings=embeddings

    def __len__(self):
        return len(self.labels)

    def __getitem__(self,idx):
        return self.embeddings[idx],self.labels[idx]


ic('importing embedding vectors')
preq_embeddings={}
resource_path="/home/jrubiope/FslQnlp/resources/embeddings/common_crawl/glove.42B.300d.txt"
#resource_path="resources/embeddings/common_crawl/glove.42B.300d.txt"


with open(resource_path, 'r', encoding="utf-8") as f:
    for line in f:
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:], "double")
        preq_embeddings[word] = vector
ic('finished importing vectors')

keys=['cooks', 'lady', 'fixes', 'bakes', 'program', 'breakfast', 'skillful', 'troubleshoots', 'supper', 'delightful', 'grills', 'delicious', 'guy', 'repairs', 'code', 'gentleman', 'dinner', 'someone', 'feast', 'sauce', 'boy', 'interesting', 'helpful', 'individual', 'man', 'software', 'runs', 'prepares', 'completes', 'useful', 'tool', 'adept', 'tasty', 'practical', 'flavorful', 'roasts', 'dexterous', 'woman', 'application', 'meal', 'noodles', 'soup', 'algorithm', 'executes', 'makes', 'person', 'snack', 'lunch', 'teenager', 'debugs', 'chicken', 'masterful']

TESTING=True

words=[]
embeddings=[]
for key, value in preq_embeddings.items():
    words.append(key)
    embeddings.append(value)

training_words=keys
training_embeddings=[preq_embeddings.get(key) for key in training_words]

dev_words=words[30:60]
dev_embeddings=embeddings[30:60]

Bigger=False


def create_data(embeddings):
    training_data=[]
    training_labels=[]
    for i in range(len(embeddings)):
        for j in range(len(embeddings)):
            cos=dot(embeddings[i][0], embeddings[j][0])/(norm(embeddings[i][0])*norm(embeddings[j][0]))
            
            new_embedding=np.append(embeddings[i],embeddings[j])

            training_labels.append(cos)
            training_data.append(new_embedding)

    if Bigger:
        training_data=training_data[0:round(len(training_data)/4)]
        ic(round(len(training_data)/4))
        training_labels=training_labels[0:round(len(training_data)/4)]

    #training_data=torch.tensor(np.array(training_data),requires_grad=True,device=device)
    training_data=torch.tensor(np.array(training_data),requires_grad=True,device='cpu')

    #training_labels=torch.tensor(np.array(training_labels),requires_grad=True)
    training_labels=torch.tensor(np.array(training_labels),requires_grad=True,device='cpu')
    
    training_labels_square=torch.square(training_labels)


    return training_data, training_labels_square

def Id(N,device='cpu'):
    gate=torch.eye(2**N,requires_grad=True,dtype=torch.complex128,device=device)    
    gate.retain_grad()
    return gate

def zero_bra(N,device='cpu'):
    gate=torch.tensor([1.+0j if i==0 else 0 for i in range(2**N)],requires_grad=True,dtype=torch.complex128,device=device)
    gate.retain_grad()
    return gate

def zero_ket(N,device='cpu'):
    gate=torch.tensor([[1.+0j] if i==0 else [0] for i in range(2**N)],requires_grad=True,dtype=torch.complex128,device=device)
    gate.retain_grad()
    return gate

def zero_1d_ket(device='cpu'):
    gate=torch.tensor([[1.+0j],[0]],requires_grad=True,dtype=torch.complex128,device=device)
    gate.retain_grad()
    return gate

def zero_1d_bra(device='cpu'):
    gate=torch.tensor([[1.+0j,0]],requires_grad=True,dtype=torch.complex128,device=device)
    gate.retain_grad()
    return gate

def one_1d_ket(device='cpu'):
    gate=torch.tensor([[0],[1.+0j]],requires_grad=True,dtype=torch.complex128,device=device)
    gate.retain_grad()
    return gate

def one_1d_bra(device='cpu'):
    gate=torch.tensor([[0,1.+0j]],requires_grad=True,dtype=torch.complex128,device=device)
    gate.retain_grad()
    return gate

def Ry(theta,device='cpu'):
    y_gate=torch.tensor([[0,-1j],[1j,0]],dtype=torch.complex128,requires_grad=True,device=device)
    gate = torch.linalg.matrix_exp(-0.5j*theta[:,:,None]*y_gate)
    return gate

def Rx(theta,device='cpu'):
    x_gate=torch.tensor([[0,1],[1,0]],dtype=torch.complex128,requires_grad=True,device=device)
    gate = torch.linalg.matrix_exp(-0.5j*theta[:,:,None]*x_gate)
    return gate

def Rz(theta,device='cpu'):
    z_gate=torch.tensor([[1,0],[0,-1]],dtype=torch.complex128,requires_grad=True,device=device)
    gate = torch.linalg.matrix_exp(-0.5j*theta[:,:,None]*z_gate)
    return gate

CRx=lambda x,device='cpu': kron(Id(1,device),matmul(zero_1d_ket(device),zero_1d_bra(device)))+kron(Rx(x,device),matmul(one_1d_ket(device),one_1d_bra(device)))

training_data,training_labels=create_data(training_embeddings)
dev_data,dev_labels=create_data(dev_embeddings)

ic(len(training_embeddings)**2)
report_times=500
Batches=2000#4
B_SIZE=1#round(training_data.shape[0]/Batches)

report_times=4
Batches=1000
B_SIZE=round(training_data.shape[0]/Batches)

B=round(Batches/report_times)

training_object=CustomDataset(training_labels,training_data)
dev_object=CustomDataset(dev_labels,dev_data)

training_loader=DataLoader(training_object,batch_size=B_SIZE)
validation_loader=DataLoader(dev_object)

loss_fn = torch.nn.MSELoss()

N_Qubits=int(sys.argv[1])
N_PARAMS=3*N_Qubits-1
N_Layers=int(sys.argv[2])

einsum_rule='zab'
for i in range(1,N_Qubits-1):
    einsum_rule+= (',z'+letters[i]+letters[i+1])
einsum_rule+=('->z'+einsum_rule[1]+einsum_rule[-1])

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
            nn.Linear(50,N_PARAMS*N_Layers)
        ).to(device=torch.device('cuda:0'))
        self.double()

    def forward(self, x):
        first_device=torch.device('cuda:0')

        logits1=self.linear_relu_stack(x[:,0:300])
        logits2=self.linear_relu_stack(x[:,300:600])

        logits1_reshaped=torch.reshape(logits1,(logits1.shape[0],logits1.shape[1],1))
        logits2_reshaped=torch.reshape(logits2,(logits2.shape[0],logits2.shape[1],1))
        #ic(logits1_reshaped.shape)

        bra=torch.stack([zero_bra(N_Qubits,torch.device('cuda:3'))[None] for i in range(logits1_reshaped.shape[0])])
        ket=torch.stack([zero_ket(N_Qubits,torch.device('cuda:3')) for i in range(logits1_reshaped.shape[0])])

        if N_Layers == 2:
            circuit1=bmm(bmm(bra,self.get_quantum_state(parameters=logits2_reshaped,circN=1).mH),self.get_quantum_state(parameters=logits2_reshaped[:,N_PARAMS:,:],circN=1).mH)
            circuit2=bmm(self.get_quantum_state(parameters=logits1_reshaped[:,N_PARAMS:,:],circN=2),bmm(self.get_quantum_state(parameters=logits1_reshaped,circN=2),ket))
            inner_product=self.flatten(bmm(circuit1,circuit2))
            fidelity=torch.square(torch.abs(inner_product)).to(first_device)
            return fidelity

        circuit1=bmm(bra,self.get_quantum_state(parameters=logits2_reshaped,circN=1).mH)
        circuit2=bmm(self.get_quantum_state(parameters=logits1_reshaped,circN=2),ket)

        inner_product=self.flatten(bmm(circuit1,circuit2))
        fidelity=torch.square(torch.abs(inner_product)).to(first_device)


        return fidelity

    def get_quantum_state(self,parameters,circN):

        first_device=torch.device('cuda:0')
        second_device=torch.device('cuda:1')
        third_device=torch.device('cuda:2')
        fourth_device=torch.device('cuda:3')


        #On first device
        first_layer=torch.stack([self.recursiveRx(i,parameters,N_Qubits-1,device=first_device) for i in range(parameters.shape[0])])
        
        try:
            torch.cuda.empty_cache()
        except Exception as e:
            ic(e)

        #On first device
        second_layer=torch.stack([self.recursiveRy(i,parameters,2*N_Qubits-1,device=first_device) for i in range(parameters.shape[0])])

        #On first device
        rotation_layers=bmm(first_layer,second_layer)

        #On second/third device
        if circN==1:
            parameters_out=parameters.to(second_device)
            Cx_layers=[  kron(Id(i,device=second_device),kron(CRx(parameters_out[:,2*N_Qubits+i],device=second_device),Id(N_Qubits-i-2,device=second_device))) for i in range(N_Qubits-1) ]
        else:
            parameters_out=parameters.to(third_device)
            Cx_layers=[  kron(Id(i,device=third_device),kron(CRx(parameters_out[:,2*N_Qubits+i],device=third_device),Id(N_Qubits-i-2,device=third_device))) for i in range(N_Qubits-1) ]



        #First Device
        #crxlayers=self.compose(layers=Cx_layers,device=third_device)

        crxlayers=torch.einsum(einsum_rule,*Cx_layers).to(first_device)

        #On fourth device
        output=bmm(rotation_layers,crxlayers).to(fourth_device)       

        return output

    def recursiveRx(self, i, parameters_in,counter,device):
        parameters=parameters_in.to(device)        
        if counter==1:
            return kron(Rx(parameters[:,counter],device=device)[i],Rx(parameters[:,1],device=device)[i])
        else:
            return kron(self.recursiveRx(i,parameters,counter-1,device=device),Rx(parameters[:,counter],device=device)[i])

    def recursiveRy(self, i, parameters_in,counter,device):  
        parameters=parameters_in.to(device)        
        if counter==N_Qubits+1:
            return kron(Ry(parameters[:,counter-1],device=device)[i],Ry(parameters[:,counter],device=device)[i])
        else:
            return kron(self.recursiveRy(i,parameters,counter-1,device=device),Ry(parameters[:,counter],device=device)[i])

    def compose(self,layers,device='cpu'):
        if len(layers)==2:
            return bmm(layers[0].to(device),layers[1].to(device))
        
        else:
            last_element=layers.pop()
            return bmm(self.compose(layers,device),last_element.to(device))
        

model = PreQ()#.to(device)

#model = torch.nn.DataParallel(model2,device_ids=[0,1,2,3]) 

optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

def train_one_epoch(epoch_index, tb_writer):
    running_loss = 0.
    last_loss = 0.

    for i, data in enumerate(training_loader):
        # Every data instance is an input + label pair
        try:
            torch.cuda.empty_cache()
        except Exception as e:
            ic(e)

        inputs,labels=data
        inputs,labels=inputs.to(torch.device('cuda:0')),labels.to(torch.device('cuda:0'))

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch

        output = model(inputs)


        # Compute the loss and its gradients
        loss = loss_fn(output, labels)
        loss.backward(retain_graph = True)
        
        # ic(loss.item())

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        mm=i%B_SIZE
        # if i % B_SIZE == B_SIZE-1:
        if i%B == B-1:
            last_loss = running_loss / B_SIZE # loss per batch
            ic('  batch {} loss: {}'.format(i + 1, last_loss*100))
            tb_x = epoch_index * len(training_loader) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.

    return last_loss

# Initializing in a separate cell so we can easily add more epochs to the same run
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
runs='/home/jrubiope/FslQnlp/runs/NN_outputs/AUW_{}_{}_{}/Tensor_Board_Events/Prueba{}'.format(N_Qubits,N_PARAMS,N_Layers,timestamp)
#runs='runs/NN_outputs/AUW_{}_{}/Tensor_Board_Events/Prueba{}'.format(N_Qubits,N_PARAMS,timestamp)
path = pathlib.Path(runs)
path.mkdir(parents=True, exist_ok=True)
ic(runs)
model_path='/home/jrubiope/FslQnlp/runs/NN_outputs/AUW_{}_{}_{}/Models'.format(N_Qubits,N_PARAMS,N_Layers)
#model_path='runs/NN_outputs/AUW_{}_{}/Models'.format(N_Qubits,N_PARAMS)
path = pathlib.Path(model_path)
path.mkdir(parents=True, exist_ok=True)

resources_path_model=f'/home/jrubiope/FslQnlp/resources/embeddings/NN/AUW_{N_Qubits}_{N_PARAMS}_{N_Layers}/Models'
path = pathlib.Path(resources_path_model)
path.mkdir(parents=True, exist_ok=True)


writer = SummaryWriter(runs)
epoch_number = 0
EPOCHS = 10
best_vloss = 1_000_000.

for epoch in range(EPOCHS):
    ep='EPOCH {}:'.format(epoch_number + 1)
    ic(ep)

    # Make sure gradient tracking is on, and do a pass over the data
    model.train(True)
    avg_loss = train_one_epoch(epoch_number, writer)


    running_vloss = 0.0
    # Set the model to evaluation mode, disabling dropout and using population
    # statistics for batch normalization.
    model.eval()

    # Disable gradient computation and reduce memory consumption.
    with torch.no_grad():
        for i, vdata in enumerate(validation_loader):
            vinputs, vlabels = vdata
            vinputs,vlabels=vinputs.to(torch.device('cuda:0')),vlabels.to(torch.device('cuda:0'))

            voutputs = model(vinputs)

            vloss = loss_fn(voutputs, vlabels)
            running_vloss += vloss

    avg_vloss = running_vloss / (i + 1)
    calue='LOSS-- Train: {} Valid: {}'.format(avg_loss*100, avg_vloss*100)
    #ic(value)

    # Log the running loss averaged per batch
    # for both training and validation
    writer.add_scalars('Training vs. Validation Loss',
                    { 'Training' : avg_loss, 'Validation' : avg_vloss },
                    epoch_number + 1)
    writer.flush()

    # Track best performance, and save the model's state
    if avg_vloss < best_vloss:
        best_vloss = avg_vloss
        best_model_path = model_path+'/best_model'
        best_model_resources=resources_path_model+'/best_model'
        #model_path = 'runs/NN_outputs/AUW_{}_{}/Models/best_model'.format(N_Qubits,N_PARAMS)
        torch.save(model.state_dict(), best_model_path)
        torch.save(model.state_dict(), best_model_resources)
        


    epoch_number += 1


final_model_path = model_path+'/final_model'
ic(final_model_path)
torch.save(model.state_dict(), final_model_path)