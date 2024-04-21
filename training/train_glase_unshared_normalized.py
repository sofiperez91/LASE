import sys
sys.path.append("../")

import time
import math
import torch
import numpy as np
import argparse
import json
import pickle
from tqdm import tqdm
from collections import Counter

from torch_geometric.utils import to_dense_adj
from torch_geometric.loader import DataLoader

from models.GLASE_unshared_normalized import gLASE 
from models.early_stopper import EarlyStopper
from training.get_init import get_x_init



parser = argparse.ArgumentParser(description='Dataset')
parser.add_argument('--dataset', type=str, default='sbm5_unbalanced_negative')
parser.add_argument('--gd_steps', type=int, default=5)
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--init', type=str, default='glase_init')

args = parser.parse_args()
dataset = args.dataset
gd_steps = args.gd_steps
epochs = args.epochs
init = args.init


# Load the config file
with open('../data/data_config.json', 'r') as file:
    config = json.load(file)

device = 'cuda'
lr = 1e-4
d = config[dataset]['d']
n = config[dataset]['n']
num_nodes = np.sum(n)

if config[dataset]['mode'] == "simple":
    MODEL_FILE = f'../saved_models/test/glase_{dataset}_d{d}_normalized_{init}_{gd_steps}steps.pt'
    TRAIN_DATA_FILE = f'../data/{dataset}_train.pkl'
    VAL_DATA_FILE = f'../data/{dataset}_val.pkl'
    Q_FILE = f'../data/{dataset}_q_file.pkl'

elif config[dataset]['mode'] == "subgraphs":
    dropout = config[dataset]['dropout']
    MODEL_FILE = f'../saved_models/test/glase_{dataset}_0{dropout}_d{d}_normalized_{init}_{gd_steps}steps.pt'
    TRAIN_DATA_FILE = f'../data/{dataset}_0{dropout}_train.pkl'
    VAL_DATA_FILE = f'../data/{dataset}_0{dropout}_val.pkl'
    Q_FILE = f'../data/{dataset}_0{dropout}_q_file.pkl'

model = gLASE(d,d, gd_steps)
model.init_lase(lr)
model.to(device)

    
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
early_stopper = EarlyStopper(patience=10, min_delta=0)

with open(TRAIN_DATA_FILE, 'rb') as f:
    df_train = pickle.load(f)

with open(VAL_DATA_FILE, 'rb') as f:
    df_val = pickle.load(f)

train_loader =  DataLoader(df_train, batch_size=1, shuffle = True)
val_loader =  DataLoader(df_val, batch_size=1, shuffle = False)

# Calculate q
q_array = []
for data in df_train:
    adj_matrix = to_dense_adj(data.edge_index).squeeze(0)
    u, V = torch.linalg.eig(adj_matrix)
    list_q=[]
    for i in range(d):
        if u[i].numpy()>0:
            list_q.append(1)
        else:
            list_q.append(-1)
    list_q.sort(reverse=True)
    q_array.append(list_q)

tuples = [tuple(lst) for lst in q_array]
count = Counter(tuples)
most_common_element, max_frequency = count.most_common(1)[0]
q = torch.tensor(most_common_element).float()
Q=torch.diag(q).to(device)
print('Q=',Q)

with open(Q_FILE, 'wb') as f:
    pickle.dump(q, f)

train_loss_epoch=[]
val_loss_epoch=[]
min_val_loss = np.inf

start = time.time()

for epoch in range(epochs):
    # Train
    train_loss_step=[]
    model.train()
    train_loop = tqdm(train_loader)
    for i, batch in enumerate(train_loop):  
        batch.to(device) 
        optimizer.zero_grad()

        ## Initialization x
        if init == 'random':
            x = batch.x
        elif init == 'glase_init':
            x = get_x_init(num_nodes, d, 0, math.pi/2, 0, math.pi/2).to(device)
        elif init == 'ones':
            x = torch.ones((num_nodes, d)).to(device)

        out = model(x, batch.edge_index, batch.edge_index_2, Q, batch.mask)
        loss = torch.norm((out@Q@out.T - to_dense_adj(batch.edge_index).squeeze(0))*to_dense_adj(batch.mask).squeeze(0))
        loss.backward() 
        optimizer.step() 

        train_loss_step.append(loss.detach().to('cpu').numpy())
        train_loop.set_description(f"Epoch [{epoch}/{epochs}]")
        train_loop.set_postfix(loss=loss)

    train_loss_epoch.append(np.array(train_loss_step).mean())
        
    # Validation
    val_loss_step=[] 
    model.eval()
    val_loop = tqdm(val_loader)
    for i, batch in enumerate(val_loop):
        batch.to(device)       

        ## Initialization x
        if init == 'random':
            x = batch.x
        elif init == 'glase_init':
            x = get_x_init(num_nodes, d, 0, math.pi/2, 0, math.pi/2).to(device)
        elif init == 'ones':
            x = torch.ones((num_nodes, d)).to(device)

        out = model(x, batch.edge_index, batch.edge_index_2, Q, batch.mask)
        loss = torch.norm((out@Q@out.T - to_dense_adj(batch.edge_index).squeeze(0))*to_dense_adj(batch.mask).squeeze(0))

        val_loss_step.append(loss.detach().to('cpu').numpy())
        val_loop.set_description(f"Epoch [{epoch}/{epochs}]")
        val_loop.set_postfix(loss=loss)

    val_loss_epoch.append(np.array(val_loss_step).mean())

    if val_loss_epoch[epoch] < min_val_loss:
        torch.save(model.state_dict(), MODEL_FILE)
        min_val_loss = val_loss_epoch[epoch]
        print("Best model updated")
        print("Val loss Avg: ", min_val_loss)

    if early_stopper.early_stop(val_loss_epoch[epoch]):    
        optimal_epoch = np.argmin(val_loss_epoch)
        print("Optimal epoch: ", optimal_epoch)         
        stop = time.time()
        print(f"Training time: {stop - start}s")
        break

stop = time.time()
print(f"Training time: {stop - start}s")