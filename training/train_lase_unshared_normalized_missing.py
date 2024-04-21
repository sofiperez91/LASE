import sys
sys.path.append("../")

import time
import torch
import numpy as np
import json
import argparse
import pickle
from tqdm import tqdm


from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_adj

from models.LASE_unshared_normalized import LASE
from models.early_stopper import EarlyStopper



parser = argparse.ArgumentParser(description='Dataset')
parser.add_argument('--dataset', type=str, default='sbm2_unbalanced_positive')
parser.add_argument('--gd_steps', type=int, default=5)
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--init', type=str, default='random')
parser.add_argument('--mask_threshold', type=float, default=0.7)


args = parser.parse_args()
dataset = args.dataset
gd_steps = args.gd_steps
epochs = args.epochs
init = args.init
mask_threshold = args.mask_threshold


# Load the config file
with open('../data/data_config.json', 'r') as file:
    config = json.load(file)

device = 'cuda'
lr = 1e-3
d = config[dataset]['d']
n = config[dataset]['n']
num_nodes = np.sum(n)

if config[dataset]['mode'] == "simple":
    MODEL_FILE=f'../saved_models/test/lase_{dataset}_d{d}_normalized_{init}_{gd_steps}steps_M0{int(mask_threshold*10)}.pt'
    TRAIN_DATA_FILE=f'../data/{dataset}_train.pkl'
    VAL_DATA_FILE=f'../data/{dataset}_val.pkl'
elif config[dataset]['mode'] == "subgraphs":
    dropout = config[dataset]['dropout']
    MODEL_FILE=f'../saved_models/test/lase_{dataset}_0{dropout}_d{d}_normalized_{init}_{gd_steps}steps_M0{int(mask_threshold*10)}.pt'
    TRAIN_DATA_FILE=f'../data/{dataset}_0{dropout}_train.pkl'
    VAL_DATA_FILE=f'../data/{dataset}_0{dropout}_val.pkl'


model = LASE(d,d, gd_steps)
model.init_lase(lr, d)
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
early_stopper = EarlyStopper(patience=10, min_delta=0)

with open(TRAIN_DATA_FILE, 'rb') as f:
    df_train = pickle.load(f)

with open(VAL_DATA_FILE, 'rb') as f:
    df_val = pickle.load(f)

train_loader=  DataLoader(df_train, batch_size=1, shuffle = True)
val_loader =  DataLoader(df_val, batch_size=1, shuffle = False)


## MASK
n_1 = 10
n_2 = 5
selected_nodes = list(range(n_1)) + list(range(n[0],n[0]+n_2))


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
            x = batch.x_init
        elif init == 'ones':
            x = torch.ones((num_nodes, d)).to(device)

        ## MASK
        mask_matrix = torch.ones([num_nodes, num_nodes]).squeeze(0)
        for i in selected_nodes:
            nodes = (torch.rand(1, num_nodes) < mask_threshold).int()
            mask_matrix[i, :] = nodes
            mask_matrix[:, i] = nodes
        mask = mask_matrix.nonzero().t().contiguous().to(device)

        out = model(x, batch.edge_index, batch.edge_index_2, mask)
        loss = torch.norm((out@out.T - to_dense_adj(batch.edge_index).squeeze(0))*to_dense_adj(mask).squeeze(0))
        loss.backward() 
        optimizer.step() 

        train_loss_step.append(loss.detach().to('cpu').numpy())
        train_loop.set_description(f"Epoch [{epoch}/{epochs}]")
        train_loop.set_postfix(loss=loss)
        
        # Break if loss is NaN
        if torch.isnan(loss):
            print(loss)
            break
        
    # Break if loss is NaN
    if torch.isnan(loss):
        print(loss)
        break

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
            x = batch.x_init
        elif init == 'ones':
            x = torch.ones((num_nodes, d)).to(device)

        ## MASK
        mask_matrix = torch.ones([num_nodes, num_nodes]).squeeze(0)
        for i in selected_nodes:
            nodes = (torch.rand(1, num_nodes) < mask_threshold).int()
            mask_matrix[i, :] = nodes
            mask_matrix[:, i] = nodes
        mask = mask_matrix.nonzero().t().contiguous().to(device)

        out = model(x, batch.edge_index, batch.edge_index_2, mask)
        loss = torch.norm((out@out.T - to_dense_adj(batch.edge_index).squeeze(0))*to_dense_adj(mask).squeeze(0))

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