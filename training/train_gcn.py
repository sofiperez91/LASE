import sys
sys.path.append("../")
import time
import torch
import numpy as np
import json
from torch_geometric.utils import to_dense_adj
from models.GCN import GCN
from models.early_stopper import EarlyStopper
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import pickle
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser(description='Dataset')
parser.add_argument('--dataset', type=str, default='sbm2_sim_positive')
parser.add_argument('--epochs', type=int, default=5000)

args = parser.parse_args()
dataset = args.dataset
epochs = args.epochs


# Load the config file
with open('../data/data_config.json', 'r') as file:
    config = json.load(file)

device = 'cuda'
lr = 1e-4
d = config[dataset]['d']
n = config[dataset]['n']
num_nodes = np.sum(n)
K=3


if config[dataset]['mode'] == "simple":
    MODEL_FILE=f'../saved_models/gcn_{dataset}_loss2.pt'
    TRAIN_DATA_FILE=f'../data/synthetic_dataset/sbm/{dataset}_train.pkl'
    VAL_DATA_FILE=f'../data/synthetic_dataset/sbm/{dataset}_val.pkl'
elif config[dataset]['mode'] == "subgraphs":
    dropout = config[dataset]['dropout']
    MODEL_FILE=f'../saved_models/gcn_{dataset}_loss2.pt'
    TRAIN_DATA_FILE=f'../data/synthetic_dataset/sbm/{dataset}_0{dropout}_train.pkl'
    VAL_DATA_FILE=f'../data/synthetic_dataset/sbm/{dataset}_0{dropout}_val.pkl'


model = GCN(d,d, K)
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
early_stopper = EarlyStopper(patience=10, min_delta=0)

with open(TRAIN_DATA_FILE, 'rb') as f:
    df_train = pickle.load(f)

with open(VAL_DATA_FILE, 'rb') as f:
    df_val = pickle.load(f)

train_loader=  DataLoader(df_train, batch_size=1, shuffle = True)
val_loader =  DataLoader(df_val, batch_size=1, shuffle = False)

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
        x = torch.rand((num_nodes, d)).to(device)
        out = model(x, batch.edge_index)
        # loss = torch.norm(out@out.T - to_dense_adj(batch.edge_index).squeeze(0))
        loss = torch.norm(out@out.T - to_dense_adj(batch.edge_index).squeeze(0)*to_dense_adj(batch.mask).squeeze(0))
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
        x = torch.rand((num_nodes, d)).to(device)
        out = model(x, batch.edge_index)
        # loss = torch.norm(out@out.T - to_dense_adj(batch.edge_index).squeeze(0))
        loss = torch.norm(out@out.T - to_dense_adj(batch.edge_index).squeeze(0)*to_dense_adj(batch.mask).squeeze(0))

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