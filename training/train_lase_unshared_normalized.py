import sys
sys.path.append("../")

import time
import torch
import numpy as np
import json
import argparse
import pickle
from tqdm import tqdm

from torch_geometric.utils import to_dense_adj, erdos_renyi_graph
from torch_geometric.utils.convert import from_networkx
from torch_geometric.loader import DataLoader
from networkx import watts_strogatz_graph

from models.LASE_unshared_normalized import LASE
from models.early_stopper import EarlyStopper
from models.bigbird_attention import big_bird_attention


parser = argparse.ArgumentParser(description='Dataset')
parser.add_argument('--dataset', type=str, default='sbm2_unbalanced_positive')
parser.add_argument('--gd_steps', type=int, default=5)
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--init', type=str, default='random')
parser.add_argument('--att', type=str, default='FULL')

args = parser.parse_args()
dataset = args.dataset
gd_steps = args.gd_steps
epochs = args.epochs
init = args.init
att = args.att


# Load the config file
with open('../data/data_config.json', 'r') as file:
    config = json.load(file)

device = 'cuda'
lr = 1e-3
d = config[dataset]['d']
n = config[dataset]['n']
num_nodes = np.sum(n)

if config[dataset]['mode'] == "simple":
    MODEL_FILE=f'../saved_models/test/lase_{dataset}_d{d}_normalized_{init}_{gd_steps}steps_{att}.pt'
    TRAIN_DATA_FILE=f'../data/{dataset}_train.pkl'
    VAL_DATA_FILE=f'../data/{dataset}_val.pkl'
elif config[dataset]['mode'] == "subgraphs":
    dropout = config[dataset]['dropout']
    MODEL_FILE=f'../saved_models/test/lase_{dataset}_0{dropout}_d{d}_normalized_{init}_{gd_steps}steps_{att}.pt'
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
        elif init == 'ones':
            x = torch.ones((num_nodes, d)).to(device)

        ## Attention mask
        if att == 'FULL':
            edge_index_2 = batch.edge_index_2
        if att == 'ER05':
            edge_index_2 = erdos_renyi_graph(num_nodes, 0.5, directed=False).to(device)
        if att == 'ER03':
            edge_index_2 = erdos_renyi_graph(num_nodes, 0.3, directed=False).to(device)
        if att == 'ER01':
            edge_index_2 = erdos_renyi_graph(num_nodes, 0.1, directed=False).to(device)
        if att == 'WS05':
            edge_index_2 = from_networkx(watts_strogatz_graph(num_nodes, int(num_nodes*0.5), 0.1, seed=None)).edge_index.to(device)        
        if att == 'WS03':
            edge_index_2 = from_networkx(watts_strogatz_graph(num_nodes, int(num_nodes*0.3), 0.1, seed=None)).edge_index.to(device)
        if att == 'WS01':
            edge_index_2 = from_networkx(watts_strogatz_graph(num_nodes, int(num_nodes*0.1), 0.1, seed=None)).edge_index.to(device)
        if att == 'BB05':
            edge_index_2 = big_bird_attention(int(num_nodes*0.25), 0.1, num_nodes).to(device)       
        if att == 'BB03':
            edge_index_2 = big_bird_attention(int(num_nodes*0.125), 0.1, num_nodes).to(device)
        if att == 'BB01':
            edge_index_2 = big_bird_attention(int(num_nodes*0.025), 0.05, num_nodes).to(device)

        out = model(x, batch.edge_index, edge_index_2, batch.mask)
        loss = torch.norm((out@out.T - to_dense_adj(batch.edge_index).squeeze(0))*to_dense_adj(batch.mask).squeeze(0))
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
        elif init == 'ones':
            x = torch.ones((num_nodes, d)).to(device)

        ## Attention mask
        if att == 'FULL':
            edge_index_2 = batch.edge_index_2
        if att == 'ER05':
            edge_index_2 = erdos_renyi_graph(num_nodes, 0.5, directed=False).to(device)
        if att == 'ER03':
            edge_index_2 = erdos_renyi_graph(num_nodes, 0.3, directed=False).to(device)
        if att == 'ER01':
            edge_index_2 = erdos_renyi_graph(num_nodes, 0.1, directed=False).to(device)
        if att == 'WS05':
            edge_index_2 = from_networkx(watts_strogatz_graph(num_nodes, int(num_nodes*0.5), 0.1, seed=None)).edge_index.to(device)        
        if att == 'WS03':
            edge_index_2 = from_networkx(watts_strogatz_graph(num_nodes, int(num_nodes*0.3), 0.1, seed=None)).edge_index.to(device)
        if att == 'WS01':
            edge_index_2 = from_networkx(watts_strogatz_graph(num_nodes, int(num_nodes*0.1), 0.1, seed=None)).edge_index.to(device)
        if att == 'BB05':
            edge_index_2 = big_bird_attention(int(num_nodes*0.25), 0.1, num_nodes).to(device)       
        if att == 'BB03':
            edge_index_2 = big_bird_attention(int(num_nodes*0.125), 0.1, num_nodes).to(device)
        if att == 'BB01':
            edge_index_2 = big_bird_attention(int(num_nodes*0.025), 0.05, num_nodes).to(device)

        out = model(x, batch.edge_index, edge_index_2, batch.mask)
        loss = torch.norm((out@out.T - to_dense_adj(batch.edge_index).squeeze(0))*to_dense_adj(batch.mask).squeeze(0))

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