import sys
sys.path.append("../")
import math
import time
import torch
import numpy as np
from torch_geometric.utils import to_dense_adj
from models.GCN import GCN
# from models.GD_GCN_v2 import GD_Unroll as GCN
from models.early_stopper import EarlyStopper
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import pickle
from tqdm import tqdm
from training.get_init import get_x_init


device = 'cuda'
epochs = 50000
lr=1e-3
K=3

# d = 3
# MODEL_FILE='../saved_models/gcn_d3.pt'
# TRAIN_DATA_FILE='../data/sbm3_unbalanced_train.pkl'
# VAL_DATA_FILE='../data/sbm3_unbalanced_val.pkl'

# num_nodes = 150
# n = [70, 50, 30]

# p = [
#      [0.5, 0.1, 0.3],
#      [0.1, 0.9, 0.2], 
#      [0.3, 0.2, 0.7]
# ]

d = 3
MODEL_FILE='../saved_models/gcn_d3_sim_random.pt'
TRAIN_DATA_FILE='../data/sbm3_sim_train.pkl'
VAL_DATA_FILE='../data/sbm3_sim_val.pkl'

num_nodes = 150
n = [50, 50, 50]

p = [
     [0.7, 0.1, 0.1],
     [0.1, 0.7, 0.1], 
     [0.1, 0.1, 0.7]
]


model = GCN(d,d, K)
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
early_stopper = EarlyStopper(patience=10, min_delta=0)

with open(TRAIN_DATA_FILE, 'rb') as f:
    df_train = pickle.load(f)
df_train = [Data(x = data.x, edge_index = data.edge_index) for data in df_train]

with open(VAL_DATA_FILE, 'rb') as f:
    df_val = pickle.load(f)
df_val = [Data(x = data.x, edge_index = data.edge_index) for data in df_val]

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
        loss = torch.norm(out@out.T - to_dense_adj(batch.edge_index).squeeze(0))
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
        loss = torch.norm(out@out.T - to_dense_adj(batch.edge_index).squeeze(0))

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