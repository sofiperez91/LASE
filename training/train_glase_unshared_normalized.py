import sys
sys.path.append("../")
import math
import time
import torch
import numpy as np
import torch_geometric as pyg
from torch_geometric.utils import stochastic_blockmodel_graph, to_dense_adj, erdos_renyi_graph
from models.GLASE_unshared_normalized import gLASE 
from models.early_stopper import EarlyStopper
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import pickle
from tqdm import tqdm
from networkx import watts_strogatz_graph
from torch_geometric.utils.convert import from_networkx
from models.bigbird_attention import big_bird_attention
from training.get_init import get_x_init


device = 'cpu'
epochs = 50000
lr=1e-4
gd_steps = 20

d = 5
MODEL_FILE='../saved_models/glase_unshared_d5_normalized_unbalanced_initx_20steps.pt'
TRAIN_DATA_FILE='../data/sbm5_neg_train.pkl'
VAL_DATA_FILE='../data/sbm5_neg_train.pkl'

num_nodes = 350
n = [100, 50, 90, 60, 50]
p = [
    [0.1, 0.9, 0.7, 0.9, 0.8],
    [0.9, 0.8, 0.8, 0.9, 0.7],
    [0.7, 0.8, 0.3, 0.8, 0.7],
    [0.9, 0.9, 0.8, 0.8, 0.8],
    [0.8, 0.7, 0.7, 0.8, 0.1],
]

q = torch.Tensor([[1,-1,-1,-1,-1]])
Q=torch.diag(q[0]).to(device)
print('Q=',Q)

# Define mask
mask = (torch.ones([num_nodes,num_nodes],)-torch.eye(num_nodes)).nonzero().t().contiguous().to(device)
edge_index_2 = torch.ones([num_nodes,num_nodes],).nonzero().t().contiguous().to(device)


model = gLASE(d,d, gd_steps)
model.to(device)

## Initialization
for step in range(gd_steps):
    model.gd[step].lin1.weight.data = torch.nn.init.xavier_uniform_(model.gd[step].lin1.weight)*lr
    model.gd[step].lin2.weight.data = torch.nn.init.xavier_uniform_(model.gd[step].lin2.weight)*lr

    

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
        x  = get_x_init(num_nodes, d, 0, math.pi/2, 0, math.pi/2).to(device)
        out = model(x, batch.edge_index, edge_index_2, Q, mask)
        loss = torch.norm((out@Q@out.T - to_dense_adj(batch.edge_index).squeeze(0))*to_dense_adj(mask).squeeze(0))
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
        x  = get_x_init(num_nodes, d,0, math.pi/2, 0, math.pi/2).to(device)
        out = model(x, batch.edge_index, edge_index_2, Q, mask)
        loss = torch.norm((out@Q@out.T - to_dense_adj(batch.edge_index).squeeze(0))*to_dense_adj(mask).squeeze(0))

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