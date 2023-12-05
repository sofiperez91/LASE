import sys
sys.path.append("../")
import math
import time
import torch
import numpy as np
import torch_geometric as pyg
from torch_geometric.utils import stochastic_blockmodel_graph, to_dense_adj, erdos_renyi_graph
from models.LASE_unshared_normalized import LASE
from models.early_stopper import EarlyStopper
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import pickle
from tqdm import tqdm
from networkx import watts_strogatz_graph
from torch_geometric.utils.convert import from_networkx
from models.bigbird_attention import big_bird_attention
from training.get_init import get_x_init


device = 'cuda'
epochs = 200
lr=1e-3
gd_steps = 5

d = 2
MODEL_FILE='../saved_models/lase_unshared_d2_normalized_unbalanced_rand_M03.pt'
TRAIN_DATA_FILE='../data/sbm2_unbalanced_train.pkl'
VAL_DATA_FILE='../data/sbm2_unbalanced_val.pkl'

num_nodes = 100
n = [70, 30]
p = [
     [0.9, 0.1],
     [0.1, 0.5]
]

edge_index_2 = torch.ones([num_nodes,num_nodes],).nonzero().t().contiguous().to(device)


model = LASE(d,d, gd_steps)
model.to(device)

## Initialization
for step in range(gd_steps):
    # TAGConv
    model.gd[step].gcn.lins[0].weight.data = torch.eye(d).to(device)
    model.gd[step].gcn.lins[0].weight.requires_grad = False
    model.gd[step].gcn.lins[1].weight.data = torch.nn.init.xavier_uniform_(model.gd[step].gcn.lins[1].weight)*lr

    # TransformerBlock
    model.gd[step].gat.lin2.weight.data = lr*torch.nn.init.xavier_uniform_(model.gd[step].gat.lin2.weight.data).to(device)

    model.gd[step].gat.lin3.weight.data = torch.eye(d).to(device)
    model.gd[step].gat.lin3.weight.requires_grad = False
    model.gd[step].gat.lin4.weight.data = torch.eye(d).to(device)
    model.gd[step].gat.lin4.weight.requires_grad = False
    

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
early_stopper = EarlyStopper(patience=10, min_delta=0)

with open(TRAIN_DATA_FILE, 'rb') as f:
    df_train = pickle.load(f)
    
n_1 = 10
n_2 = 5
selected_nodes = list(range(n_1)) + list(range(n[0],n[0]+n_2))

df_train = []
for data in df_train:
    M03 = torch.ones([num_nodes,num_nodes]).squeeze(0)
    for i in selected_nodes:
        votos = (torch.rand(1, num_nodes) < 0.3).int()
        M03[i,:] = votos
        M03[:,i] = votos        
    df_train.append(Data(x = data.x, edge_index = data.edge_index, mask = M03.nonzero().t().contiguous()))

with open(VAL_DATA_FILE, 'rb') as f:
    df_val = pickle.load(f)
df_val = []
for data in df_val:
    M03 = torch.ones([num_nodes,num_nodes]).squeeze(0)
    for i in selected_nodes:
        votos = (torch.rand(1, num_nodes) < 0.3).int()
        M03[i,:] = votos
        M03[:,i] = votos     
    df_val.append(Data(x = data.x, edge_index = data.edge_index, mask = M03.nonzero().t().contiguous()))

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
        out = model(x, batch.edge_index, edge_index_2, batch.mask)
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
        x = torch.rand((num_nodes, d)).to(device)
        out = model(x, batch.edge_index, edge_index_2, batch.mask)
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