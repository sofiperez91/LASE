import sys
sys.path.append("../")
import math
import time
import torch
import numpy as np
from torch_geometric.utils import to_dense_adj
from models.LASE_unshared_normalized import LASE
from models.early_stopper import EarlyStopper
from torch_geometric.loader import DataLoader
import pickle
from tqdm import tqdm
from training.get_init import get_x_init


device = 'cuda'
epochs = 200
lr=1e-3
gd_steps = 5

d = 4
MODEL_FILE='../saved_models/lase_unshared_normalized_unbalanced_bipartite_rand.pt'
TRAIN_DATA_FILE='../data/bipartite_unbalanced_positive_train_subgraphs_095.pkl'
VAL_DATA_FILE='../data/bipartite_unbalanced_positive_val_subgraphs_095.pkl'

# d = 3
# MODEL_FILE='../saved_models/lase_unshared_d3_normalized_unbalanced_subgraphs_rand.pt'
# TRAIN_DATA_FILE='../data/sbm3_unbalanced_positive_train_subgraphs_095.pkl'
# VAL_DATA_FILE='../data/sbm3_unbalanced_positive_val_subgraphs_095.pkl'

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
        # x  = get_x_init(batch.num_nodes, d, 0, math.pi/2, 0, math.pi/2).to(device)
        x = torch.rand((batch.num_nodes, d)).to(device)
        out = model(x, batch.edge_index, batch.edge_index_2, batch.mask)
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
        # x  = get_x_init(batch.num_nodes, d,0, math.pi/2, 0, math.pi/2).to(device)
        x = torch.rand((batch.num_nodes, d)).to(device)
        out = model(x, batch.edge_index, batch.edge_index_2, batch.mask)
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