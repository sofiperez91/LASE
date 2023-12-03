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

# d = 3
# MODEL_FILE='../saved_models/lase_unshared_d3_normalized_unbalanced_sim_random.pt'
# TRAIN_DATA_FILE='../data/sbm3_sim_train.pkl'
# VAL_DATA_FILE='../data/sbm3_sim_val.pkl'

# num_nodes = 150
# device = 'cpu'
# n = [50, 50, 50]

# p = [
#      [0.7, 0.1, 0.1],
#      [0.1, 0.7, 0.1], 
#      [0.1, 0.1, 0.7]
# ]


# num_nodes = 350
# n = [100, 50, 90, 60, 50]
# p = [
#     [0.9, 0.1, 0.2, 0.2, 0.1],
#     [0.1, 0.8, 0.1, 0.1, 0.2],
#     [0.2, 0.1, 0.7, 0.2, 0.1],
#     [0.2, 0.1, 0.2, 0.8, 0.2],
#     [0.1, 0.2, 0.1, 0.2, 0.9],
# ]

# d = 10
# MODEL_FILE='../saved_models/lase_unshared_d10_normalized_unbalanced_initx_20steps.pt'
# TRAIN_DATA_FILE='../data/sbm10_unbalanced_train.pkl'
# VAL_DATA_FILE='../data/sbm10_unbalanced_val.pkl'

# num_nodes = 550
# n = [80, 50, 80, 40, 50, 60, 50, 50, 40, 50]
# p = [
#     [0.9, 0.1, 0.1, 0.3, 0.1, 0.1, 0.2, 0.1, 0.1, 0.1],
#     [0.1, 0.6, 0.3, 0.3, 0.2, 0.2, 0.1, 0.2, 0.3, 0.2],
#     [0.1, 0.3, 0.8, 0.2, 0.1, 0.3, 0.1, 0.1, 0.2, 0.2],
#     [0.3, 0.3, 0.2, 0.7, 0.3, 0.1, 0.3, 0.1, 0.3, 0.3],
#     [0.1, 0.2, 0.1, 0.3, 0.9, 0.1, 0.2, 0.1, 0.1, 0.2],
#     [0.1, 0.2, 0.3, 0.1, 0.1, 0.5, 0.2, 0.1, 0.1, 0.3],
#     [0.2, 0.1, 0.1, 0.3, 0.2, 0.2, 0.8, 0.2, 0.3, 0.1],
#     [0.1, 0.2, 0.1, 0.1, 0.1, 0.1, 0.2, 0.6, 0.1, 0.1],
#     [0.1, 0.3, 0.2, 0.3, 0.1, 0.1, 0.3, 0.1, 0.9, 0.1],
#     [0.1, 0.2, 0.2, 0.3, 0.2, 0.3, 0.1, 0.1, 0.1, 0.7],  
# ]

# d = 5
# MODEL_FILE='../saved_models/lase_unshared_d5_normalized_unbalanced_rand_20steps.pt'
# TRAIN_DATA_FILE='../data/sbm5_unbalanced_train.pkl'
# VAL_DATA_FILE='../data/sbm5_unbalanced_val.pkl'

# num_nodes = 350
# n = [100, 50, 90, 60, 50]
# p = [
#     [0.9, 0.1, 0.2, 0.2, 0.1],
#     [0.1, 0.8, 0.1, 0.1, 0.2],
#     [0.2, 0.1, 0.7, 0.2, 0.1],
#     [0.2, 0.1, 0.2, 0.8, 0.2],
#     [0.1, 0.2, 0.1, 0.2, 0.9],
# ]

# d = 3
# MODEL_FILE='../saved_models/lase_unshared_d3_normalized_unbalanced_rand_20steps.pt'
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
MODEL_FILE='../saved_models/lase_unshared_d3_normalized_unbalanced_rand_v2_WS05.pt'
TRAIN_DATA_FILE='../data/sbm3_train_2.pkl'
VAL_DATA_FILE='../data/sbm3_val_2.pkl'

num_nodes = 240
n = [120, 80, 40]

p = [
     [0.9, 0.2, 0.1],
     [0.2, 0.6, 0.2],
     [0.1, 0.2, 0.7]
]

# d = 2
# MODEL_FILE='../saved_models/lase_unshared_d2_normalized_unbalanced_rand.pt'
# TRAIN_DATA_FILE='../data/sbm2_unbalanced_train.pkl'
# VAL_DATA_FILE='../data/sbm2_unbalanced_val.pkl'

# num_nodes = 100
# n = [70, 30]
# p = [
#      [0.9, 0.1],
#      [0.1, 0.5]
# ]

# Define mask
mask = (torch.ones([num_nodes,num_nodes],)-torch.eye(num_nodes)).nonzero().t().contiguous().to(device)
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
        # x = torch.ones((num_nodes, d)).to(device)
        x = torch.rand((num_nodes, d)).to(device)
        # x  = get_x_init(num_nodes, d, 0, math.pi/2, 0, math.pi/2).to(device)
        # ER05 = erdos_renyi_graph(num_nodes, 0.5, directed=False).to(device)
        # ER03 = erdos_renyi_graph(num_nodes, 0.3, directed=False).to(device)
        # ER01 = erdos_renyi_graph(num_nodes, 0.1, directed=False).to(device)
        WS05 = from_networkx(watts_strogatz_graph(num_nodes, 120, 0.1, seed=None)).edge_index.to(device)
        # WS03 = from_networkx(watts_strogatz_graph(num_nodes, 72, 0.1, seed=None)).edge_index.to(device)
        # WS01 = from_networkx(watts_strogatz_graph(num_nodes, 24, 0.1, seed=None)).edge_index.to(device)
        # BB_05_01 = big_bird_attention(70, 0.05, 240).to(device)  
        # BB_03_01 = big_bird_attention(30, 0.1, 240).to(device)
        # BB_01_005 = big_bird_attention(6, 0.05, 240).to(device)
        # BB=big_bird_attention(5, 0.08, 350).to(device)
        # BB=big_bird_attention(10, 0.05, 350).to(device)
        # BB=big_bird_attention(15, 0.02, num_nodes).to(device)
        # BB=big_bird_attention(50, 0.05, num_nodes).to(device)
        out = model(x, batch.edge_index, WS05, mask)
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
        # x = torch.ones((num_nodes, d)).to(device)
        x = torch.rand((num_nodes, d)).to(device)
        # x  = get_x_init(num_nodes, d,0, math.pi/2, 0, math.pi/2).to(device)
        # ER05 = erdos_renyi_graph(num_nodes, 0.5, directed=False).to(device)
        # ER03 = erdos_renyi_graph(num_nodes, 0.3, directed=False).to(device)
        # ER01 = erdos_renyi_graph(num_nodes, 0.1, directed=False).to(device)
        WS05 = from_networkx(watts_strogatz_graph(num_nodes, 120, 0.1, seed=None)).edge_index.to(device)
        # WS03 = from_networkx(watts_strogatz_graph(num_nodes, 72, 0.1, seed=None)).edge_index.to(device)
        # WS01 = from_networkx(watts_strogatz_graph(num_nodes, 24, 0.1, seed=None)).edge_index.to(device)
        # BB_05_01 = big_bird_attention(70, 0.05, 240).to(device) 
        # BB_03_01 = big_bird_attention(30, 0.1, 240).to(device)
        # BB_01_005 = big_bird_attention(6, 0.05, 240).to(device)
        # BB=big_bird_attention(5, 0.08, 350).to(device)
        # BB=big_bird_attention(10, 0.05, 350).to(device)
        # BB=big_bird_attention(15, 0.02, num_nodes).to(device)
        # BB=big_bird_attention(50, 0.05, num_nodes).to(device)
        out = model(x, batch.edge_index, WS05, mask)
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