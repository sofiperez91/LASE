import sys
sys.path.append("../")

import warnings
warnings.filterwarnings("ignore")

import torch
import time
import pickle
import math
import argparse
import numpy as np
from tqdm import tqdm

from torch_geometric.utils import to_dense_adj
from torch_geometric.loader import DataLoader

from models.GLASE_unshared_normalized import gLASE
from models.early_stopper import EarlyStopper
from training.get_init import get_x_init


parser = argparse.ArgumentParser(description='Embeddings')
parser.add_argument('--dataset', type=str, default='cora', help='[cora, citeseer, amazon, chameleon, squirrel, cornell]')
parser.add_argument('--mask', type=str, default='FULL', help='FULL, M08, M06, M04, M02]')
parser.add_argument('--d', type=int, default=6)
parser.add_argument('--glase_steps', type=int, default=5)


args = parser.parse_args()
dataset = args.dataset
mask = args.mask
d = args.d
gd_steps = args.glase_steps

MODEL_FILE=f'../saved_models/{dataset}/{dataset}_glase_unshared_{gd_steps}steps_d{d}_{mask}.pt'
TRAIN_DATA_FILE = f'../data/real_dataset/{dataset}_train_subgraphs.pkl'
VAL_DATA_FILE = f'../data/real_dataset/{dataset}_val_subgraphs.pkl'
q_file = f'../data/real_dataset/{dataset}_q.pkl'


## Load data
with open(TRAIN_DATA_FILE, 'rb') as f:
    df_train = pickle.load(f)

with open(VAL_DATA_FILE, 'rb') as f:
    df_val = pickle.load(f)

with open(q_file, 'rb') as f:
    q = pickle.load(f)

train_loader = DataLoader(df_train, batch_size=1, shuffle = True)
val_loader = DataLoader(df_val, batch_size=1, shuffle = False)   

device = 'cuda'
epochs = 300
lr=1e-3

Q = torch.diag(q[:d]).to(device)

print('Starting training')
print('Q=',Q)

model = gLASE(d,d, gd_steps)
model.init_lase(lr)
model.to(device)    

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
early_stopper = EarlyStopper(patience=10, min_delta=0)

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

        ## MASK
        if mask == 'FULL':
            mask_matrix = batch.edge_index_2
        elif mask == 'M08':
            mask_matrix = batch.sub_mask_M08
        elif mask == 'M06':
            mask_matrix = batch.sub_mask_M06
        elif mask == 'M04':
            mask_matrix = batch.sub_mask_M04
        elif mask == 'M02':
            mask_matrix = batch.sub_mask_M02

        out = model(batch.x_init, batch.edge_index, batch.edge_index_2, Q, mask_matrix)
        loss = torch.norm((out@Q@out.T - to_dense_adj(batch.edge_index, max_num_nodes=batch.num_nodes).squeeze(0))*to_dense_adj(mask_matrix, max_num_nodes=batch.num_nodes).squeeze(0))
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

        ## MASK
        if mask == 'FULL':
            mask_matrix = batch.edge_index_2
        elif mask == 'M08':
            mask_matrix = batch.sub_mask_M08
        elif mask == 'M06':
            mask_matrix = batch.sub_mask_M06
        elif mask == 'M04':
            mask_matrix = batch.sub_mask_M04
        elif mask == 'M02':
            mask_matrix = batch.sub_mask_M02

        out = model(batch.x_init, batch.edge_index, batch.edge_index_2, Q, mask_matrix)
        loss = torch.norm((out@Q@out.T - to_dense_adj(batch.edge_index, max_num_nodes=batch.num_nodes).squeeze(0))* to_dense_adj(mask_matrix, max_num_nodes=batch.num_nodes).squeeze(0))

        val_loss_step.append(loss.detach().to('cpu').numpy())
        val_loop.set_description(f"Epoch [{epoch}/{epochs}]")
        val_loop.set_postfix(loss=loss)

    val_loss_epoch.append(np.array(val_loss_step).mean())

    if val_loss_epoch[epoch] < min_val_loss:
        torch.save(model.state_dict(), MODEL_FILE)
        min_val_loss = val_loss_epoch[epoch]
        print("Best model updated")
        print("Val loss: ", min_val_loss)

    if early_stopper.early_stop(val_loss_epoch[epoch]):    
        optimal_epoch = np.argmin(val_loss_epoch)
        print("Optimal epoch: ", optimal_epoch)         
        break    

stop = time.time()
print(f"Training time: {stop - start}s")


## GENERATE EMBEDDINGS

DATASET_FILE = f'../data/real_dataset/{dataset}_dataset.pkl'
Q_FILE = f'../data/real_dataset/{dataset}_q.pkl'
MASK_FILE = f'../data/real_dataset/{dataset}_mask_{mask}.pkl'
EMBEDDING_FILE = f'../data/real_dataset/{dataset}_glase_embeddings_d{d}_{gd_steps}steps_{mask}.pkl'

with open(DATASET_FILE, 'rb') as f:
    data = pickle.load(f)
data.to(device)

num_nodes = data.num_nodes
print(num_nodes)
edge_index_2 = torch.ones([num_nodes,num_nodes],).nonzero().t().contiguous().to(device)
x_init = get_x_init(num_nodes, d, 0, math.pi/2, 0, math.pi/2).to(device)

with open(MASK_FILE, 'rb') as f:
    mask = pickle.load(f)
mask = mask.to(device)

with open(Q_FILE, 'rb') as f:
    q = pickle.load(f)
Q = torch.diag(q[:d]).to(device)


model = gLASE(d,d, gd_steps)
model.load_state_dict(torch.load(MODEL_FILE))
model.to('cuda')

model.eval()
x_glase = model(x_init, data.edge_index, edge_index_2, Q, mask)

with open(EMBEDDING_FILE, 'wb') as f:
    pickle.dump(x_glase, f)