import sys
sys.path.append("../")

import warnings
warnings.filterwarnings("ignore")

import argparse
import torch
import math
import time
import pickle
import torch.nn as nn
import numpy as np 

from torch_geometric.utils import to_dense_adj, from_networkx
from networkx import watts_strogatz_graph

from models.glase_classifier import gLASE_e2e_GAT
from training.get_init import get_x_init


parser = argparse.ArgumentParser(description='Classifier')
parser.add_argument('--dataset', type=str, default='cora', help='[cora, amazon, chameleon]')
parser.add_argument('--mask', type=str, default='FULL', help='[cora, amazon, chameleon]')
parser.add_argument('--d', type=int, default=4)
parser.add_argument('--att_mask', type=str, default='FULL')
parser.add_argument('--glase_steps', type=int, default=5)
parser.add_argument('--iter', type=int, default=10)
parser.add_argument('--alpha', type=float, default=0.99)


args = parser.parse_args()
dataset = args.dataset
mask = args.mask
d = args.d
att_mask = args.att_mask
gd_steps = args.glase_steps
total_iter = args.iter
alpha = args.alpha

Q_FILE = f'../data/real_dataset/{dataset}_q.pkl'
DATASET_FILE = f'../data/real_dataset/{dataset}_dataset.pkl'
MASK_FILE = f'../data/real_dataset/{dataset}_mask_{mask}.pkl'

GLASE_XINIT = f'../data/real_dataset/{dataset}_glase_e2e_xinit_d{d}_{gd_steps}steps_{mask}_{alpha}.pkl'
GLASE_MODEL_FILE=f'../saved_models/{dataset}/{dataset}_gat_classifier_glase_e2e_d{d}_{mask}_{alpha}.pt'
E2E_RESULTS = f'./results/{dataset}/{dataset}_glase_e2e_results_{mask}_d{d}_{alpha}.pkl'


## LOAD DATASET 
with open(DATASET_FILE, 'rb') as f:
    data = pickle.load(f)

with open(Q_FILE, 'rb') as f:
    q = pickle.load(f)


device = 'cuda'
feature_dim = data.x.shape[1] # dimensionality of the word embeddings
embedding_dim = d  # dimensionality of the graph embeddings
hidden_dim = 32  # number of hidden units
h_embedding_dim = 32
output_dim = torch.unique(data.y).shape[0] # number of classes
n_layers = 3
dropout1 = 0.5
dropout2 = 0.5
epochs = 1000
lr=1e-2



data.to(device)

num_nodes = data.num_nodes
Q = torch.diag(q[:d]).to(device)

if att_mask == 'FULL':
    edge_index_2 = torch.ones([num_nodes,num_nodes],).nonzero().t().contiguous().to(device)
else:
    edge_index_2 =  from_networkx(watts_strogatz_graph(num_nodes, 700, 0.1, seed=None)).edge_index.to(device)


with open(MASK_FILE, 'rb') as f:
    mask = pickle.load(f)
mask = mask.to(device)

x = get_x_init(num_nodes, d, 0, math.pi/2, 0, math.pi/2).to(device)

with open(GLASE_XINIT, 'wb') as f:
    pickle.dump(x, f)

acc_glase_val = []
loss_glase_val = []
acc_glase_test = [] 
for iter in range(total_iter):
    train_index = data.train_idx[:,iter]
    val_index = data.val_idx[:,iter]
    test_index = data.test_idx[:,iter]

    model = gLASE_e2e_GAT(feature_dim, embedding_dim, hidden_dim, h_embedding_dim,output_dim, n_layers, dropout1, dropout2, gd_steps, num_heads=1)
    model.init_lase(lr)
    model.to(device)    

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    train_loss_epoch=[]
    val_loss_epoch=[]
    test_loss_epoch=[]
    best_valid_acc = 0 
    best_epoch = 0 

    start = time.time()

    for epoch in range(epochs):
        ## Train
        model.train()
        optimizer.zero_grad()
        out, x_glase = model(data.x, x, data.edge_index, edge_index_2, Q, mask)
        loss1 = criterion(out[train_index], data.y[train_index].squeeze().to(device))
        loss2 = torch.norm((x_glase@x_glase.T - to_dense_adj(data.edge_index, max_num_nodes=num_nodes).squeeze(0))*to_dense_adj(mask, max_num_nodes=num_nodes).squeeze(0))
        loss = alpha*loss1 + (1-alpha)*loss2

        loss.backward() 
        optimizer.step() 
        
        # Calculate accuracy 
        _, predicted_labels = torch.max(out[train_index].squeeze(),1)
        total_train_correct = (predicted_labels.squeeze() == data.y[train_index].squeeze().to(device)).sum().item()
        total_train_samples = len(train_index)
        train_acc = total_train_correct / total_train_samples
        total_train_loss = loss.item()    
        
        ## Val
        model.eval()
        out, x_glase = model(data.x, x, data.edge_index, edge_index_2, Q, mask)
        loss1 = criterion(out[val_index], data.y[val_index].squeeze().to(device))
        loss2 = torch.norm((x_glase@x_glase.T - to_dense_adj(data.edge_index, max_num_nodes=num_nodes).squeeze(0))*to_dense_adj(mask, max_num_nodes=num_nodes).squeeze(0))
        loss = alpha*loss1 + (1-alpha)*loss2
            
        # Calculate accuracy
        _, predicted_labels = torch.max(out[val_index].squeeze(),1)
        total_val_correct = (predicted_labels.squeeze() == data.y[val_index].squeeze().to(device)).sum().item()
        total_val_samples = len(val_index)
        valid_acc = total_val_correct / total_val_samples
        total_val_loss = loss.item()            
        
        ## Test
        loss1 = criterion(out[train_index], data.y[train_index].squeeze().to(device))
        loss2 = torch.norm((x_glase@x_glase.T - to_dense_adj(data.edge_index, max_num_nodes=num_nodes).squeeze(0))*to_dense_adj(mask, max_num_nodes=num_nodes).squeeze(0))
        loss = alpha*loss1 + (1-alpha)*loss2
            
        # Calculate accuracy
        _, predicted_labels = torch.max(out[test_index].squeeze(),1)
        total_test_correct = (predicted_labels.squeeze() == data.y[test_index].squeeze().to(device)).sum().item()
        total_test_samples = len(test_index)
        test_acc = total_test_correct / total_test_samples
        total_test_loss = loss.item()      
                
        train_loss_epoch.append(total_train_loss)
        val_loss_epoch.append(total_val_loss)
        test_loss_epoch.append(total_test_loss)
        
        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            torch.save(model.state_dict(), GLASE_MODEL_FILE)
            best_epoch = epoch
        
        if epoch % 100 == 0:
            # Print epoch statistics
            print(f'Epoch: {epoch:02d}, '
                f'Loss: {total_train_loss:.4f}, '
                f'Loss1: {loss1.item():.4f}, '
                f'Loss2: {loss2.item():.4f}, '
                f'Train: {100 * train_acc:.2f}%, '
                f'Valid: {100 * valid_acc:.2f}% '
                f'Test: {100 * test_acc:.2f}%')

    stop = time.time()
    print(f"Training time: {stop - start}s")


    model.load_state_dict(torch.load(GLASE_MODEL_FILE))
    model.to(device)

    model.eval()
    out, x_glase = model(data.x, x, data.edge_index, edge_index_2, Q, mask)

    # Calculate accuracy val
    loss1 = criterion(out[val_index], data.y[val_index].squeeze().to(device))
    loss2 = torch.norm((x_glase@x_glase.T - to_dense_adj(data.edge_index, max_num_nodes=num_nodes).squeeze(0))*to_dense_adj(mask, max_num_nodes=num_nodes).squeeze(0))
    loss = alpha*loss1 + (1-alpha)*loss2
    _, predicted_labels = torch.max(out[val_index].squeeze(),1)
    total_val_correct = (predicted_labels.squeeze() == data.y[val_index].squeeze().to(device)).sum().item()
    total_val_samples = len(val_index)
    val_acc = total_val_correct / total_val_samples
    total_val_loss = loss.item()      
    acc_glase_val.append(val_acc)
    loss_glase_val.append(total_val_loss)


    loss1 = criterion(out[test_index], data.y[test_index].squeeze().to(device))
    loss2 = torch.norm((x_glase@x_glase.T - to_dense_adj(data.edge_index, max_num_nodes=num_nodes).squeeze(0))*to_dense_adj(mask, max_num_nodes=num_nodes).squeeze(0))
    loss = alpha*loss1 + (1-alpha)*loss2

    # Calculate accuracy
    _, predicted_labels = torch.max(out[test_index].squeeze(),1)
    total_test_correct = (predicted_labels.squeeze() == data.y[test_index].squeeze().to(device)).sum().item()
    total_test_samples = len(test_index)
    test_acc = total_test_correct / total_test_samples
    total_test_loss = loss.item()      
    acc_glase_test.append(test_acc)
    
    with open(E2E_RESULTS, 'wb') as f:
        pickle.dump([acc_glase_val, loss_glase_val, acc_glase_test], f)
    
    
        
    
print(f'{np.array(acc_glase_test).mean()*100} +/- {np.array(acc_glase_test).std()*100}')


    