import sys
sys.path.append("../")

import argparse
import torch
import torch.nn as nn
from torch_geometric.utils import to_dense_adj, stochastic_blockmodel_graph, from_networkx
from models.glase_classifier import gLASE_e2e_GAT, gLASE_e2e_Transformer
import numpy as np
import time
import math
from training.get_init import get_x_init
import pickle
from networkx import watts_strogatz_graph


parser = argparse.ArgumentParser(description='Classifier')
parser.add_argument('--dataset', type=str, default='cora', help='[cora, amazon, chameleon]')
parser.add_argument('--mask', type=str, default='FULL', help='[cora, amazon, chameleon]')

args = parser.parse_args()
dataset = args.dataset
mask = args.mask
    

Q_FILE = f'../data/{dataset}_q.pkl'
DATASET_FILE = f'../data/{dataset}_dataset.pkl'
GLASE_MODEL_FILE=f'../saved_models/glase_unshared_{dataset}_e2e_{mask}_v2.pt'
MASK_FILE = f'../data/{dataset}_mask_{mask}.pkl'
E2E_RESULTS = f'./results/glase_{dataset}_results_e2e_{mask}_v2.pkl'


## LOAD DATASET 
with open(DATASET_FILE, 'rb') as f:
    data = pickle.load(f)

print(data)

with open(Q_FILE, 'rb') as f:
    q = pickle.load(f)


device = 'cuda'
d = torch.unique(data.y).shape[0]
feature_dim = data.x.shape[1] # dimensionality of the word embeddings
embedding_dim = d  # dimensionality of the graph embeddings
hidden_dim = 32  # number of hidden units
h_embedding_dim = 32
output_dim = d  # number of classes
n_layers = 3
dropout1 = 0.5
dropout2 = 0.5
gd_steps = 5
epochs = 1000
lr=1e-2
alpha = 0.99


# A = to_dense_adj(data.edge_index).squeeze(0)
# u, V = torch.linalg.eig(A)

# list_q=[]
# for i in range(d):
#     if u[i].numpy()>0:
#         list_q.append(1)
#     else:
#         list_q.append(-1)
# list_q.sort(reverse=True)
# q = torch.Tensor(list_q)

print("Vector q: ", q)

data.to(device)

num_nodes = data.num_nodes
Q = torch.diag(q).to(device)
edge_index_2 = torch.ones([num_nodes,num_nodes],).nonzero().t().contiguous().to(device)
# edge_index_2 =  from_networkx(watts_strogatz_graph(num_nodes, 700, 0.1, seed=None)).edge_index.to(device)
with open(MASK_FILE, 'rb') as f:
    mask = pickle.load(f)
mask = mask.to(device)

adj_matrix = to_dense_adj(data.edge_index, max_num_nodes = num_nodes).squeeze(0)
print(adj_matrix.shape)
mask_matrix = to_dense_adj(mask, max_num_nodes = num_nodes).squeeze(0)
print(mask_matrix.shape)
masked_adj = adj_matrix*mask_matrix
print(masked_adj.shape)
edge_index = masked_adj.nonzero().t().contiguous().to(device)

x = get_x_init(num_nodes, d, 0, math.pi/2, 0, math.pi/2).to(device)



acc_glase_test = []
for iter in range(10):
    train_index = data.train_idx[:,iter]
    val_index = data.val_idx[:,iter]
    test_index = data.test_idx[:,iter]

    model = gLASE_e2e_Transformer(feature_dim, embedding_dim, hidden_dim, h_embedding_dim,output_dim, n_layers, dropout1, dropout2, gd_steps, num_heads=1)
    model.to(device)    

    ## Initialization
    for step in range(gd_steps):
        model.gd[step].lin1.weight.data = torch.nn.init.xavier_uniform_(model.gd[step].lin1.weight)*lr
        model.gd[step].lin2.weight.data = torch.nn.init.xavier_uniform_(model.gd[step].lin2.weight)*lr

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
        out, x_glase = model(data.x, x, edge_index, edge_index_2, Q, mask)
        # loss =  alpha*criterion(out[train_index], data.y[train_index].squeeze().to(device)) #+ (1-alpha)*torch.norm((x_glase@x_glase.T - to_dense_adj(data.edge_index).squeeze(0))*to_dense_adj(mask).squeeze(0))
        loss1 = criterion(out[train_index], data.y[train_index].squeeze().to(device))
        loss2 = torch.norm((x_glase@x_glase.T - to_dense_adj(edge_index, max_num_nodes=num_nodes).squeeze(0))*to_dense_adj(mask, max_num_nodes=num_nodes).squeeze(0))
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
        out, x_glase = model(data.x, x, edge_index, edge_index_2, Q, mask)
        loss =  alpha*criterion(out[train_index], data.y[train_index].squeeze().to(device)) #+ (1-alpha)*torch.norm((x_glase@x_glase.T - to_dense_adj(data.edge_index).squeeze(0))*to_dense_adj(mask).squeeze(0))

            
        # Calculate accuracy
        _, predicted_labels = torch.max(out[val_index].squeeze(),1)
        total_val_correct = (predicted_labels.squeeze() == data.y[val_index].squeeze().to(device)).sum().item()
        total_val_samples = len(val_index)
        valid_acc = total_val_correct / total_val_samples
        total_val_loss = loss.item()            
        
        ## Test
        loss =  alpha*criterion(out[train_index], data.y[train_index].squeeze().to(device)) #+ (1-alpha)*torch.norm((x_glase@x_glase.T - to_dense_adj(data.edge_index).squeeze(0))*to_dense_adj(mask).squeeze(0))

            
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
    loss =  alpha*criterion(out[train_index], data.y[train_index].squeeze().to(device)) #+ (1-alpha)*torch.norm((x_glase@x_glase.T - to_dense_adj(data.edge_index).squeeze(0))*to_dense_adj(mask).squeeze(0))


    # Calculate accuracy
    _, predicted_labels = torch.max(out[test_index].squeeze(),1)
    total_test_correct = (predicted_labels.squeeze() == data.y[test_index].squeeze().to(device)).sum().item()
    total_test_samples = len(test_index)
    test_acc = total_test_correct / total_test_samples
    total_test_loss = loss.item()      

    print('Best epoch: ', best_epoch)
    print(f'Test accuracy:  {100 * test_acc:.2f}')
    acc_glase_test.append(test_acc)
    
    with open(E2E_RESULTS, 'wb') as f:
        pickle.dump(acc_glase_test, f)
    
print(f'{np.array(acc_glase_test).mean()*100} +/- {np.array(acc_glase_test).std()*100}')


    