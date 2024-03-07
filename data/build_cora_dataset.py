import sys
sys.path.append("../")

import torch
import numpy as np
import random
from torch_geometric.utils import to_dense_adj, erdos_renyi_graph
from torch_geometric.data import Data
import pickle
from torch_geometric.utils import dropout_node
from torch_geometric.datasets import Planetoid
import random



Q_FILE = 'cora_q.pkl'
SUBGRAPH_TRAIN_FILE = './cora_train_subgraphs.pkl'
SUBGRAPH_VAL_FILE = './cora_train_subgraphs.pkl'
DATASET_FILE = './cora_dataset.pkl'
MASK_08_FILE = './cora_mask_M08.pkl'
MASK_06_FILE = './cora_mask_M06.pkl'
MASK_04_FILE = './cora_mask_M04.pkl'
MASK_02_FILE = './cora_mask_M02.pkl'

# LOAD ORIGINAL GRAPH
dataset = Planetoid(root='./dataset/Cora', name='Cora', split='public', transform=None)
data = dataset[0]

# # CHOOSE d
# d = torch.unique(data.y).shape[0] 
# print("Embedding dimension: ", d) 

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

# print("Vector q: ", q)


# # CREATE SUBGRAPHS
# train_data = []
# val_data = []
# for i in range(1000):
#     # print(i)
#     sub_edge_index, _, _ = dropout_node(data.edge_index, p=0.80)
#     adj_matrix = to_dense_adj(sub_edge_index).squeeze(0)
#     non_zero_rows = (adj_matrix.sum(dim=1) != 0)
#     adj_matrix = adj_matrix[non_zero_rows]
#     adj_matrix = adj_matrix[:, non_zero_rows]
#     sub_edge_index = adj_matrix.nonzero().t().contiguous()  
#     n_nodes = adj_matrix.shape[0]
#     x = torch.rand((n_nodes, d))
#     edge_index_2 = torch.ones([n_nodes,n_nodes],).nonzero().t().contiguous()
#     if i < 800:
#         train_data.append(Data(x=x, edge_index=sub_edge_index, edge_index_2=edge_index_2, num_nodes=n_nodes))
#     else:
#         val_data.append(Data(x=x, edge_index=sub_edge_index, edge_index_2=edge_index_2, num_nodes=n_nodes))
        
        
# with open(SUBGRAPH_TRAIN_FILE, 'wb') as f:
#     pickle.dump(train_data, f)
# with open(SUBGRAPH_VAL_FILE, 'wb') as f:
#     pickle.dump(val_data, f)
# with open(Q_FILE, 'wb') as f:
#     pickle.dump(q, f)
    

# # TRAIN - VAL - TEST split
num_nodes = data.y.shape[0]
# num_train = int(0.6*num_nodes)
# num_val = int(0.2*num_nodes)
# num_test = num_nodes - num_train - num_val

# train_idx=torch.zeros(num_train, 10, dtype=torch.int64)
# val_idx=torch.zeros(num_val, 10, dtype=torch.int64)
# test_idx=torch.zeros(num_test, 10, dtype=torch.int64)

# for iter in range(10):
#     torch.manual_seed(iter)
#     node_indexes = torch.randperm(num_nodes)
#     train_idx[:,iter] = node_indexes[:num_train]
#     val_idx[:, iter] =  node_indexes[num_train:num_train+num_val]
#     test_idx[:, iter] = node_indexes[num_train+num_val:]


# data_split = Data(x=data.x, edge_index=data.edge_index, y = data.y, train_idx=train_idx, val_idx=val_idx, test_idx=test_idx, num_nodes=num_nodes)

# print(data_split)
        
# with open(DATASET_FILE, 'wb') as f:
#     pickle.dump(data_split, f)


# MASKS
torch.manual_seed(42)
random.seed(42)
selected_nodes = random.sample(range(num_nodes), int(num_nodes*0.3))

M08 = torch.ones([num_nodes,num_nodes]).squeeze(0)
for i in selected_nodes:
    votos = (torch.rand(1, num_nodes) < 0.8).int()
    M08[i,:] = votos
    M08[:,i] = votos
    
M06 = torch.ones([num_nodes,num_nodes]).squeeze(0)
for i in selected_nodes:
    votos = (torch.rand(1, num_nodes) < 0.6).int()
    M06[i,:] = votos
    M06[:,i] = votos
    
M04 = torch.ones([num_nodes,num_nodes]).squeeze(0)
for i in selected_nodes:
    votos = (torch.rand(1, num_nodes) < 0.4).int()
    M04[i,:] = votos
    M04[:,i] = votos

M02 = torch.ones([num_nodes,num_nodes]).squeeze(0)
for i in selected_nodes:
    votos = (torch.rand(1, num_nodes) < 0.2).int()
    M02[i,:] = votos
    M02[:,i] = votos
    




# mask = (torch.ones([num_nodes,num_nodes],)-torch.eye(num_nodes)).nonzero().t().contiguous()
# with open('./amazon_mask_FULL.pkl', 'wb') as f:
#     pickle.dump(mask,f)

# ER08 = erdos_renyi_graph(num_nodes, 0.8, directed=False)
# ER06 = erdos_renyi_graph(num_nodes, 0.6, directed=False)
# ER04 = erdos_renyi_graph(num_nodes, 0.4, directed=False)
# ER02 = erdos_renyi_graph(num_nodes, 0.2, directed=False)

with open(MASK_08_FILE, 'wb') as f:
    pickle.dump(M08.nonzero().t().contiguous(), f)
    
with open(MASK_06_FILE, 'wb') as f:
    pickle.dump(M06.nonzero().t().contiguous(), f)
    
with open(MASK_04_FILE, 'wb') as f:
    pickle.dump(M04.nonzero().t().contiguous(), f)
    
with open(MASK_02_FILE, 'wb') as f:
    pickle.dump(M02.nonzero().t().contiguous(), f)
    