import sys
sys.path.append("../")

import torch
import numpy as np
import pickle
from torch_geometric.utils import stochastic_blockmodel_graph, to_dense_adj
from torch_geometric.data import Data
from graspologic.embed import AdjacencySpectralEmbed 

# num_nodes = 150
# d = 3
device = 'cpu'
# TRAIN_DATA_FILE='./sbm3_sim_train.pkl'
# VAL_DATA_FILE='./sbm3_sim_val.pkl'

# n = [int(num_nodes/3), int(num_nodes/3), int(num_nodes/3)]

# p = [
#      [0.7, 0.1, 0.1],
#      [0.1, 0.7, 0.1], 
#      [0.1, 0.1, 0.7]
# ]



# d = 3
# num_nodes = 240
# n = [120, 80, 40]
# TRAIN_DATA_FILE='./sbm3_train_2.pkl'
# VAL_DATA_FILE='./sbm3_val_2.pkl'
# p = [
#      [0.9, 0.2, 0.1],
#      [0.2, 0.6, 0.2],
#      [0.1, 0.2, 0.7]
# ]


## DEFINE BIPARTITE GRAPH
TRAIN_DATA_FILE='./bipartite_d6_train.pkl'
VAL_DATA_FILE='./bipartite_d6_val.pkl'

d = 6
n_P1 = 50
n_P2 = 50
n_L1 = 100
n_L2 = 100
n_L3 = 30

P1_L1 = 0.9
P1_L2 = 0.01
P1_L3 = 0.2  
P2_L1 = 0.1
P2_L2 = 0.8
P2_L3 = 0.3


p = [
    [0, 0, P1_L1, P1_L2, P1_L3],
    [0, 0, P2_L1, P2_L2, P2_L3],
    [P1_L1, P2_L1, 0, 0, 0], 
    [P1_L2, P2_L2, 0, 0, 0], 
    [P1_L3, P2_L3, 0, 0, 0]
    ]

n = [n_P1, n_P2, n_L1, n_L2, n_L3]

# p = [
#     [0, 0, P1_L1, P1_L2],
#     [0, 0, P2_L1, P2_L2],
#     [P1_L1, P2_L1, 0, 0], 
#     [P1_L2, P2_L2, 0, 0]
#     ]

# n = [n_P1, n_P2, n_L1, n_L2]

num_nodes = np.sum(n)

edge_index = stochastic_blockmodel_graph(n, p).to(device)


## MASK

n_P1_np = 5
n_P2_np = 3
senadores_no_presentes = list(range(n_P1_np)) + list(range(n_P1,n_P1+n_P2_np))

mask = torch.ones([num_nodes,num_nodes]).squeeze(0)
for i in senadores_no_presentes:
    votos = (torch.rand(1, num_nodes) < 0.3).int()
    mask[i,:] = votos
    mask[:,i] = votos


## ASE 

adj_matrix = to_dense_adj(edge_index.to('cpu')).squeeze(0).numpy()
ase = AdjacencySpectralEmbed(n_components=d, diag_aug=True, algorithm='full')
masked_adj = adj_matrix*mask.numpy()
x_ase = ase.fit_transform(masked_adj)
x_ase = torch.from_numpy(x_ase)


df_train = []
df_val = []

for j in range(1000):
    edge_index = stochastic_blockmodel_graph(n, p).to(device)
    data = Data(x = x_ase, edge_index = edge_index, mask = mask, num_nodes=num_nodes)
    if j < 800:
        df_train.append(data)
    else:
        df_val.append(data)

with open(TRAIN_DATA_FILE, 'wb') as f:
    pickle.dump(df_train, f)
with open(VAL_DATA_FILE, 'wb') as f:
    pickle.dump(df_val, f)