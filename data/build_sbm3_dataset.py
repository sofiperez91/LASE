import sys
sys.path.append("../")

import torch
import pickle
from torch_geometric.utils import stochastic_blockmodel_graph
from torch_geometric.data import Data

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



d = 3
num_nodes = 240
n = [120, 80, 40]
TRAIN_DATA_FILE='./sbm3_train_2.pkl'
VAL_DATA_FILE='./sbm3_val_2.pkl'
p = [
     [0.9, 0.2, 0.1],
     [0.2, 0.6, 0.2],
     [0.1, 0.2, 0.7]
]


df_train = []
df_val = []

for j in range(1000):
    x = torch.rand((num_nodes, d))
    edge_index = stochastic_blockmodel_graph(n, p).to(device)
    data = Data(x = x, edge_index = edge_index)
    if j < 800:
        df_train.append(data)
    else:
        df_val.append(data)

with open(TRAIN_DATA_FILE, 'wb') as f:
    pickle.dump(df_train, f)
with open(VAL_DATA_FILE, 'wb') as f:
    pickle.dump(df_val, f)