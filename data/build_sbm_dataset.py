import sys
sys.path.append("../")

import torch
import pickle
import numpy as np
import argparse
import json

from torch_geometric.utils import stochastic_blockmodel_graph, dropout_node, to_dense_adj
from torch_geometric.data import Data


parser = argparse.ArgumentParser(description='Dataset')
parser.add_argument('--dataset', type=str, default='sbm10_unbalanced')
args = parser.parse_args()
dataset = args.dataset

# Load the config file
with open('./data_config.json', 'r') as file:
    config = json.load(file)

d = config[dataset]['d']
n = config[dataset]['n']
p = config[dataset]['p']
num_nodes = np.sum(n)
total_samples = config[dataset]['total_samples']
train_samples = config[dataset]['train_samples']
mode = config[dataset]['mode']
df_train = []
df_val = []


if mode == 'simple':

    TRAIN_DATA_FILE=f'{dataset}_train.pkl'
    VAL_DATA_FILE=f'{dataset}_val.pkl'

    for j in range(total_samples):
        x = torch.rand((num_nodes, d))
        edge_index = stochastic_blockmodel_graph(n, p)
        edge_index_2 = torch.ones([num_nodes,num_nodes],).nonzero().t().contiguous()
        mask = (torch.ones([num_nodes,num_nodes],)-torch.eye(num_nodes)).nonzero().t().contiguous()
        data = Data(x = x, edge_index = edge_index,  edge_index_2=edge_index_2, mask=mask, num_nodes=num_nodes)
        if j < train_samples:
            df_train.append(data)
        else:
            df_val.append(data)
        if j % 100 == 0:
            print(j)

elif mode == 'subgraphs':

    dropout = config[dataset]['dropout']
    ORIGINAL_GRAPH = f'{dataset}_original_graph.pkl'
    TRAIN_DATA_FILE=f'{dataset}_0{dropout}_train.pkl'
    VAL_DATA_FILE=f'{dataset}_0{dropout}_val_.pkl'

    edge_index = stochastic_blockmodel_graph(n, p)
    with open(ORIGINAL_GRAPH, 'wb') as f:
        pickle.dump(Data(edge_index=edge_index, num_nodes=num_nodes), f)

    # CREATE SUBGRAPHS
    for j in range(total_samples):
        sub_edge_index, _, _ = dropout_node(edge_index, p=dropout/100)
        adj_matrix = to_dense_adj(sub_edge_index).squeeze(0)
        non_zero_rows = (adj_matrix.sum(dim=1) != 0)
        adj_matrix = adj_matrix[non_zero_rows]
        adj_matrix = adj_matrix[:, non_zero_rows]
        sub_edge_index = adj_matrix.nonzero().t().contiguous()  
        n_nodes = adj_matrix.shape[0] 
        x = torch.rand((n_nodes, d))
        edge_index_2 = torch.ones([n_nodes,n_nodes],).nonzero().t().contiguous()
        mask = (torch.ones([n_nodes,n_nodes],)-torch.eye(n_nodes)).nonzero().t().contiguous()
        if j < train_samples:
            df_train.append(Data(x=x, edge_index=sub_edge_index, edge_index_2=edge_index_2, mask=mask, num_nodes=n_nodes))
        else:
            df_val.append(Data(x=x, edge_index=sub_edge_index, edge_index_2=edge_index_2, mask=mask, num_nodes=n_nodes))
        if j % 100 == 0:
            print(j)
            print(n_nodes)

with open(TRAIN_DATA_FILE, 'wb') as f:
    pickle.dump(df_train, f)
    
with open(VAL_DATA_FILE, 'wb') as f:
    pickle.dump(df_val, f)