import sys
sys.path.append("../")

import torch
from torch_geometric.utils import stochastic_blockmodel_graph, to_dense_adj
from torch_geometric.data import Data
import pickle
from torch_geometric.utils import dropout_node

ORIGINAL_GRAPH = './bipartite_unbalanced_positive_original_graph_095.pkl'
TRAIN_SUBGRAPH = './bipartite_unbalanced_positive_train_subgraphs_095.pkl'
VAL_SUBGRAPH = './bipartite_unbalanced_positive_val_subgraphs_095.pkl'


# BUILD ORIGINAL GRAPH

## BIPARTITE
d = 4 
num_nodes_P = 250
num_nodes_L = 1000
P1 = torch.zeros((num_nodes_P,num_nodes_P))

P1_L1 = (torch.rand(num_nodes_P, num_nodes_L) < 0.9).int()
P1_L2 = (torch.rand(num_nodes_P, num_nodes_L) < 0.1).int()

P2 = torch.zeros((num_nodes_P,num_nodes_P))
P2_L1 = (torch.rand(num_nodes_P, num_nodes_L) < 0.1).int()
P2_L2 = (torch.rand(num_nodes_P, num_nodes_L) < 0.9).int()

L1 = torch.zeros((num_nodes_L,num_nodes_L))
L2 = torch.zeros((num_nodes_L,num_nodes_L))

F1 = torch.cat((P1, P2, P1_L1, P1_L2), dim=1) 
F2 = torch.cat((P2, P1, P2_L1, P2_L2), dim=1)
F3 = torch.cat((P1_L1.T,P2_L1.T,L1,L2), dim=1)
F4 = torch.cat((P1_L2.T,P2_L2.T,L1,L2), dim=1)

A = torch.cat((F1,F2,F3,F4), dim=0)

num_nodes = num_nodes_L*2 + num_nodes_P*2

# d = 3
# num_nodes = 12000
# n = [6000, 4000, 2000]
# p = [
#      [0.9, 0.2, 0.1],
#      [0.2, 0.6, 0.2],
#      [0.1, 0.2, 0.7]
# ]

# p = [
#      [0.6, 0.9, 0.8],
#      [0.9, 0.3, 0.7],
#      [0.8, 0.7, 0.5]
# ]

edge_index = A.nonzero().t().contiguous()

with open(ORIGINAL_GRAPH, 'wb') as f:
    pickle.dump(Data(edge_index=edge_index, num_nodes=num_nodes), f)

# CREATE SUBGRAPHS
train_data = []
val_data = []
for i in range(1000):
    # print(i)
    sub_edge_index, _, _ = dropout_node(edge_index, p=0.95)
    adj_matrix = to_dense_adj(sub_edge_index).squeeze(0)
    non_zero_rows = (adj_matrix.sum(dim=1) != 0)
    adj_matrix = adj_matrix[non_zero_rows]
    adj_matrix = adj_matrix[:, non_zero_rows]
    sub_edge_index = adj_matrix.nonzero().t().contiguous()  
    n_nodes = adj_matrix.shape[0] 
    x = torch.rand((n_nodes, d))
    print(n_nodes)
    edge_index_2 = torch.ones([n_nodes,n_nodes],).nonzero().t().contiguous()
    mask = (torch.ones([n_nodes,n_nodes],)-torch.eye(n_nodes)).nonzero().t().contiguous()
    if i < 800:
        train_data.append(Data(x=x, edge_index=sub_edge_index, edge_index_2=edge_index_2, mask=mask, num_nodes=n_nodes))
    else:
        val_data.append(Data(x=x, edge_index=sub_edge_index, edge_index_2=edge_index_2, mask=mask, num_nodes=n_nodes))
        
with open(TRAIN_SUBGRAPH, 'wb') as f:
    pickle.dump(train_data, f)
    
with open(VAL_SUBGRAPH, 'wb') as f:
    pickle.dump(val_data, f)