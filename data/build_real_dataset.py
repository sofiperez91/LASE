import sys
sys.path.append("../")

import torch
import argparse
import random
import math 
import pickle
from collections import Counter

from torch_geometric.utils import to_dense_adj, dropout_node
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid, Amazon, WikipediaNetwork, WebKB

from training.get_init import get_x_init


parser = argparse.ArgumentParser(description='Dataset')
parser.add_argument('--dataset', type=str, default='cora')
parser.add_argument('--d', type=int, default=6)
args = parser.parse_args()
dataset = args.dataset
d = args.d

Q_FILE = f'{dataset}_q.pkl'
SUBGRAPH_TRAIN_FILE = f'./{dataset}_train_subgraphs.pkl'
SUBGRAPH_VAL_FILE = f'./{dataset}_val_subgraphs.pkl'
DATASET_FILE = f'./{dataset}_dataset.pkl'
MASK_FULL_FILE = f'./{dataset}_mask_FULL.pkl'
MASK_08_FILE = f'./{dataset}_mask_M08.pkl'
MASK_06_FILE = f'./{dataset}_mask_M06.pkl'
MASK_04_FILE = f'./{dataset}_mask_M04.pkl'
MASK_02_FILE = f'./{dataset}_mask_M02.pkl'

# LOAD ORIGINAL GRAPH
if dataset == 'cora':
    dataset = Planetoid(root='./dataset/Cora', name='Cora', split='public', transform=None)
    dropout = 0.80
elif dataset == 'amazon':
    dataset = Amazon(root='./dataset/amazon', name = 'Photo')
    dropout = 0.90
elif dataset == 'chameleon':
    dataset = WikipediaNetwork(root='./dataset/Chameleon', name='chameleon')
elif dataset == 'squirrel':
    dataset = WikipediaNetwork(root='./dataset/Squirrel', name='squirrel')
elif dataset == 'crocodile':
    dataset = WikipediaNetwork(root='./dataset/Crocodile', name='crocodile')
elif dataset == 'cornell':
    dataset = WebKB(root='./dataset/Cornell', name='Cornell')
elif dataset == 'texas':
    dataset = WebKB(root='./dataset/Texas', name='Texas')
elif dataset == 'wisconsin':
    dataset = WebKB(root='./dataset/Wisconsin', name='Wisconsin')

    
data = dataset[0]
print("Embedding dimension: ", d) 


# TRAIN - VAL - TEST split
num_nodes = data.y.shape[0]
num_train = int(0.6*num_nodes)
num_val = int(0.2*num_nodes)
num_test = num_nodes - num_train - num_val

train_idx=torch.zeros(num_train, 10, dtype=torch.int64)
val_idx=torch.zeros(num_val, 10, dtype=torch.int64)
test_idx=torch.zeros(num_test, 10, dtype=torch.int64)

for iter in range(10):
    torch.manual_seed(iter)
    node_indexes = torch.randperm(num_nodes)
    train_idx[:,iter] = node_indexes[:num_train]
    val_idx[:, iter] =  node_indexes[num_train:num_train+num_val]
    test_idx[:, iter] = node_indexes[num_train+num_val:]

data_split = Data(x=data.x, edge_index=data.edge_index, y = data.y, train_idx=train_idx, val_idx=val_idx, test_idx=test_idx, num_nodes=num_nodes)
        
with open(DATASET_FILE, 'wb') as f:
    pickle.dump(data_split, f)


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

mask = (torch.ones([num_nodes,num_nodes],)-torch.eye(num_nodes)).nonzero().t().contiguous()

with open(MASK_FULL_FILE, 'wb') as f:
    pickle.dump(mask,f)

with open(MASK_08_FILE, 'wb') as f:
    pickle.dump(M08.nonzero().t().contiguous(), f)
    
with open(MASK_06_FILE, 'wb') as f:
    pickle.dump(M06.nonzero().t().contiguous(), f)
    
with open(MASK_04_FILE, 'wb') as f:
    pickle.dump(M04.nonzero().t().contiguous(), f)
    
with open(MASK_02_FILE, 'wb') as f:
    pickle.dump(M02.nonzero().t().contiguous(), f)
    

# CREATE SUBGRAPHS
train_data = []
val_data = []
q_array = []
for i in range(1000):

    sub_edge_index, _, _ = dropout_node(data.edge_index, p=dropout)
    adj_matrix = to_dense_adj(sub_edge_index).squeeze(0)
    non_zero_rows = (adj_matrix.sum(dim=1) != 0)
    adj_matrix = adj_matrix[non_zero_rows]
    adj_matrix = adj_matrix[:, non_zero_rows]

    ## MASKS 
    sub_mask_M08 = M08[sub_edge_index[0].unique(),:]
    sub_mask_M08 = sub_mask_M08[:,sub_edge_index[0].unique()]
    sub_mask_M08_edge = sub_mask_M08.nonzero().t().contiguous()  

    sub_mask_M06 = M06[sub_edge_index[0].unique(),:]
    sub_mask_M06 = sub_mask_M06[:,sub_edge_index[0].unique()]
    sub_mask_M06_edge = sub_mask_M06.nonzero().t().contiguous()  

    sub_mask_M04 = M04[sub_edge_index[0].unique(),:]
    sub_mask_M04 = sub_mask_M04[:,sub_edge_index[0].unique()]   
    sub_mask_M04_edge = sub_mask_M04.nonzero().t().contiguous()   

    sub_mask_M02 = M02[sub_edge_index[0].unique(),:]
    sub_mask_M02 = sub_mask_M02[:,sub_edge_index[0].unique()]
    sub_mask_M02_edge = sub_mask_M02.nonzero().t().contiguous()  

    sub_edge_index = adj_matrix.nonzero().t().contiguous()  
    n_nodes = adj_matrix.shape[0]

    x = torch.rand((n_nodes, d))
    x_init = get_x_init(n_nodes, d, 0, math.pi/2, 0, math.pi/2)

    edge_index_2 = torch.ones([n_nodes,n_nodes],).nonzero().t().contiguous()

    if i < 800:
        train_data.append(Data(x=x, 
                               x_init=x_init, 
                               edge_index=sub_edge_index, 
                               edge_index_2=edge_index_2, 
                               num_nodes=n_nodes, 
                               sub_mask_M08=sub_mask_M08_edge, 
                               sub_mask_M06=sub_mask_M06_edge, 
                               sub_mask_M04=sub_mask_M04_edge, 
                               sub_mask_M02=sub_mask_M02_edge))
        u, V = torch.linalg.eig(adj_matrix)
        list_q=[]
        for i in range(d):
            if u[i].numpy()>0:
                list_q.append(1)
            else:
                list_q.append(-1)
        list_q.sort(reverse=True)
        q_array.append(list_q)
    else:
        val_data.append(Data(x=x, 
                             x_init=x_init, 
                             edge_index=sub_edge_index, 
                             edge_index_2=edge_index_2, 
                             num_nodes=n_nodes, 
                             sub_mask_M08=sub_mask_M08_edge, 
                             sub_mask_M06=sub_mask_M06_edge, 
                             sub_mask_M04=sub_mask_M04_edge, 
                             sub_mask_M02=sub_mask_M02_edge))
        
with open(SUBGRAPH_TRAIN_FILE, 'wb') as f:
    pickle.dump(train_data, f)
with open(SUBGRAPH_VAL_FILE, 'wb') as f:
    pickle.dump(val_data, f)


# Calculate q
tuples = [tuple(lst) for lst in q_array]
count = Counter(tuples)
most_common_element, max_frequency = count.most_common(1)[0]

q = torch.tensor(most_common_element).float()

with open(Q_FILE, 'wb') as f:
    pickle.dump(q, f)

