import sys
sys.path.append("../")

import warnings
warnings.filterwarnings("ignore")

import torch
import numpy as np
from torch_geometric.utils import stochastic_blockmodel_graph, to_dense_adj
from graspologic.embed import AdjacencySpectralEmbed 
from models.RDPG_GD import GRDPG_GD_Armijo
from models.GLASE_unshared_normalized import gLASE 
# from models.GLASE_unshared_normalized_v2 import gLASE_v2
import matplotlib.pyplot as plt
from torch_geometric.nn import GCNConv
from sklearn.metrics import roc_auc_score
from torch_geometric.utils import negative_sampling
from models.link_prediction import Net, train_link_prediction, eval_link_prediction
from training.run_link_prediction import link_prediction

torch.manual_seed(42)

d = 4

n_P1 = 100 # NUMERO DE SENADORES P1
n_P2 = 100 # NUMERO DE SENADORES P2
n_L1 = 200 # NUMERO DE LEYES P1
n_L2 = 200 # NUMERO DE LEYES P2
n_L3 = 60 # NUMERO DE LEYES NEUTRAS

P1_L1 = 0.9 ## Votos de senadores del partido 1 a leyes grupo 1
P1_L2 = 0.01 ## Votos de senadores del partido 1 a leyes grupo 2
P1_L3 = 0.2 ## Votos de senadores del partido 1 a leyes grupo 3
P2_L1 = 0.1 ## Votos de senadores del partido 2 a leyes grupo 1
P2_L2 = 0.8 ## Votos de senadores del partido 2 a leyes grupo 2
P2_L3 = 0.3 ## Votos de senadores del partido 2 a leyes grupo 3


p = [
    [0, 0, P1_L1, P1_L2, P1_L3],
    [0, 0, P2_L1, P2_L2, P2_L3],
    [P1_L1, P2_L1, 0, 0, 0], 
    [P1_L2, P2_L2, 0, 0, 0], 
    [P1_L3, P2_L3, 0, 0, 0]
    ]

n = [n_P1, n_P2, n_L1, n_L2, n_L3]

num_nodes = np.sum(n)
edge_index = stochastic_blockmodel_graph(n, p)


## MASK
n_P1_np = 20
n_P2_np = 10
senadores_no_presentes = list(range(n_P1_np)) + list(range(n_P1,n_P1+n_P2_np))

mask = torch.ones([num_nodes,num_nodes]).squeeze(0)
for i in senadores_no_presentes:
    votos = (torch.rand(1, num_nodes) < 0.7).int()
    mask[i,:] = votos
    mask[:,i] = votos


## ASE 
adj_matrix = to_dense_adj(edge_index.to('cpu')).squeeze(0)
ase = AdjacencySpectralEmbed(n_components=d, diag_aug=True, algorithm='full')
masked_adj = adj_matrix*mask
x_ase = ase.fit_transform(masked_adj.numpy())
x_ase = torch.from_numpy(x_ase)

A = to_dense_adj(edge_index.to('cpu'), max_num_nodes=num_nodes).squeeze(0)

u, V = torch.linalg.eig(A)

list_q=[]
for i in range(d):
    if u[i].numpy()>0:
        list_q.append(1)
    else:
        list_q.append(-1)
        
# list_q.sort(reverse=True)
q = torch.Tensor(list_q)
Q=torch.diag(q)

print(Q)


torch.norm((x_ase@Q@x_ase.T - to_dense_adj(edge_index).squeeze(0))*mask)


x_grdpg, cost, k  = GRDPG_GD_Armijo(x_ase, edge_index, Q, mask.nonzero().t().contiguous())
x_grdpg = x_grdpg.detach()
print("Iteraciones: ", k)
print("Loss: ", torch.norm((x_grdpg@Q@x_grdpg.T - to_dense_adj(edge_index).squeeze(0))*to_dense_adj(mask.nonzero().t().contiguous()).squeeze(0)))


gd_steps = 20
lr = 1e-2
device = 'cuda'
model = gLASE(d,d, gd_steps)
model.to(device)


epochs = 400

## Initialization
for step in range(gd_steps):
    model.gd[step].lin1.weight.data = (torch.eye(d,d)*lr).to(device)#torch.nn.init.xavier_uniform_(model.gd[step].lin1.weight)*lr
    model.gd[step].lin2.weight.data = (torch.eye(d,d)*lr).to(device)#torch.nn.init.xavier_uniform_(model.gd[step].lin2.weight)*lr
    

optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# Define ATT mask
edge_index_2 = torch.ones([num_nodes,num_nodes],).nonzero().t().contiguous().to(device)
mask = mask.to(device)
x_ase = x_ase.to(device)
edge_index = edge_index.to(device)
Q = Q.to(device)

for epoch in range(epochs):
    # Train
    model.train()

    optimizer.zero_grad()
    out = model(x_ase, edge_index, edge_index_2, Q, mask.nonzero().t().contiguous())
    loss = torch.norm((out@Q@out.T - to_dense_adj(edge_index).squeeze(0))*mask)
    loss.backward() 
    optimizer.step() 

    if epoch % 100 ==0:
        print(loss)



loss = torch.norm((out@Q@out.T - to_dense_adj(edge_index).squeeze(0))*mask)
print(loss)
x_glase = out.detach().to('cpu')
x_ase = x_ase.to('cpu')

from models.SVD_truncate import align_Xs
x_grdpg = align_Xs(x_grdpg, x_ase)
x_glase = align_Xs(x_glase, x_ase)


torch.manual_seed(42)
random_features=torch.rand([num_nodes, 5])
print(random_features)

## Split Train, Val, Test
from torch_geometric.data import Data
import torch_geometric.transforms as T

device = 'cuda'

masked_edge_index = masked_adj.nonzero().t().contiguous()

data = Data(x=random_features.float(), x_ase=x_ase, x_glase=x_glase, x_grdpg=x_grdpg, edge_index=masked_edge_index)

transform = T.Compose([
    T.ToDevice(device),
    T.RandomLinkSplit(num_val=0.05, num_test=0.01, is_undirected=True,
                      add_negative_train_samples=False),
])


train_data, val_data, test_data = transform(data)

x_train = torch.concatenate((train_data.x, train_data.x_ase), axis=1)
x_val = torch.concatenate((val_data.x, val_data.x_ase), axis=1)
x_test = torch.concatenate((test_data.x, test_data.x_ase), axis=1)
    
inverted_mask_matrix = (torch.ones([num_nodes,num_nodes]).squeeze(0) - mask.to('cpu'))
for i in range(10):
    link_prediction(edge_index, mask, inverted_mask_matrix, data, 5, 4)