import sys
sys.path.append("../")

import warnings
warnings.filterwarnings("ignore")

import torch
import pickle
import argparse
import numpy as np

from torch_geometric.data import Data
from torch_geometric.utils import stochastic_blockmodel_graph, to_dense_adj
from graspologic.embed import AdjacencySpectralEmbed 

from models.RDPG_GD import GRDPG_GD_Armijo
from models.GLASE_unshared_normalized import gLASE 
from models.SVD_truncate import align_Xs
from training.link_prediction import link_prediction_Transformer

parser = argparse.ArgumentParser(description='Dataset')
parser.add_argument('--mask_threshold', type=float, default=0.7)
parser.add_argument('--iter', type=int, default=10)

args = parser.parse_args()
mask_threshold = args.mask_threshold
iter = args.iter



torch.manual_seed(42)

d = 4

n_P1 = 100 # NUMERO DE SENADORES P1
n_P2 = 100 # NUMERO DE SENADORES P2
n_L1 = 200 # NUMERO DE LEYES P1
n_L2 = 200 # NUMERO DE LEYES P2
n_L3 = 60 # NUMERO DE LEYES NEUTRAS

P1_L1 = 0.8 ## Votos de senadores del partido 1 a leyes grupo 1
P1_L2 = 0.01 ## Votos de senadores del partido 1 a leyes grupo 2
P1_L3 = 0.2 ## Votos de senadores del partido 1 a leyes grupo 3
P2_L1 = 0.01 ## Votos de senadores del partido 2 a leyes grupo 1
P2_L2 = 0.8 ## Votos de senadores del partido 2 a leyes grupo 2
P2_L3 = 0.2 ## Votos de senadores del partido 2 a leyes grupo 3


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

n_P1_np = 30
n_P2_np = 24
senadores_no_presentes = list(range(n_P1_np)) + list(range(n_P1,n_P1+n_P2_np))


## MASK

mask = torch.ones([num_nodes,num_nodes]).squeeze(0)
for i in senadores_no_presentes:
    votos = (torch.rand(1, n_L1+n_L2+n_L3) < mask_threshold).int()
    mask[i, n_P1+n_P2:] = votos
    mask[n_P1+n_P2:,i] = votos
    
    
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
        
q = torch.Tensor(list_q)
Q=torch.diag(q)
print(Q)

## GD GRDPG 
edge_index = edge_index.to('cpu')
x_grdpg, cost, k  = GRDPG_GD_Armijo(x_ase, edge_index, Q, mask.nonzero().t().contiguous())
x_grdpg = x_grdpg.detach()
print("Iteraciones: ", k)
print("Loss: ", torch.norm((x_grdpg@Q@x_grdpg.T - to_dense_adj(edge_index).squeeze(0))*to_dense_adj(mask.nonzero().t().contiguous()).squeeze(0)))


## GLASE
gd_steps = 20
lr = 1e-2
device = 'cuda'
model = gLASE(d,d, gd_steps)
model.init_lase(lr)
model.to(device)


epochs = 500

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


x_grdpg = align_Xs(x_grdpg, x_ase)
x_glase = align_Xs(x_glase, x_ase)

torch.manual_seed(42)
random_features=torch.rand([num_nodes, 5])
masked_edge_index = masked_adj.nonzero().t().contiguous()
data = Data(x=random_features.float(), x_init = x_ase, x_ase=x_ase, x_glase=x_glase, x_grdpg=x_grdpg, edge_index=masked_edge_index)


acc_gcn_array = []
acc_ase_array = []
acc_grdpg_array = []
acc_glase_array = []
acc_glase_e2e_array = []


for i in range(iter):
    inverted_mask_matrix = (torch.ones([num_nodes,num_nodes]).squeeze(0) - mask.to('cpu'))
    model_1, model_2, model_3, model_4, model_5, acc_gcn, acc_ase, acc_grdpg, acc_glase, acc_glase_e2e = link_prediction_Transformer(edge_index, edge_index_2, Q, mask, inverted_mask_matrix, data, 5, 4)

    print(f'GCN acc: {acc_gcn:.4f}, ASE acc: {acc_ase:.4f}, GD-GRDPG acc: {acc_grdpg:.4f}, GLASE acc: {acc_glase:.4f}, GLASE E2E acc: {acc_glase_e2e:.4f}')
    
    acc_gcn_array.append(acc_gcn)
    acc_ase_array.append(acc_ase)
    acc_grdpg_array.append(acc_grdpg)
    acc_glase_array.append(acc_glase)
    acc_glase_e2e_array.append(acc_glase_e2e)


results = {"gcn": acc_gcn_array, 
 "ase": acc_ase_array,
 "grdpg": acc_grdpg_array,
 "glase": acc_glase_array,
 "glase_e2e": acc_glase_e2e_array
}

with open(f'../training/results/link_pred_senators_results_0{int(mask_threshold*10)}.pkl', 'wb') as f:
    pickle.dump(results, f)
