import sys
sys.path.append("../")

import warnings
warnings.filterwarnings("ignore")

import torch
import random
import csv
from typing import List
from torch_geometric.utils import to_dense_adj
from torch_geometric.data import Data

from graspologic.embed import AdjacencySpectralEmbed  

from models.onu_fuctions import load_un_dataset, create_un_graphs, process_un_graph_2
from models.RDPG_GD import GRDPG_GD_Armijo
from models.GLASE_unshared_normalized import gLASE 
from training.link_prediction import link_prediction, link_prediction_Transformer, link_prediction_GraphTransformer


def generate_embeddings(adj_matrix, mask, d):

    num_nodes = adj_matrix.shape[0]
    edge_index = torch.tensor(adj_matrix).nonzero().t().contiguous()
    
    ## Calculate Embeddings
    ## ASE 
    adj_matrix = to_dense_adj(edge_index.to('cpu'), max_num_nodes=num_nodes).squeeze(0)
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

    x_grdpg, cost, k  = GRDPG_GD_Armijo(x_ase, edge_index, Q, mask.nonzero().t().contiguous())
    x_grdpg = x_grdpg.detach()


    gd_steps = 20
    lr = 1e-2
    device = 'cuda'
    model = gLASE(d,d, gd_steps)
    model.init_lase(lr)
    model.to(device)

    epochs = 300
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
        loss = torch.norm((out@Q@out.T - to_dense_adj(edge_index, max_num_nodes=num_nodes).squeeze(0))*mask)
        loss.backward() 
        optimizer.step() 
            
    loss = torch.norm((out@Q@out.T - to_dense_adj(edge_index, max_num_nodes=num_nodes).squeeze(0))*mask)

    x_glase = out.detach().to('cpu')
    x_ase = x_ase.to('cpu')

    masked_edge_index = masked_adj.nonzero().t().contiguous()
    
    return x_ase, x_grdpg, x_glase, masked_edge_index, edge_index_2, Q


def build_onu_dataset(year: int = 2018, d: int = 4, unknown_countries: List[str] = ['FRA', 'SUD'], mask_threshold: float = 0.7, random_features: bool = False):
    votes_df = load_un_dataset('../data/UNVotes-1.csv', unknown_votes=True)
    
    all_graphs = create_un_graphs(votes_df[votes_df.year==year])
    
    adj_matrix, country_indexes, res_indexes, unknown_edges, features, mask_nodes, mask, selected_resolutions, inverted_mask_matrix, mask_unknown = process_un_graph_2(all_graphs, mask_countries=unknown_countries, mask_threshold=mask_threshold) 
    
    num_nodes = adj_matrix.shape[0]
    edge_index = torch.tensor(adj_matrix).nonzero().t().contiguous()
    
    x_ase, x_grdpg, x_glase, masked_edge_index, edge_index_2, Q = generate_embeddings(adj_matrix, mask, d)
    Q = Q.to('cuda')
    edge_index_2 = edge_index_2.to('cuda')
    mask = mask.to('cuda')
    
    if random_features:
        torch.manual_seed(42)
        random_features=torch.rand([num_nodes, 12])
        data = Data(x=random_features.float(), x_init=x_ase, x_ase=x_ase, x_glase=x_glase, x_grdpg=x_grdpg, edge_index=masked_edge_index)
    else:
        data = Data(x=features.float(), x_init=x_ase, x_ase=x_ase, x_glase=x_glase, x_grdpg=x_grdpg, edge_index=masked_edge_index)

    return data, edge_index, edge_index_2, Q, mask, inverted_mask_matrix, selected_resolutions


def link_prediction_onu(year: int = 2018, d: int = 4, unknown_countries: List[str] = ['FRA', 'SUD'], mask_threshold: float = 0.7, random_features: bool = False, iter: int =1, with_e2e: bool = False):

    data, edge_index, edge_index_2, Q, mask, inverted_mask_matrix, selected_resolutions = build_onu_dataset(year, d, unknown_countries, mask_threshold, random_features)

    acc_gcn_array = []
    acc_ase_array = []
    acc_grdpg_array = []
    acc_glase_array = []
    acc_glase_e2e_array = []
        
    for i in range(iter):
        if with_e2e:
            model_1, model_2, model_3, model_4, model_5, acc_gcn, acc_ase, acc_grdpg, acc_glase, acc_glase_e2e = link_prediction(edge_index, edge_index_2, Q, mask, inverted_mask_matrix, data, 12, d, 5, with_e2e=with_e2e)
            acc_gcn_array.append(acc_gcn.numpy().item())
            acc_ase_array.append(acc_ase.numpy().item())
            acc_grdpg_array.append(acc_grdpg.numpy().item())
            acc_glase_array.append(acc_glase.numpy().item())
            acc_glase_e2e_array.append(acc_glase_e2e.numpy().item())
        else:
            model_1, model_2, model_3, model_4, acc_gcn, acc_ase, acc_grdpg, acc_glase = link_prediction(edge_index, edge_index_2, Q, mask, inverted_mask_matrix, data, 12, d, 5, with_e2e=with_e2e)
            acc_gcn_array.append(acc_gcn.numpy().item())
            acc_ase_array.append(acc_ase.numpy().item())
            acc_grdpg_array.append(acc_grdpg.numpy().item())
            acc_glase_array.append(acc_glase.numpy().item())            
            
    if with_e2e:
        return acc_gcn_array, acc_ase_array, acc_grdpg_array, acc_glase_array, acc_glase_e2e_array, selected_resolutions
    else:
        return acc_gcn_array, acc_ase_array, acc_grdpg_array, acc_glase_array, selected_resolutions
        

def link_prediction_onu_Transformer(year: int = 2018, d: int = 4, unknown_countries: List[str] = ['FRA', 'SUD'], mask_threshold: float = 0.7, random_features: bool = False, iter: int =1, with_e2e: bool = False):

    data, edge_index, edge_index_2, Q, mask, inverted_mask_matrix, selected_resolutions = build_onu_dataset(year, d, unknown_countries, mask_threshold, random_features)

    acc_gcn_array = []
    acc_ase_array = []
    acc_grdpg_array = []
    acc_glase_array = []
    acc_glase_e2e_array = []
    
    for i in range(iter):
        if with_e2e:
            model_1, model_2, model_3, model_4, model_5, acc_gcn, acc_ase, acc_grdpg, acc_glase, acc_glase_e2e = link_prediction_Transformer(edge_index, edge_index_2, Q, mask, inverted_mask_matrix, data, 12, d, 5, with_e2e=with_e2e)
            acc_gcn_array.append(acc_gcn.numpy().item())
            acc_ase_array.append(acc_ase.numpy().item())
            acc_grdpg_array.append(acc_grdpg.numpy().item())
            acc_glase_array.append(acc_glase.numpy().item())
            acc_glase_e2e_array.append(acc_glase_e2e.numpy().item())
        else:
            model_1, model_2, model_3, model_4, acc_gcn, acc_ase, acc_grdpg, acc_glase = link_prediction_Transformer(edge_index, edge_index_2, Q, mask, inverted_mask_matrix, data, 12, d, 5, with_e2e=with_e2e)
            acc_gcn_array.append(acc_gcn.numpy().item())
            acc_ase_array.append(acc_ase.numpy().item())
            acc_grdpg_array.append(acc_grdpg.numpy().item())
            acc_glase_array.append(acc_glase.numpy().item())            
            
    if with_e2e:
        return acc_gcn_array, acc_ase_array, acc_grdpg_array, acc_glase_array, acc_glase_e2e_array, selected_resolutions
    else:
        return acc_gcn_array, acc_ase_array, acc_grdpg_array, acc_glase_array, selected_resolutions


def link_prediction_onu_GraphTransformer(year: int = 2018, d: int = 4, unknown_countries: List[str] = ['FRA', 'SUD'], mask_threshold: float = 0.7, random_features: bool = False, iter: int =1, with_e2e: bool = False):
    
    data, edge_index, edge_index_2, Q, mask, inverted_mask_matrix, selected_resolutions = build_onu_dataset(year, d, unknown_countries, mask_threshold, random_features)

    acc_gcn_array = []
    acc_ase_array = []
    acc_grdpg_array = []
    acc_glase_array = []
    acc_glase_e2e_array = []
    
    for i in range(iter):
        if with_e2e:
            model_1, model_2, model_3, model_4, model_5, acc_gcn, acc_ase, acc_grdpg, acc_glase, acc_glase_e2e = link_prediction_GraphTransformer(edge_index, edge_index_2, Q, mask, inverted_mask_matrix, data, 12, d, 5, with_e2e=with_e2e, epochs = 101, output_dim= 32, pe_out_dim= 8, n_layers= 3, dropout=0.5, num_heads=4, batch_norm=False, lr=0.001)
            acc_gcn_array.append(acc_gcn.numpy().item())
            acc_ase_array.append(acc_ase.numpy().item())
            acc_grdpg_array.append(acc_grdpg.numpy().item())
            acc_glase_array.append(acc_glase.numpy().item())
            acc_glase_e2e_array.append(acc_glase_e2e.numpy().item())
        else:
            model_1, model_2, model_3, model_4, acc_gcn, acc_ase, acc_grdpg, acc_glase = link_prediction_GraphTransformer(edge_index, edge_index_2, Q, mask, inverted_mask_matrix, data, 12, d, 5, with_e2e=with_e2e)
            acc_gcn_array.append(acc_gcn.numpy().item())
            acc_ase_array.append(acc_ase.numpy().item())
            acc_grdpg_array.append(acc_grdpg.numpy().item())
            acc_glase_array.append(acc_glase.numpy().item())            
            
    if with_e2e:
        return acc_gcn_array, acc_ase_array, acc_grdpg_array, acc_glase_array, acc_glase_e2e_array, selected_resolutions
    else:
        return acc_gcn_array, acc_ase_array, acc_grdpg_array, acc_glase_array, selected_resolutions


votes_df = load_un_dataset('../data/UNVotes-1.csv', unknown_votes=True)

## ORIGINAL FEATURES + RANDOM MISSING COUNTRIES

for year in votes_df.year.unique():
    countries = random.sample(votes_df[votes_df.year == year]['Country'].unique().tolist(), 6)
    acc_gcn_array, acc_ase_array, acc_grdpg_array, acc_glase_array, acc_glase_e2e_array, selected_resolutions = link_prediction_onu_Transformer(year = year, d=4, unknown_countries = countries, mask_threshold = 0.3, random_features=False, iter=10, with_e2e=True)    

    with open('./results/onu/onu_original_feat_results.csv', 'w', newline='') as file:
        writer = csv.writer(file, delimiter=';')
        writer.writerow([year, countries, selected_resolutions, acc_gcn_array, acc_ase_array, acc_grdpg_array, acc_glase_array, acc_glase_e2e_array])


## ORIGINAL FEATURES + SELECTED MISSING COUNTRIES

for year in votes_df.year.unique():
    countries = ['ISR', 'GRB', 'NDL', 'CUB', 'TUR', 'VNM']
    acc_gcn_array, acc_ase_array, acc_grdpg_array, acc_glase_array, acc_glase_e2e_array, selected_resolutions = link_prediction_onu_Transformer(year = year, d=4, unknown_countries = countries, mask_threshold = 0.3, random_features=False, iter=10, with_e2e=True)    

    with open('./results/onu/onu_original_feat_selected_results.csv', 'w', newline='') as file:
        writer = csv.writer(file, delimiter=';')
        writer.writerow([year, countries, selected_resolutions, acc_gcn_array, acc_ase_array, acc_grdpg_array, acc_glase_array, acc_glase_e2e_array])


## RANDOM FEATURES + RANDOM MISSING COUNTRIES

for year in votes_df.year.unique()[:1]:
    countries = random.sample(votes_df[votes_df.year == year]['Country'].unique().tolist(), 6)
    acc_gcn_array, acc_ase_array, acc_grdpg_array, acc_glase_array, acc_glase_e2e_array, selected_resolutions = link_prediction_onu_Transformer(year = year, d=4, unknown_countries = countries, mask_threshold = 0.3, random_features=True, iter=10, with_e2e=True)    

    # Open a CSV file to write to
    with open('./results/onu/onu_random_feat_results.csv', 'w', newline='') as file:
        writer = csv.writer(file, delimiter=';')
        writer.writerow([year, countries, selected_resolutions, acc_gcn_array, acc_ase_array, acc_grdpg_array, acc_glase_array, acc_glase_e2e_array])


## RANDOM FEATURES + SELECTED MISSING COUNTRIES

for year in votes_df.year.unique():
    countries = ['ISR', 'GRB', 'NDL', 'CUB', 'TUR', 'VNM']
    acc_gcn_array, acc_ase_array, acc_grdpg_array, acc_glase_array, acc_glase_e2e_array, selected_resolutions = link_prediction_onu_Transformer(year = year, d=4, unknown_countries = countries, mask_threshold = 0.3, random_features=True, iter=10, with_e2e=True)    

    # Open a CSV file to write to
    with open('./results/onu/onu_random_feat_selected_results.csv', 'w', newline='') as file:
        writer = csv.writer(file, delimiter=';')
        writer.writerow([year, countries, selected_resolutions, acc_gcn_array, acc_ase_array, acc_grdpg_array, acc_glase_array, acc_glase_e2e_array])




