import torch
import random
import networkx as nx
from torch_geometric.utils import to_dense_adj
from graspologic.embed import AdjacencySpectralEmbed  

import sys
sys.path.append("../")
from models.onu_fuctions import load_un_dataset, create_un_graphs, process_un_graph, process_un_graph_2
from models.link_prediction import train_link_prediction, eval_link_prediction, train_link_prediction_GAT, train_link_prediction_Transformer, train_link_prediction_GraphTransformer
from models.glase_e2e_link_prediction import train_link_prediction_e2e, eval_link_prediction_e2e, train_link_prediction_GAT_e2e, train_link_prediction_Transformer_e2e, train_link_prediction_GraphTransformer_e2e
# from models.link_prediction import Net2
from models.RDPG_GD import GRDPG_GD_Armijo
from models.GLASE_unshared_normalized import gLASE 
from training.generate_embeddings import generate_embeddings
from torch_geometric.data import Data
import torch_geometric.transforms as T


import copy
from typing import List


def link_prediction(edge_index, edge_index_2, Q, mask, inverted_mask_matrix, data, feat_dim: int, d: int = 4, gd_steps: int = 20, with_e2e: bool = True):

    num_nodes = mask.shape[0]
    adj_matrix = to_dense_adj(edge_index.to('cpu'), max_num_nodes=num_nodes).squeeze(0)

    ## Split Train, Val, Test
    device = 'cuda'
    transform = T.Compose([
        T.ToDevice(device),
        T.RandomLinkSplit(num_val=0.05, num_test=0.01, is_undirected=True,
                        add_negative_train_samples=False),
    ])

    train_data, val_data, test_data = transform(data)
    data = data.to('cuda')
    
    ## GCN    
    x_train = train_data.x
    x_val = val_data.x
    x_test = test_data.x
    
    model_1 = train_link_prediction(x_train, x_val, x_test, train_data, val_data, test_data, input_dim=feat_dim)

    ## Predict on entire masked graph
    x_eval = data.x
    acc_gcn = eval_link_prediction(x_eval, data.edge_index, model_1, adj_matrix, inverted_mask_matrix)

    ## ASE 
    x_train = torch.concatenate((train_data.x, train_data.x_ase), axis=1)
    x_val = torch.concatenate((val_data.x, val_data.x_ase), axis=1)
    x_test = torch.concatenate((test_data.x, test_data.x_ase), axis=1)
    
    model_2 = train_link_prediction(x_train, x_val, x_test, train_data, val_data, test_data, input_dim=feat_dim+d)

    ## Predict on entire masked graph
    x_eval = torch.concatenate((data.x, data.x_ase), axis=1)
    acc_ase = eval_link_prediction(x_eval, data.edge_index, model_2, adj_matrix, inverted_mask_matrix)
    
    
    ## GD-GRDPG
    x_train = torch.concatenate((train_data.x, train_data.x_grdpg), axis=1)
    x_val = torch.concatenate((val_data.x, val_data.x_grdpg), axis=1)
    x_test = torch.concatenate((test_data.x, test_data.x_grdpg), axis=1)

    model_3 = train_link_prediction(x_train, x_val, x_test, train_data, val_data, test_data, input_dim=feat_dim+d)
    
    ## Predict on entire masked graph
    x_eval = torch.concatenate((data.x, data.x_grdpg), axis=1)
    acc_grdpg = eval_link_prediction(x_eval, data.edge_index, model_3, adj_matrix, inverted_mask_matrix)
    
    
    ## GLASE
    x_train = torch.concatenate((train_data.x, train_data.x_glase), axis=1)
    x_val = torch.concatenate((val_data.x, val_data.x_glase), axis=1)
    x_test = torch.concatenate((test_data.x, test_data.x_glase), axis=1)
    
    model_4 = train_link_prediction(x_train, x_val, x_test, train_data, val_data, test_data, input_dim=feat_dim+d)       

    ## Predict on entire masked graph
    x_eval = torch.concatenate((data.x, data.x_glase), axis=1)
    acc_glase = eval_link_prediction(x_eval, data.edge_index, model_4, adj_matrix, inverted_mask_matrix)
    
    if with_e2e:
        ## GLASE E2E
        
        model_5 = train_link_prediction_e2e(train_data, val_data, test_data, edge_index_2, Q, mask, feat_dim, d, gd_steps)
        acc_glase_e2e = eval_link_prediction_e2e(data.x, data.x_init, data.edge_index, edge_index_2, Q, mask, model_5, adj_matrix, inverted_mask_matrix)
        
        
        return model_1, model_2, model_3, model_4, model_5, acc_gcn, acc_ase, acc_grdpg, acc_glase, acc_glase_e2e
    else: 
        return model_1, model_2, model_3, model_4, acc_gcn, acc_ase, acc_grdpg, acc_glase
    

def link_prediction_GAT(edge_index, edge_index_2, Q, mask, inverted_mask_matrix, data, feat_dim: int, d: int = 4, gd_steps: int = 20, with_e2e: bool = True):

    num_nodes = mask.shape[0]
    adj_matrix = to_dense_adj(edge_index.to('cpu'), max_num_nodes=num_nodes).squeeze(0)

    ## Split Train, Val, Test
    device = 'cuda'
    transform = T.Compose([
        T.ToDevice(device),
        T.RandomLinkSplit(num_val=0.05, num_test=0.01, is_undirected=True,
                        add_negative_train_samples=False),
    ])

    train_data, val_data, test_data = transform(data)
    
    ## GCN    
    x_train = train_data.x
    x_val = val_data.x
    x_test = test_data.x
    
    model_1 = train_link_prediction_GAT(x_train, x_val, x_test, train_data, val_data, test_data, input_dim=feat_dim)

    ## Predict on entire masked graph
    x_eval = data.x
    acc_gcn = eval_link_prediction(x_eval, data.edge_index, model_1, adj_matrix, inverted_mask_matrix)

    ## ASE 
    x_train = torch.concatenate((train_data.x, train_data.x_ase), axis=1)
    x_val = torch.concatenate((val_data.x, val_data.x_ase), axis=1)
    x_test = torch.concatenate((test_data.x, test_data.x_ase), axis=1)
    
    model_2 = train_link_prediction_GAT(x_train, x_val, x_test, train_data, val_data, test_data, input_dim=feat_dim+d)

    ## Predict on entire masked graph
    x_eval = torch.concatenate((data.x, data.x_ase), axis=1)
    acc_ase = eval_link_prediction(x_eval, data.edge_index, model_2, adj_matrix, inverted_mask_matrix)
    
    
    ## GD-GRDPG
    x_train = torch.concatenate((train_data.x, train_data.x_grdpg), axis=1)
    x_val = torch.concatenate((val_data.x, val_data.x_grdpg), axis=1)
    x_test = torch.concatenate((test_data.x, test_data.x_grdpg), axis=1)

    model_3 = train_link_prediction_GAT(x_train, x_val, x_test, train_data, val_data, test_data, input_dim=feat_dim+d)
    
    ## Predict on entire masked graph
    x_eval = torch.concatenate((data.x, data.x_grdpg), axis=1)
    acc_grdpg = eval_link_prediction(x_eval, data.edge_index, model_3, adj_matrix, inverted_mask_matrix)
    
    
    ## GLASE
    x_train = torch.concatenate((train_data.x, train_data.x_glase), axis=1)
    x_val = torch.concatenate((val_data.x, val_data.x_glase), axis=1)
    x_test = torch.concatenate((test_data.x, test_data.x_glase), axis=1)
    
    model_4 = train_link_prediction_GAT(x_train, x_val, x_test, train_data, val_data, test_data, input_dim=feat_dim+d)       

    ## Predict on entire masked graph
    x_eval = torch.concatenate((data.x, data.x_glase), axis=1)
    acc_glase = eval_link_prediction(x_eval, data.edge_index, model_4, adj_matrix, inverted_mask_matrix)
    
    if with_e2e:
        ## GLASE E2E
        model_5 = train_link_prediction_GAT_e2e(train_data, val_data, test_data, edge_index_2, Q, mask, feat_dim, d, gd_steps)
        acc_glase_e2e = eval_link_prediction_e2e(data.x, data.x_init, data.edge_index, edge_index_2, Q, mask, model_5, adj_matrix, inverted_mask_matrix)
        
        
        return model_1, model_2, model_3, model_4, model_5, acc_gcn, acc_ase, acc_grdpg, acc_glase, acc_glase_e2e
    else: 
        return model_1, model_2, model_3, model_4, acc_gcn, acc_ase, acc_grdpg, acc_glase



def link_prediction_Transformer(edge_index, edge_index_2, Q, mask, inverted_mask_matrix, data, feat_dim: int, d: int = 4, gd_steps: int = 20, with_e2e: bool = True):

    num_nodes = mask.shape[0]
    adj_matrix = to_dense_adj(edge_index.to('cpu'), max_num_nodes=num_nodes).squeeze(0)

    ## Split Train, Val, Test
    device = 'cuda'
    transform = T.Compose([
        T.ToDevice(device),
        T.RandomLinkSplit(num_val=0.05, num_test=0.01, is_undirected=True,
                        add_negative_train_samples=False),
    ])

    train_data, val_data, test_data = transform(data)
    data = data.to('cuda')
    
    ## GCN    
    x_train = train_data.x
    x_val = val_data.x
    x_test = test_data.x
    
    model_1 = train_link_prediction_Transformer(x_train, x_val, x_test, train_data, val_data, test_data, input_dim=feat_dim)

    ## Predict on entire masked graph
    x_eval = data.x
    acc_gcn = eval_link_prediction(x_eval, data.edge_index, model_1, adj_matrix, inverted_mask_matrix)

    ## ASE 
    x_train = torch.concatenate((train_data.x, train_data.x_ase), axis=1)
    x_val = torch.concatenate((val_data.x, val_data.x_ase), axis=1)
    x_test = torch.concatenate((test_data.x, test_data.x_ase), axis=1)
    
    model_2 = train_link_prediction_Transformer(x_train, x_val, x_test, train_data, val_data, test_data, input_dim=feat_dim+d)

    ## Predict on entire masked graph
    x_eval = torch.concatenate((data.x, data.x_ase), axis=1)
    acc_ase = eval_link_prediction(x_eval, data.edge_index, model_2, adj_matrix, inverted_mask_matrix)
    
    
    ## GD-GRDPG
    x_train = torch.concatenate((train_data.x, train_data.x_grdpg), axis=1)
    x_val = torch.concatenate((val_data.x, val_data.x_grdpg), axis=1)
    x_test = torch.concatenate((test_data.x, test_data.x_grdpg), axis=1)

    model_3 = train_link_prediction_Transformer(x_train, x_val, x_test, train_data, val_data, test_data, input_dim=feat_dim+d)
    
    ## Predict on entire masked graph
    x_eval = torch.concatenate((data.x, data.x_grdpg), axis=1)
    acc_grdpg = eval_link_prediction(x_eval, data.edge_index, model_3, adj_matrix, inverted_mask_matrix)
    
    
    ## GLASE
    x_train = torch.concatenate((train_data.x, train_data.x_glase), axis=1)
    x_val = torch.concatenate((val_data.x, val_data.x_glase), axis=1)
    x_test = torch.concatenate((test_data.x, test_data.x_glase), axis=1)
    
    model_4 = train_link_prediction_Transformer(x_train, x_val, x_test, train_data, val_data, test_data, input_dim=feat_dim+d)       

    ## Predict on entire masked graph
    x_eval = torch.concatenate((data.x, data.x_glase), axis=1)
    acc_glase = eval_link_prediction(x_eval, data.edge_index, model_4, adj_matrix, inverted_mask_matrix)
    
    if with_e2e:
        ## GLASE E2E
        model_5 = train_link_prediction_Transformer_e2e(train_data, val_data, test_data, edge_index_2, Q, mask, feat_dim, d, gd_steps)
        acc_glase_e2e = eval_link_prediction_e2e(data.x, data.x_init, data.edge_index, edge_index_2, Q, mask, model_5, adj_matrix, inverted_mask_matrix)
        
        
        return model_1, model_2, model_3, model_4, model_5, acc_gcn, acc_ase, acc_grdpg, acc_glase, acc_glase_e2e
    else: 
        return model_1, model_2, model_3, model_4, acc_gcn, acc_ase, acc_grdpg, acc_glase
    

def link_prediction_GraphTransformer(edge_index, edge_index_2, Q, mask, inverted_mask_matrix, data, feat_dim: int, d: int = 4, gd_steps: int = 20, with_e2e: bool = True):

    num_nodes = mask.shape[0]
    adj_matrix = to_dense_adj(edge_index.to('cpu'), max_num_nodes=num_nodes).squeeze(0)

    ## Split Train, Val, Test
    device = 'cuda'
    transform = T.Compose([
        T.ToDevice(device),
        T.RandomLinkSplit(num_val=0.05, num_test=0.01, is_undirected=True,
                        add_negative_train_samples=False),
    ])

    train_data, val_data, test_data = transform(data)
    data = data.to('cuda')
    
    ## GCN    
    x_train = train_data.x
    x_val = val_data.x
    x_test = test_data.x
    
    model_1 = train_link_prediction(x_train, x_val, x_test, train_data, val_data, test_data, input_dim=feat_dim)

    ## Predict on entire masked graph
    x_eval = data.x
    acc_gcn = eval_link_prediction(x_eval, data.edge_index, model_1, adj_matrix, inverted_mask_matrix)

    ## ASE 
    # x_train = torch.concatenate((train_data.x, train_data.x_ase), axis=1)
    # x_val = torch.concatenate((val_data.x, val_data.x_ase), axis=1)
    # x_test = torch.concatenate((test_data.x, test_data.x_ase), axis=1)
    
    x_train = train_data.x, train_data.x_ase
    x_val = val_data.x, val_data.x_ase
    x_test = test_data.x, test_data.x_ase
    
    model_2 = train_link_prediction_GraphTransformer(x_train, x_val, x_test, train_data, val_data, test_data, input_dim=feat_dim, pe_dim=d)

    ## Predict on entire masked graph
    # x_eval = torch.concatenate((data.x, data.x_ase), axis=1)
    x_eval = data.x, data.x_ase
    acc_ase = eval_link_prediction(x_eval, data.edge_index, model_2, adj_matrix, inverted_mask_matrix)
    
    
    ## GD-GRDPG
    # x_train = torch.concatenate((train_data.x, train_data.x_grdpg), axis=1)
    # x_val = torch.concatenate((val_data.x, val_data.x_grdpg), axis=1)
    # x_test = torch.concatenate((test_data.x, test_data.x_grdpg), axis=1)
    x_train = train_data.x, train_data.x_grdpg
    x_val = val_data.x, val_data.x_grdpg
    x_test = test_data.x, test_data.x_grdpg
    

    model_3 = train_link_prediction_GraphTransformer(x_train, x_val, x_test, train_data, val_data, test_data, input_dim=feat_dim, pe_dim=d)
    
    ## Predict on entire masked graph
    # x_eval = torch.concatenate((data.x, data.x_grdpg), axis=1)
    x_eval = data.x, data.x_grdpg
    acc_grdpg = eval_link_prediction(x_eval, data.edge_index, model_3, adj_matrix, inverted_mask_matrix)
    
    
    ## GLASE
    # x_train = torch.concatenate((train_data.x, train_data.x_glase), axis=1)
    # x_val = torch.concatenate((val_data.x, val_data.x_glase), axis=1)
    # x_test = torch.concatenate((test_data.x, test_data.x_glase), axis=1)
    
    x_train = train_data.x, train_data.x_glase
    x_val = val_data.x, val_data.x_glase
    x_test = test_data.x, test_data.x_glase
    
    model_4 = train_link_prediction_GraphTransformer(x_train, x_val, x_test, train_data, val_data, test_data, input_dim=feat_dim, pe_dim=d)   

    ## Predict on entire masked graph
    # x_eval = torch.concatenate((data.x, data.x_glase), axis=1)
    x_eval = data.x, data.x_glase
    acc_glase = eval_link_prediction(x_eval, data.edge_index, model_4, adj_matrix, inverted_mask_matrix)
    
    if with_e2e:
        ## GLASE E2E
        model_5 = train_link_prediction_GraphTransformer_e2e(train_data, val_data, test_data, edge_index_2, Q, mask, feat_dim, d, gd_steps)
        acc_glase_e2e = eval_link_prediction_e2e(data.x, data.x_init, data.edge_index, edge_index_2, Q, mask, model_5, adj_matrix, inverted_mask_matrix)
        
        return model_1, model_2, model_3, model_4, model_5, acc_gcn, acc_ase, acc_grdpg, acc_glase, acc_glase_e2e
    else: 
        return model_1, model_2, model_3, model_4, acc_gcn, acc_ase, acc_grdpg, acc_glase

# def link_prediction_onu(year: int = 2018, d: int = 4, unknown_countries: int = 30, mask_threshold: float = 0.7, random_features: bool = False):
#     votes_df = load_un_dataset('data/UNVotes-1.csv', unknown_votes=True)
    
#     all_graphs = create_un_graphs(votes_df[votes_df.year==year])
    
#     adj_matrix, country_indexes, res_indexes, features, missing_countries, selected_resolutions, mask, inverted_mask_matrix = process_un_graph(all_graphs,countries=unknown_countries, mask_threshold=mask_threshold) 
#     print(selected_resolutions)
#     num_nodes = adj_matrix.shape[0]
#     edge_index = torch.tensor(adj_matrix).nonzero().t().contiguous()
    
#     ## Calculate Embeddings
#     ## ASE 
#     adj_matrix = to_dense_adj(edge_index.to('cpu'), max_num_nodes=num_nodes).squeeze(0)
#     ase = AdjacencySpectralEmbed(n_components=d, diag_aug=True, algorithm='full')
#     masked_adj = adj_matrix*mask
#     x_ase = ase.fit_transform(masked_adj.numpy())
#     x_ase = torch.from_numpy(x_ase)

#     A = to_dense_adj(edge_index.to('cpu'), max_num_nodes=num_nodes).squeeze(0)

#     u, V = torch.linalg.eig(A)

#     list_q=[]
#     for i in range(d):
#         if u[i].numpy()>0:
#             list_q.append(1)
#         else:
#             list_q.append(-1)
            
#     q = torch.Tensor(list_q)
#     Q=torch.diag(q)

#     x_grdpg, cost, k  = GRDPG_GD_Armijo(x_ase, edge_index, Q, mask.nonzero().t().contiguous())
#     x_grdpg = x_grdpg.detach()


#     gd_steps = 20
#     lr = 1e-2
#     device = 'cuda'
#     model = gLASE(d,d, gd_steps)
#     model.to(device)


#     epochs = 300

#     ## Initialization
#     for step in range(gd_steps):
#         model.gd[step].lin1.weight.data = (torch.eye(d,d)*lr).to(device)#torch.nn.init.xavier_uniform_(model.gd[step].lin1.weight)*lr
#         model.gd[step].lin2.weight.data = (torch.eye(d,d)*lr).to(device)#torch.nn.init.xavier_uniform_(model.gd[step].lin2.weight)*lr
        

#     optimizer = torch.optim.Adam(model.parameters(), lr=lr)

#     # Define ATT mask
#     edge_index_2 = torch.ones([num_nodes,num_nodes],).nonzero().t().contiguous().to(device)
#     mask = mask.to(device)
#     x_ase = x_ase.to(device)
#     edge_index = edge_index.to(device)
#     Q = Q.to(device)

#     for epoch in range(epochs):
#         # Train
#         model.train()

#         optimizer.zero_grad()
#         out = model(x_ase, edge_index, edge_index_2, Q, mask.nonzero().t().contiguous())
#         loss = torch.norm((out@Q@out.T - to_dense_adj(edge_index, max_num_nodes=num_nodes).squeeze(0))*mask)
#         loss.backward() 
#         optimizer.step() 
            
#     loss = torch.norm((out@Q@out.T - to_dense_adj(edge_index, max_num_nodes=num_nodes).squeeze(0))*mask)


#     x_glase = out.detach().to('cpu')
#     x_ase = x_ase.to('cpu')


#     masked_edge_index = masked_adj.nonzero().t().contiguous()
    
#     ###
    
#     if random_features:
#         torch.manual_seed(42)
#         random_features=torch.rand([num_nodes, 12])
#         data = Data(x=random_features.float(), x_ase=x_ase, x_glase=x_glase, x_grdpg=x_grdpg, edge_index=masked_edge_index)
#     else:
#         data = Data(x=features.float(), x_ase=x_ase, x_glase=x_glase, x_grdpg=x_grdpg, edge_index=masked_edge_index)

#     best_model_1, best_model_2, best_model_3, best_model_4, acc_gcn, acc_ase, acc_grdpg, acc_glase = link_prediction(edge_index, mask, inverted_mask_matrix, data, 12, d)
    
#     return best_model_1, best_model_2, best_model_3, best_model_4, acc_gcn, acc_ase, acc_grdpg, acc_glase


def link_prediction_onu_2(year: int = 2018, d: int = 4, unknown_countries: List[str] = ['FRA', 'SUD'], mask_threshold: float = 0.7, random_features: bool = False, iter: int =1, with_e2e: bool = False):
    votes_df = load_un_dataset('data/UNVotes-1.csv', unknown_votes=True)
    
    all_graphs = create_un_graphs(votes_df[votes_df.year==year])
    
    adj_matrix, country_indexes, res_indexes, unknown_edges, features, mask_nodes, mask, selected_resolutions, inverted_mask_matrix, mask_unknown = process_un_graph_2(all_graphs, mask_countries=unknown_countries, mask_threshold=mask_threshold) 

    # print(selected_resolutions)
    
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
    votes_df = load_un_dataset('data/UNVotes-1.csv', unknown_votes=True)
    
    all_graphs = create_un_graphs(votes_df[votes_df.year==year])
    
    adj_matrix, country_indexes, res_indexes, unknown_edges, features, mask_nodes, mask, selected_resolutions, inverted_mask_matrix, mask_unknown = process_un_graph_2(all_graphs, mask_countries=unknown_countries, mask_threshold=mask_threshold) 

    # print(selected_resolutions)
    
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
    #return best_model_1, best_model_2, best_model_3, best_model_4, acc_gcn, acc_ase, acc_grdpg, acc_glase, selected_resolutions




    # ## Calculate Embeddings
    # ## ASE 
    # adj_matrix = to_dense_adj(edge_index.to('cpu'), max_num_nodes=num_nodes).squeeze(0)
    # ase = AdjacencySpectralEmbed(n_components=d, diag_aug=True, algorithm='full')
    # masked_adj = adj_matrix*mask
    # x_ase = ase.fit_transform(masked_adj.numpy())
    # x_ase = torch.from_numpy(x_ase)

    # A = to_dense_adj(edge_index.to('cpu'), max_num_nodes=num_nodes).squeeze(0)

    # u, V = torch.linalg.eig(A)

    # list_q=[]
    # for i in range(d):
    #     if u[i].numpy()>0:
    #         list_q.append(1)
    #     else:
    #         list_q.append(-1)
            
    # q = torch.Tensor(list_q)
    # Q=torch.diag(q)

    # x_grdpg, cost, k  = GRDPG_GD_Armijo(x_ase, edge_index, Q, mask.nonzero().t().contiguous())
    # x_grdpg = x_grdpg.detach()


    # gd_steps = 20
    # lr = 1e-2
    # device = 'cuda'
    # model = gLASE(d,d, gd_steps)
    # model.to(device)


    # epochs = 300

    # ## Initialization
    # for step in range(gd_steps):
    #     model.gd[step].lin1.weight.data = (torch.eye(d,d)*lr).to(device)#torch.nn.init.xavier_uniform_(model.gd[step].lin1.weight)*lr
    #     model.gd[step].lin2.weight.data = (torch.eye(d,d)*lr).to(device)#torch.nn.init.xavier_uniform_(model.gd[step].lin2.weight)*lr
        

    # optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # # Define ATT mask
    # edge_index_2 = torch.ones([num_nodes,num_nodes],).nonzero().t().contiguous().to(device)
    # mask = mask.to(device)
    # x_ase = x_ase.to(device)
    # edge_index = edge_index.to(device)
    # Q = Q.to(device)

    # for epoch in range(epochs):
    #     # Train
    #     model.train()

    #     optimizer.zero_grad()
    #     out = model(x_ase, edge_index, edge_index_2, Q, mask.nonzero().t().contiguous())
    #     loss = torch.norm((out@Q@out.T - to_dense_adj(edge_index, max_num_nodes=num_nodes).squeeze(0))*mask)
    #     loss.backward() 
    #     optimizer.step() 

    #     # if epoch % 100 ==0:
    #     #     print(loss)
            
    # loss = torch.norm((out@Q@out.T - to_dense_adj(edge_index, max_num_nodes=num_nodes).squeeze(0))*mask)
    # # print(loss)

    # x_glase = out.detach().to('cpu')
    # x_ase = x_ase.to('cpu')


    # masked_edge_index = masked_adj.nonzero().t().contiguous()
    
    ###