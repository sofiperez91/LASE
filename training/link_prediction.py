import torch
from torch_geometric.utils import to_dense_adj

import sys
sys.path.append("../")
from models.link_prediction import train_link_prediction, eval_link_prediction, train_link_prediction_GAT, train_link_prediction_Transformer, train_link_prediction_GraphTransformer
from models.glase_e2e_link_prediction import train_link_prediction_e2e, eval_link_prediction_e2e, train_link_prediction_GAT_e2e, train_link_prediction_Transformer_e2e, train_link_prediction_GraphTransformer_e2e

import torch_geometric.transforms as T


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
    data = data.to('cuda')
    
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
    
    model_1 = train_link_prediction(x_train, x_val, x_test, train_data, val_data, test_data, input_dim=feat_dim)

    ## Predict on entire masked graph
    x_eval = data.x
    acc_gcn = eval_link_prediction(x_eval, data.edge_index, model_1, adj_matrix, inverted_mask_matrix)

    ## ASE 
    x_train = train_data.x, train_data.x_ase
    x_val = val_data.x, val_data.x_ase
    x_test = test_data.x, test_data.x_ase
    
    
    model_2 = train_link_prediction_Transformer(x_train, x_val, x_test, train_data, val_data, test_data, input_dim=feat_dim, pe_dim=d)

    ## Predict on entire masked graph
    x_eval = data.x, data.x_ase
    acc_ase = eval_link_prediction(x_eval, data.edge_index, model_2, adj_matrix, inverted_mask_matrix)
    
    
    ## GD-GRDPG
    x_train = train_data.x, train_data.x_grdpg
    x_val = val_data.x, val_data.x_grdpg
    x_test = test_data.x, test_data.x_grdpg

    model_3 = train_link_prediction_Transformer(x_train, x_val, x_test, train_data, val_data, test_data, input_dim=feat_dim, pe_dim=d)
    
    ## Predict on entire masked graph
    x_eval = data.x, data.x_grdpg
    acc_grdpg = eval_link_prediction(x_eval, data.edge_index, model_3, adj_matrix, inverted_mask_matrix)
    
    
    ## GLASE
    x_train = train_data.x, train_data.x_glase
    x_val = val_data.x, val_data.x_glase
    x_test = test_data.x, test_data.x_glase
    
    model_4 = train_link_prediction_Transformer(x_train, x_val, x_test, train_data, val_data, test_data, input_dim=feat_dim, pe_dim=d)       

    ## Predict on entire masked graph
    x_eval = data.x, data.x_glase
    acc_glase = eval_link_prediction(x_eval, data.edge_index, model_4, adj_matrix, inverted_mask_matrix)
    
    if with_e2e:
        ## GLASE E2E
        model_5 = train_link_prediction_Transformer_e2e(train_data, val_data, test_data, edge_index_2, Q, mask, feat_dim, d, gd_steps)
        acc_glase_e2e = eval_link_prediction_e2e(data.x, data.x_init, data.edge_index, edge_index_2, Q, mask, model_5, adj_matrix, inverted_mask_matrix)
        
        
        return model_1, model_2, model_3, model_4, model_5, acc_gcn, acc_ase, acc_grdpg, acc_glase, acc_glase_e2e
    else: 
        return model_1, model_2, model_3, model_4, acc_gcn, acc_ase, acc_grdpg, acc_glase
    

def link_prediction_GraphTransformer(edge_index, edge_index_2, Q, mask, inverted_mask_matrix, data, feat_dim: int, d: int = 4, gd_steps: int = 20, with_e2e: bool = True,
                                     epochs = 101, output_dim= 32, pe_out_dim= 8, n_layers= 3, dropout=0.5, num_heads=4, batch_norm=True, lr=0.01):

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
    x_train = train_data.x, train_data.x_ase
    x_val = val_data.x, val_data.x_ase
    x_test = test_data.x, test_data.x_ase
    
    
    
    model_2 = train_link_prediction_GraphTransformer(x_train, x_val, x_test, train_data, val_data, test_data, input_dim=feat_dim, pe_dim=d, epochs = epochs, 
                                                     output_dim= output_dim, pe_out_dim= pe_out_dim, n_layers= n_layers, dropout=dropout, num_heads=num_heads, batch_norm=batch_norm, lr=lr)
    ## Predict on entire masked graph
    x_eval = data.x, data.x_ase
    acc_ase = eval_link_prediction(x_eval, data.edge_index, model_2, adj_matrix, inverted_mask_matrix)
    
    ## GD-GRDPG
    x_train = train_data.x, train_data.x_grdpg
    x_val = val_data.x, val_data.x_grdpg
    x_test = test_data.x, test_data.x_grdpg
    

    model_3 = train_link_prediction_GraphTransformer(x_train, x_val, x_test, train_data, val_data, test_data, input_dim=feat_dim, pe_dim=d,epochs = epochs, 
                                                     output_dim= output_dim, pe_out_dim= pe_out_dim, n_layers= n_layers, dropout=dropout, num_heads=num_heads, batch_norm=batch_norm, lr=lr)
    
    ## Predict on entire masked graph
    x_eval = data.x, data.x_grdpg
    acc_grdpg = eval_link_prediction(x_eval, data.edge_index, model_3, adj_matrix, inverted_mask_matrix)
    
    ## GLASE
    x_train = train_data.x, train_data.x_glase
    x_val = val_data.x, val_data.x_glase
    x_test = test_data.x, test_data.x_glase
    
    model_4 = train_link_prediction_GraphTransformer(x_train, x_val, x_test, train_data, val_data, test_data, input_dim=feat_dim, pe_dim=d,epochs = epochs, 
                                                     output_dim= output_dim, pe_out_dim= pe_out_dim, n_layers= n_layers, dropout=dropout, num_heads=num_heads, batch_norm=batch_norm, lr=lr)   

    ## Predict on entire masked graph
    x_eval = data.x, data.x_glase
    acc_glase = eval_link_prediction(x_eval, data.edge_index, model_4, adj_matrix, inverted_mask_matrix)
    
    if with_e2e:
        ## GLASE E2E
        model_5 = train_link_prediction_GraphTransformer_e2e(train_data, val_data, test_data, edge_index_2, Q, mask, feat_dim, d, gd_steps)
        acc_glase_e2e = eval_link_prediction_e2e(data.x, data.x_init, data.edge_index, edge_index_2, Q, mask, model_5, adj_matrix, inverted_mask_matrix)
        
        return model_1, model_2, model_3, model_4, model_5, acc_gcn, acc_ase, acc_grdpg, acc_glase, acc_glase_e2e
    else: 
        return model_1, model_2, model_3, model_4, acc_gcn, acc_ase, acc_grdpg, acc_glase

