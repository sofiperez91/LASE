import torch 
from torch import nn
from torch_geometric.nn import GCNConv
from torch_geometric.nn import TransformerConv
from models.GraphTransformerLayer import GraphTransformerLayer
from models.GAT import GATv2
from sklearn.metrics import roc_auc_score
from torch_geometric.utils import negative_sampling
from torch_geometric.utils import to_dense_adj
from torch_geometric.utils import to_dgl
import copy
from torch_geometric.data import Data


class Net(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)

    def decode(self, z, edge_label_index):
        return (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(dim=-1)

    def decode_all(self, z):
        prob_adj = z @ z.t()
        return (prob_adj > 0).nonzero(as_tuple=False).t()
    
class Net2(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def encode(self, x, x_pe, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return torch.concatenate((x, x_pe), axis=1)

    def decode(self, z, edge_label_index):
        return (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(dim=-1)

    def decode_all(self, z):
        prob_adj = z @ z.t()
        return (prob_adj > 0).nonzero(as_tuple=False).t()
    
    
class GATLinkPrediction(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, n_layers, dropout, num_heads):
        super().__init__()
        self.gat = GATv2(in_channels, hidden_channels, out_channels, n_layers, dropout, num_heads) # concat features + embeddings

    def encode(self, x, edge_index):
        x = self.gat(x, edge_index)
        return x

    def decode(self, z, edge_label_index):
        return (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(dim=-1)

    def decode_all(self, z):
        prob_adj = z @ z.t()
        return (prob_adj > 0).nonzero(as_tuple=False).t()
    
    
class TransformerLinkPrediction(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, n_layers, dropout, num_heads):
        super().__init__()
        self.gat = TransformerConv(in_channels, out_channels, heads=num_heads, bias = True) # concat features + embeddings

    def encode(self, x, edge_index):
        x = self.gat(x, edge_index)
        return x

    def decode(self, z, edge_label_index):
        return (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(dim=-1)

    def decode_all(self, z):
        prob_adj = z @ z.t()
        return (prob_adj > 0).nonzero(as_tuple=False).t()
    
    
# class GraphTransformerLinkPrediction(torch.nn.Module):
#     def __init__(self, in_channels, hidden_channels, out_channels, n_layers, dropout, num_heads):
#         super().__init__()
#         self.layers = nn.ModuleList()
        
#         self.layers.append(GraphTransformerLayer(in_channels, hidden_channels, num_heads, dropout, residual=False))
#         for _ in range(n_layers - 2):
#             self.layers.append(GraphTransformerLayer(hidden_channels, hidden_channels, num_heads,
#                                                 dropout))
#         self.layers.append(GraphTransformerLayer(hidden_channels, out_channels, num_heads, dropout))
        

#     def encode(self, x, edge_index):
#         data = Data(x=x, edge_index=edge_index)
#         # Transform to DGL
#         g = to_dgl(data) 
        
#         # GraphTransformer Layers
#         for conv in self.layers:
#             x = conv(g, x)

#         return x

class GraphTransformerLinkPrediction(torch.nn.Module):
    def __init__(self, in_channels, out_channels, pe_dim_in, pe_dim_out, n_layers, dropout, num_heads):
        super().__init__()
        
        self.feat_lin = nn.Linear(in_channels, out_channels, bias=True)
        self.pe_lin = nn.Linear(pe_dim_in, pe_dim_out, bias=True)
        self.layers = nn.ModuleList([GraphTransformerLayer(out_channels+pe_dim_out, out_channels+pe_dim_out, num_heads, dropout) for _ in range(n_layers)])
        

    def encode(self, x_in, edge_index):
        x_feat, x_pe = x_in
        x_pe = self.pe_lin(x_pe)
        x_feat=self.feat_lin(x_feat)
        x = torch.concatenate((x_feat, x_pe), axis=1)
        data = Data(x=x, edge_index=edge_index)
        # Transform to DGL
        g = to_dgl(data) 
        
        # GraphTransformer Layers
        for conv in self.layers:
            x = conv(g, x)

        return x



    def decode(self, z, edge_label_index):
        return (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(dim=-1)

    def decode_all(self, z):
        prob_adj = z @ z.t()
        return (prob_adj > 0).nonzero(as_tuple=False).t()
    
def train(x_input, train_data, model, optimizer, criterion):
    model.train()
    optimizer.zero_grad()
    z = model.encode(x_input, train_data.edge_index)

    # We perform a new round of negative sampling for every training epoch:
    neg_edge_index = negative_sampling(
        edge_index=train_data.edge_index, num_nodes=train_data.num_nodes,
        num_neg_samples=train_data.edge_label_index.size(1), method='sparse')

    edge_label_index = torch.cat(
        [train_data.edge_label_index, neg_edge_index],
        dim=-1,
    )
    edge_label = torch.cat([
        train_data.edge_label,
        train_data.edge_label.new_zeros(neg_edge_index.size(1))
    ], dim=0)

    out = model.decode(z, edge_label_index).view(-1)
    loss = criterion(out, edge_label)
    loss.backward()
    optimizer.step()
    return loss


@torch.no_grad()
def test(x_input, data, model):
    model.eval()
    z = model.encode(x_input, data.edge_index)
    out = model.decode(z, data.edge_label_index).view(-1).sigmoid()
    return roc_auc_score(data.edge_label.cpu().numpy(), out.cpu().numpy())

def train_link_prediction(x_train, x_val, x_test, train_data, val_data, test_data, input_dim, epochs = 101): 
    device = "cuda"
    model = Net(input_dim, 128, 64).to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)
    criterion = torch.nn.BCEWithLogitsLoss()
    best_val_auc = final_test_auc = 0
    for epoch in range(1, epochs):
        loss = train(x_train, train_data, model, optimizer, criterion)
        val_auc = test(x_val, val_data, model)
        test_auc = test(x_test, test_data, model)
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            final_test_auc = test_auc
    return model

def train_link_prediction_GAT(x_train, x_val, x_test, train_data, val_data, test_data, input_dim, epochs = 101): 
    device = "cuda"
    model = GATLinkPrediction(input_dim, 128, 64, n_layers=3, dropout=0.5, num_heads=1).to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)
    criterion = torch.nn.BCEWithLogitsLoss()
    best_val_auc = final_test_auc = 0
    for epoch in range(1, epochs):
        loss = train(x_train, train_data, model, optimizer, criterion)
        val_auc = test(x_val, val_data, model)
        test_auc = test(x_test, test_data, model)
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            final_test_auc = test_auc
    return model

def train_link_prediction_Transformer(x_train, x_val, x_test, train_data, val_data, test_data, input_dim, epochs = 101): 
    device = "cuda"
    model = TransformerLinkPrediction(input_dim, 128, 64, n_layers=3, dropout=0.5, num_heads=2).to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)
    criterion = torch.nn.BCEWithLogitsLoss()
    best_val_auc = final_test_auc = 0
    for epoch in range(1, epochs):
        loss = train(x_train, train_data, model, optimizer, criterion)
        val_auc = test(x_val, val_data, model)
        test_auc = test(x_test, test_data, model)
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            final_test_auc = test_auc
    return model

def train_link_prediction_GraphTransformer(x_train, x_val, x_test, train_data, val_data, test_data, input_dim, pe_dim,  epochs = 301): 
    device = "cuda"
    model = GraphTransformerLinkPrediction(input_dim, 32, pe_dim, 8, n_layers=3, dropout=0.5, num_heads=4).to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)
    criterion = torch.nn.BCEWithLogitsLoss()
    best_val_auc = final_test_auc = 0
    for epoch in range(1, epochs):
        loss = train(x_train, train_data, model, optimizer, criterion)
        val_auc = test(x_val, val_data, model)
        test_auc = test(x_test, test_data, model)
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            final_test_auc = test_auc
    return model

def eval_link_prediction(x_eval, edge_index, model, adj_matrix, inverted_mask_matrix): 
    num_nodes = adj_matrix.shape[0]
    z = model.encode(x_eval, edge_index)
    final_edge_index = model.decode_all(z)
    predicted_adj = to_dense_adj(final_edge_index, max_num_nodes=num_nodes).squeeze(0).to('cpu')
    indices = torch.where(inverted_mask_matrix.to('cpu'))
    acc =((adj_matrix[indices[0], indices[1]]) == (predicted_adj[indices[0], indices[1]])).sum() / indices[0].shape[0]

    return acc

