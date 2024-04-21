import torch 
from torch import nn
from torch_geometric.nn import GCNConv
from torch_geometric.nn import TransformerConv
from models.GAT import GATv2, GraphTransformer
from models.GraphTransformerLayer import GraphTransformerLayer
from sklearn.metrics import roc_auc_score
from torch_geometric.utils import negative_sampling
from torch_geometric.utils import to_dense_adj
from torch_geometric.nn import Sequential
from models.GLASE_unshared_normalized import GD_Block
from torch_geometric.utils import to_dgl
from torch_geometric.data import Data

class GLASELinkPrediction(torch.nn.Module):
    def __init__(self, feat_in, emb_in, hidden_channels, out_channels, gd_steps):
        super().__init__()
        self.gd_steps = gd_steps
        self.conv1 = GCNConv(feat_in+emb_in, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        layers = []

        for _ in range(gd_steps):
            layers.append((GD_Block(emb_in, emb_in), 'x, edge_index, edge_index_2, Q, mask -> x'))
        self.gd = Sequential('x, edge_index, edge_index_2, Q, mask', [layer for layer in layers])
        
        
    def encode(self, x_feat, x_init, edge_index, edge_index_2, Q, mask):
        x_emb = self.gd(x_init, edge_index, edge_index_2, Q, mask.nonzero().t().contiguous())
        x = torch.cat([x_feat, x_emb], dim=1)
        x = self.conv1(x, edge_index).relu()
        out = self.conv2(x, edge_index)
        return out, x_emb

    def decode(self, z, edge_label_index):
        return (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(dim=-1)

    def decode_all(self, z):
        prob_adj = z @ z.t()
        return (prob_adj > 0).nonzero(as_tuple=False).t()


class GLASELinkPredictionGAT(torch.nn.Module):
    def __init__(self, in_channels, emb_in, hidden_channels, out_channels, gd_steps, n_layers, dropout, num_heads):
        super().__init__()     
        self.gd_steps = gd_steps
        self.gat = GATv2(in_channels+emb_in, hidden_channels, out_channels, n_layers, dropout, num_heads) # concat features + embeddings
        layers = []
        
        for _ in range(gd_steps):
            layers.append((GD_Block(emb_in, emb_in), 'x, edge_index, edge_index_2, Q, mask -> x'))
        self.gd = Sequential('x, edge_index, edge_index_2, Q, mask', [layer for layer in layers])
        
    def encode(self, x_feat, x_init, edge_index, edge_index_2, Q, mask):
        x_emb = self.gd(x_init, edge_index, edge_index_2, Q, mask.nonzero().t().contiguous())
        x = torch.cat([x_feat, x_emb], dim=1)
        out = self.gat(x, edge_index)
        return out, x_emb

    def decode(self, z, edge_label_index):
        return (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(dim=-1)

    def decode_all(self, z):
        prob_adj = z @ z.t()
        return (prob_adj > 0).nonzero(as_tuple=False).t()
    
class GLASELinkPredictionTransformer(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, pe_dim_in, pe_dim_out, gd_steps, n_layers, dropout, num_heads):
        super().__init__()
        self.feat_lin = nn.Linear(in_channels, hidden_channels, bias=True)
        self.pe_lin = nn.Linear(pe_dim_in, pe_dim_out, bias=True)  
        self.gd_steps = gd_steps
        self.transformer = GraphTransformer(hidden_channels+pe_dim_out, hidden_channels, out_channels, n_layers, dropout, num_heads) # concat features + embeddings

        layers = []
        
        for _ in range(gd_steps):
            layers.append((GD_Block(pe_dim_in, pe_dim_in), 'x, edge_index, edge_index_2, Q, mask -> x'))
        self.gd = Sequential('x, edge_index, edge_index_2, Q, mask', [layer for layer in layers])
        
    def encode(self, x_feat, x_init, edge_index, edge_index_2, Q, mask):
        x_feat=self.feat_lin(x_feat)
        x_pe = self.gd(x_init, edge_index, edge_index_2, Q, mask.nonzero().t().contiguous())
        x_pe = self.pe_lin(x_pe)
        x = torch.concatenate((x_feat, x_pe), axis=1)
        
        out = self.transformer(x, edge_index)
        return out, x_pe

    def decode(self, z, edge_label_index):
        return (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(dim=-1)

    def decode_all(self, z):
        prob_adj = z @ z.t()
        return (prob_adj > 0).nonzero(as_tuple=False).t()


class GLASELinkPredictionGraphTransformer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, pe_dim_in, pe_dim_out, gd_steps, n_layers, dropout, num_heads):
        super().__init__()
        self.gd_steps = gd_steps
        
        self.feat_lin = nn.Linear(in_channels, out_channels, bias=True)
        self.pe_lin = nn.Linear(pe_dim_in, pe_dim_out, bias=True)
        
        self.transformer_layers = nn.ModuleList([GraphTransformerLayer(out_channels+pe_dim_out, out_channels+pe_dim_out, num_heads, dropout) for _ in range(n_layers)])
            
        layers = []
        
        for _ in range(gd_steps):
            layers.append((GD_Block(pe_dim_in, pe_dim_in), 'x, edge_index, edge_index_2, Q, mask -> x'))
        self.gd = Sequential('x, edge_index, edge_index_2, Q, mask', [layer for layer in layers])
        
    def encode(self, x_feat, x_init, edge_index, edge_index_2, Q, mask):
        x_feat=self.feat_lin(x_feat)
        x_pe = self.gd(x_init, edge_index, edge_index_2, Q, mask.nonzero().t().contiguous())
        x_pe = self.pe_lin(x_pe)
        x = torch.cat([x_feat, x_pe], dim=1)
        
        data = Data(x=x, edge_index=edge_index)
        # Transform to DGL
        g = to_dgl(data) 
        
        # GraphTransformer Layers
        for conv in self.transformer_layers:
            x = conv(g, x)
        return x, x_pe

    def decode(self, z, edge_label_index):
        return (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(dim=-1)

    def decode_all(self, z):
        prob_adj = z @ z.t()
        return (prob_adj > 0).nonzero(as_tuple=False).t()


def train(x_feat, x_init, train_data, edge_index_2, Q, mask, model, optimizer, criterion, alpha: float = 0.99):
    model.train()
    optimizer.zero_grad()
    z, x_glase = model.encode(x_feat, x_init, train_data.edge_index, edge_index_2, Q, mask)

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
    loss1 = criterion(out, edge_label)
    loss2 = torch.norm((x_glase@x_glase.T - to_dense_adj(train_data.edge_index, max_num_nodes=train_data.num_nodes).squeeze(0))*mask)
    loss = alpha*loss1 + (1-alpha)*loss2
    loss.backward()
    optimizer.step()
    return loss

@torch.no_grad()
def test(x_feat, x_init, data, model, edge_index_2, Q, mask):
    model.eval()
    z, x_glase = model.encode(x_feat, x_init, data.edge_index, edge_index_2, Q, mask)
    out = model.decode(z, data.edge_label_index).view(-1).sigmoid()
    return roc_auc_score(data.edge_label.cpu().numpy(), out.cpu().numpy())

def train_link_prediction_e2e(train_data, val_data, test_data,edge_index_2, Q, mask, feat_dim, emb_dim, gd_steps: int = 20, epochs: int = 101): 
    device = "cuda"
    model = GLASELinkPrediction(feat_dim, emb_dim, 128, 64, gd_steps).to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)
    criterion = torch.nn.BCEWithLogitsLoss()
    best_val_auc = final_test_auc = 0
    for epoch in range(1, epochs):
        loss = train(train_data.x, train_data.x_init, train_data, edge_index_2, Q, mask, model, optimizer, criterion)
        val_auc = test(val_data.x, val_data.x_init, val_data, model, edge_index_2, Q, mask)
        test_auc = test(test_data.x, test_data.x_init, test_data, model, edge_index_2, Q, mask)
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            final_test_auc = test_auc
    return model


def train_link_prediction_GAT_e2e(train_data, val_data, test_data, edge_index_2, Q, mask, feat_dim, emb_dim, gd_steps: int = 20, epochs: int = 101): 
    device = "cuda"
    model = GLASELinkPredictionGAT(feat_dim, emb_dim, 128, 64, gd_steps,n_layers=3, dropout=0.5, num_heads=1).to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)
    criterion = torch.nn.BCEWithLogitsLoss()
    best_val_auc = final_test_auc = 0
    for epoch in range(1, epochs):
        loss = train(train_data.x, train_data.x_init, train_data, edge_index_2, Q, mask, model, optimizer, criterion)
        val_auc = test(val_data.x, val_data.x_init, val_data, model, edge_index_2, Q, mask)
        test_auc = test(test_data.x, test_data.x_init, test_data, model, edge_index_2, Q, mask)
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            final_test_auc = test_auc
    return model

def train_link_prediction_Transformer_e2e(train_data, val_data, test_data, edge_index_2, Q, mask, feat_dim, emb_dim, gd_steps: int = 20, epochs: int = 301): 
    device = "cuda"
    model = GLASELinkPredictionTransformer(feat_dim, 64, 32, emb_dim, 8, gd_steps, n_layers=3, dropout=0.5, num_heads=4).to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)
    criterion = torch.nn.BCEWithLogitsLoss()
    best_val_auc = final_test_auc = 0
    for epoch in range(1, epochs):
        loss = train(train_data.x, train_data.x_init, train_data, edge_index_2, Q, mask, model, optimizer, criterion)
        val_auc = test(val_data.x, val_data.x_init, val_data, model, edge_index_2, Q, mask)
        test_auc = test(test_data.x, test_data.x_init, test_data, model, edge_index_2, Q, mask)
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            final_test_auc = test_auc
    return model

def train_link_prediction_GraphTransformer_e2e(train_data, val_data, test_data, edge_index_2, Q, mask, feat_dim, emb_dim, gd_steps: int = 20, epochs: int = 301): 
    device = "cuda"
    model = GLASELinkPredictionGraphTransformer(feat_dim, 32, emb_dim, 8, gd_steps, n_layers=3, dropout=0.5, num_heads=4).to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)
    criterion = torch.nn.BCEWithLogitsLoss()
    best_val_auc = final_test_auc = 0
    for epoch in range(1, epochs):
        loss = train(train_data.x, train_data.x_init, train_data, edge_index_2, Q, mask, model, optimizer, criterion)
        val_auc = test(val_data.x, val_data.x_init, val_data, model, edge_index_2, Q, mask)
        test_auc = test(test_data.x, test_data.x_init, test_data, model, edge_index_2, Q, mask)
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            final_test_auc = test_auc
    return model


def eval_link_prediction_e2e(x_feat, x_init, edge_index, edge_index_2, Q, mask, model, adj_matrix, inverted_mask_matrix): 
    num_nodes = adj_matrix.shape[0]
    z, x_glase = model.encode(x_feat, x_init, edge_index, edge_index_2, Q, mask)
    final_edge_index = model.decode_all(z)
    predicted_adj = to_dense_adj(final_edge_index, max_num_nodes=num_nodes).squeeze(0).to('cpu')
    indices = torch.where(inverted_mask_matrix.to('cpu'))
    acc =((adj_matrix[indices[0], indices[1]]) == (predicted_adj[indices[0], indices[1]])).sum() / indices[0].shape[0]

    return acc