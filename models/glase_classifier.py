import torch
import torch.nn as nn
from models.GAT import GATv2, GraphTransformer
from torch_geometric.nn import Sequential
from models.GLASE_unshared_normalized import GD_Block
from torch_geometric.utils import to_dense_adj


class MultiLayerPerceptron(nn.Module):
    def __init__(self, f_in, f_hid, f_out, num_layers, dropout):
        super(MultiLayerPerceptron, self).__init__()
        self.layers = nn.ModuleList()
        self.num_layers = num_layers
        if num_layers == 1:
            self.layers.append(nn.Linear(f_in, f_out))
        else:
            self.layers.append(nn.Linear(f_in, f_hid))
            for i in range(num_layers - 2):
                self.layers.append(nn.Linear(f_hid, f_hid))
            self.layers.append(nn.Linear(f_hid, f_out))
        if num_layers > 1:
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        for iter, layer in enumerate(self.layers):
            x = layer(x)
            if iter < self.num_layers - 1:
                x = self.dropout(self.relu(x))
        return x
    
    
class glaseClassifier(nn.Module):
    def __init__(self, f_in, emb_in, f_hid, emb_hid, f_out, n_layers, dropout1, dropout2, gd_steps):
        super(glaseClassifier, self).__init__()
        self.gd_steps = gd_steps
        self.dense1 = MultiLayerPerceptron(f_in, f_hid, f_hid, n_layers, dropout1) # features
        self.dense2 = MultiLayerPerceptron(emb_in, emb_hid, emb_hid, n_layers, dropout1) # embeddings
        self.classifier = MultiLayerPerceptron(f_hid+emb_hid, f_hid, f_out, n_layers, dropout2) # concat features + embeddings
   
    def forward(self, x_feat, x_emb):        
        x_feat = self.dense1(x_feat)
        x_emb = self.dense2(x_emb)
        x = torch.cat([x_feat, x_emb], dim=1)
        out = self.classifier(x)
        return out    

class glaseClassifierGAT(nn.Module):
    def __init__(self, f_in, emb_in, f_hid, emb_hid, f_out, n_layers, dropout1, dropout2, num_heads):
        super(glaseClassifierGAT, self).__init__()
        self.dense1 = MultiLayerPerceptron(f_in, f_hid, f_hid, n_layers, dropout1) # features
        self.dense2 = MultiLayerPerceptron(emb_in, emb_hid, emb_hid, n_layers, dropout1) # embeddings
        self.classifier = GATv2(f_hid+emb_in, f_hid, f_out, n_layers, dropout2, num_heads) # concat features + embeddings
   
    def forward(self, x_feat, x_emb, edge_index):        
        x_feat = self.dense1(x_feat)
        # x_emb = self.dense2(x_emb)
        x = torch.cat([x_feat, x_emb], dim=1)
        out = self.classifier(x, edge_index)
        return out  
    
    
class glaseClassifierTransformer(nn.Module):
    def __init__(self, f_in, emb_in, f_hid, emb_hid, f_out, n_layers, dropout1, dropout2, num_heads):
        super(glaseClassifierTransformer, self).__init__()
        self.dense1 = MultiLayerPerceptron(f_in, f_hid, f_hid, n_layers, dropout1) # features
        self.dense2 = MultiLayerPerceptron(emb_in, emb_hid, emb_hid, n_layers, dropout1) # embeddings
        self.classifier = GraphTransformer(f_hid+emb_hid, f_hid, f_out, n_layers, dropout2, num_heads) # concat features + embeddings
        self.softmax = torch.nn.LogSoftmax(dim=-1)
   
    def forward(self, x_feat, x_emb, edge_index):        
        x_feat = self.dense1(x_feat)
        x_emb = self.dense2(x_emb)
        x = torch.cat([x_feat, x_emb], dim=1)
        x = self.classifier(x, edge_index)
        out = self.softmax(x)
        return out  

class FeatureClassifier(nn.Module):
    def __init__(self, f_in, f_hid, f_out, n_layers, dropout1, dropout2):
        super(FeatureClassifier, self).__init__()
        self.dense1 = MultiLayerPerceptron(f_in, f_hid, f_hid, n_layers, dropout1) # features
        self.classifier = MultiLayerPerceptron(f_hid, f_hid, f_out, n_layers, dropout2) # concat features + embeddings
   
    def forward(self, x_feat):        
        x_feat = self.dense1(x_feat)
        out = self.classifier(x_feat)
        return out    
    
class FeatureClassifierGAT(nn.Module):
    def __init__(self, f_in, f_hid, f_out, n_layers, dropout1, dropout2, num_heads):
        super(FeatureClassifierGAT, self).__init__()
        self.dense1 = MultiLayerPerceptron(f_in, f_hid, f_hid, n_layers, dropout1) # features
        self.classifier = GATv2(f_hid, f_hid, f_out, n_layers, dropout2, num_heads) # concat features + embeddings
   
    def forward(self, x_feat, edge_index):        
        x_feat = self.dense1(x_feat)
        out = self.classifier(x_feat, edge_index)
        return out
    

class gLASE_e2e(nn.Module):
    def __init__(self, f_in, emb_in, f_hid, emb_hid, f_out, n_layers, dropout1, dropout2, gd_steps):
        super(gLASE_e2e, self).__init__()
        self.gd_steps = gd_steps
        self.incep1 = MultiLayerPerceptron(f_in, f_hid, f_hid, n_layers, dropout1) # features
        self.incep2 = MultiLayerPerceptron(emb_in, emb_hid, emb_hid, n_layers, dropout1) # embeddings
        self.classifier = MultiLayerPerceptron(f_hid+emb_in, f_hid, f_out, n_layers, dropout2) # concat features + embeddings
        layers = []

        for _ in range(gd_steps):
            layers.append((GD_Block(emb_in, emb_in), 'x, edge_index, edge_index_2, Q, mask -> x'))
        self.gd = Sequential('x, edge_index, edge_index_2, Q, mask', [layer for layer in layers])
    
    def forward(self, x_feat, x, edge_index, edge_index_2, Q, mask):
        
        x_feat = self.incep1(x_feat)
        x_emb = self.gd(x, edge_index, edge_index_2, Q, mask)
        x = torch.cat([x_feat, x_emb], dim=1)

        out = self.classifier(x)
        
        return out, x_emb    
    
    
class gLASE_e2e_GAT(nn.Module):
    def __init__(self, f_in, emb_in, f_hid, emb_hid, f_out, n_layers, dropout1, dropout2, gd_steps, num_heads):
        super(gLASE_e2e_GAT, self).__init__()
        self.gd_steps = gd_steps
        self.incep1 = MultiLayerPerceptron(f_in, f_hid, f_hid, n_layers, dropout1) # features
        self.incep2 = MultiLayerPerceptron(emb_in, emb_hid, emb_hid, n_layers, dropout1) # embeddings
        self.classifier = GATv2(f_hid+emb_in, f_hid, f_out, n_layers, dropout2, num_heads) # concat features + embeddings
        layers = []

        for _ in range(gd_steps):
            layers.append((GD_Block(emb_in, emb_in), 'x, edge_index, edge_index_2, Q, mask -> x'))
        self.gd = Sequential('x, edge_index, edge_index_2, Q, mask', [layer for layer in layers])
    
    def forward(self, x_feat, x, edge_index, edge_index_2, Q, mask):
        
        x_feat = self.incep1(x_feat)
        x_emb = self.gd(x, edge_index, edge_index_2, Q, mask)
        x = torch.cat([x_feat, x_emb], dim=1)

        num_nodes = x.shape[0]
        adj_matrix = to_dense_adj(edge_index, max_num_nodes = num_nodes).squeeze(0)
        mask_matrix = to_dense_adj(mask, max_num_nodes = num_nodes).squeeze(0)
        masked_adj = (adj_matrix*mask_matrix)
        masked_edge_index = masked_adj.nonzero().t().contiguous()

        out = self.classifier(x, masked_edge_index)
        
        return out, x_emb  

    def init_lase(self, lr):
        for step in range(self.gd_steps):
            self.gd[step].lin1.weight.data = torch.nn.init.xavier_uniform_(self.gd[step].lin1.weight)*lr
            self.gd[step].lin2.weight.data = torch.nn.init.xavier_uniform_(self.gd[step].lin2.weight)*lr
    
    
class gLASE_e2e_Transformer(nn.Module):
    def __init__(self, f_in, emb_in, f_hid, emb_hid, f_out, n_layers, dropout1, dropout2, gd_steps, num_heads):
        super(gLASE_e2e_Transformer, self).__init__()
        self.gd_steps = gd_steps
        self.incep1 = MultiLayerPerceptron(f_in, f_hid, f_hid, n_layers, dropout1) # features
        self.incep2 = MultiLayerPerceptron(emb_in, emb_hid, emb_hid, n_layers, dropout1) # embeddings
        self.classifier = GraphTransformer(f_hid+emb_hid, f_hid, f_out, n_layers, dropout2, num_heads) # concat features + embeddings
        self.softmax = torch.nn.LogSoftmax(dim=-1)
        layers = []

        for _ in range(gd_steps):
            layers.append((GD_Block(emb_in, emb_in), 'x, edge_index, edge_index_2, Q, mask -> x'))
        self.gd = Sequential('x, edge_index, edge_index_2, Q, mask', [layer for layer in layers])
    
    def forward(self, x_feat, x, edge_index, edge_index_2, Q, mask):
        
        x_feat = self.incep1(x_feat)
        x_emb = self.gd(x, edge_index, edge_index_2, Q, mask)
        x_emb = self.incep2(x_emb)
        x = torch.cat([x_feat, x_emb], dim=1)

        num_nodes = x.shape[0]
        adj_matrix = to_dense_adj(edge_index, max_num_nodes = num_nodes).squeeze(0)
        mask_matrix = to_dense_adj(mask, max_num_nodes = num_nodes).squeeze(0)
        masked_adj = (adj_matrix*mask_matrix)
        masked_edge_index = masked_adj.nonzero().t().contiguous()
            
        x = self.classifier(x, edge_index)
        out = self.softmax(x)
        
        return out, x_emb  
    
    def init_lase(self, lr):
        for step in range(self.gd_steps):
            self.gd[step].lin1.weight.data = torch.nn.init.xavier_uniform_(self.gd[step].lin1.weight)*lr
            self.gd[step].lin2.weight.data = torch.nn.init.xavier_uniform_(self.gd[step].lin2.weight)*lr


