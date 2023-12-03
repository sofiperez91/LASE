import torch
import torch.nn as nn
from models.GAT import GATv2
from torch_geometric.nn import Sequential
from models.GLASE_unshared_normalized import GD_Block


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
        # x_emb_incep = self.incep2(x_emb)
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
        # x_emb_incep = self.incep2(x_emb)
        x = torch.cat([x_feat, x_emb], dim=1)

        out = self.classifier(x, edge_index)
        
        return out, x_emb  


