import torch
import torch.nn as nn
from torch_geometric.nn import TAGConv, Sequential
from torch_geometric.utils import to_dense_adj

from models.Transformer_Block import Transformer_Block

class LaseBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.gcn = TAGConv(in_channels, out_channels, K=1, normalize=False, bias=False)
        self.gat = Transformer_Block(in_channels, out_channels)

    def forward(self, input, edge_index, edge_index_2, mask): 
        ## Apply mask
        edge_index = (to_dense_adj(edge_index).squeeze(0)*to_dense_adj(mask).squeeze(0)).nonzero().t().contiguous()
        edge_index_2 = (to_dense_adj(edge_index_2).squeeze(0)*to_dense_adj(mask).squeeze(0)).nonzero().t().contiguous()

        ## Normalization parameters
        n = input.shape[0]
        p_1 = (mask.shape[1]) / n**2
        p_2 = (edge_index_2.shape[1]) / n**2
        
        x_1 = self.gcn(input, edge_index) / (n*p_1) + (n*p_1-1)/(n*p_1)*input
        x_2 = self.gat(input, edge_index_2, use_softmax=False, return_attn_matrix=False) / (n*p_2)
        return x_1 - x_2

class LASE(nn.Module):
    def __init__(self, in_channels, out_channels, gd_steps):
        super().__init__()

        self.gd_steps = gd_steps
        layers = []

        for _ in range(gd_steps):
            layers.append((LaseBlock(in_channels, out_channels), 'x, edge_index, edge_index_2, mask -> x'))
        self.gd = Sequential('x, edge_index, edge_index_2, mask', [layer for layer in layers])

    def forward(self, input, edge_index, edge_index_2, mask): 
        x = input
        x = self.gd(x, edge_index, edge_index_2, mask)        
        return x

    def init_lase(self, lr, in_channels):
        for step in range(self.gd_steps):
            # TAGConv
            self.gd[step].gcn.lins[0].weight.data = torch.eye(in_channels)
            self.gd[step].gcn.lins[0].weight.requires_grad = False
            self.gd[step].gcn.lins[1].weight.data = torch.nn.init.xavier_uniform_(self.gd[step].gcn.lins[1].weight)*lr

            # TransformerBlock
            self.gd[step].gat.lin2.weight.data = lr*torch.nn.init.xavier_uniform_(self.gd[step].gat.lin2.weight.data)

            self.gd[step].gat.lin3.weight.data = torch.eye(in_channels)
            self.gd[step].gat.lin3.weight.requires_grad = False
            self.gd[step].gat.lin4.weight.data = torch.eye(in_channels)
            self.gd[step].gat.lin4.weight.requires_grad = False