import torch
import torch.nn as nn
import torch_geometric as pyg
from torch_geometric.nn import TAGConv, LayerNorm, Sequential
# from Transformer_Block_v2 import Transformer_Block

class GD_Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.gcn = TAGConv(in_channels, out_channels, K=1, normalize=True, bias = False)

    def forward(self, input, edge_index):
        x_1 = self.gcn(input, edge_index)

        return x_1

class GD_Unroll(nn.Module):
    def __init__(self, in_channels, out_channels, gd_steps):
        super().__init__()
        layers = []

        for _ in range(gd_steps):
            layers.append((GD_Block(in_channels, out_channels), 'x, edge_index -> x'))
        self.gd = Sequential('x, edge_index', [layer for layer in layers])

    def forward(self, input, edge_index):
        x = input
        x = self.gd(x, edge_index)
        return x