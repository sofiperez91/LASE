import torch.nn as nn
from torch_geometric.nn import TAGConv


class GCN_Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.gcn = TAGConv(in_channels, out_channels, K=1, normalize=True, bias=False)

    def forward(self, input, edge_index): 
        x_1 = self.gcn(input, edge_index)
        return x_1

class GCN(nn.Module):
    def __init__(self, in_channels, out_channels, gd_steps):
        super().__init__()

        self.gd_steps = gd_steps
        self.gd = GCN_Block(in_channels,out_channels)

    def forward(self, input, edge_index): 
        x = input
        for _ in range(self.gd_steps):
             x = self.gd(x, edge_index)
        return x
