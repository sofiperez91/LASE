import torch
from torch_geometric.nn import TAGConv

class GCN(torch.nn.Module):
    def __init__(self, input_dim, output_dim, K=1):
      super(GCN, self).__init__()
      self.convs = TAGConv(input_dim, output_dim, K=K)

    def forward(self, x, edge_index):
      x = self.convs(x, edge_index)
      return x