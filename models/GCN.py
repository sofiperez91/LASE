import torch
from torch_geometric.nn import TAGConv

# class GCN(torch.nn.Module):
#     def __init__(self, input_dim, output_dim, K=1):
#       super(GCN, self).__init__()
#       self.convs = TAGConv(input_dim, output_dim, K=K)

#     def forward(self, x, edge_index):
#       x = self.convs(x, edge_index)
#       return x
    
    
class GCN(torch.nn.Module):
    def __init__(self,input_dim, output_dim, K=10, n_layers=1):
        super(GCN, self).__init__()
        # torch.manual_seed(1234)
        self.conv1 = TAGConv(input_dim, output_dim, K=K)
        self.n_layers = n_layers
        self.convs = []
        for _ in range(n_layers):
            conv = TAGConv(input_dim, output_dim, K=3)
            self.convs.append(conv)

        self.convs = torch.nn.ModuleList(self.convs)
        self.classifier = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x, edge_index):
        h = x

        h = self.conv1(h, edge_index)
        h = h.tanh()
        
        for k in range(self.n_layers):
            conv = self.convs[k]
            h = conv(h, edge_index)
            h = h.tanh()
        
        # Apply a final (linear) layer that outputs the embeddings.
        out = self.classifier(h)

        return out
    
    
    