import torch
import torch_geometric as pyg
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj

class Transformer_Block(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()

        # self.lin1 = nn.Linear(c_in, c_out, bias=False) ## skip
        self.lin2 = nn.Linear(c_in, c_out, bias=False) ## value
        self.lin3 = nn.Linear(c_in, c_out, bias=False) ## key
        self.lin4 = nn.Linear(c_in, c_out, bias=False) ## query

    def forward(self, x, edge_index, use_softmax=False, return_attn_matrix=False):

        ## Linear pass
        # x1 = self.lin1(x)
        x2 = self.lin2(x)
        _x3 = self.lin3(x)
        _x4 = self.lin4(x)

        ## Get indexes with edges
        x3 = torch.index_select(_x3, 0, edge_index[0])
        x4 = torch.index_select(_x4, 0, edge_index[1])
     
        ## Calculate attention weights
        attn_weights = torch.einsum("hc,hc->h", x3, x4)
        attn_weights = attn_weights.reshape(-1)
        
        adj_matrix = to_dense_adj(edge_index)
        

        if use_softmax:
            attn_matrix = attn_weights.new_zeros(adj_matrix.shape).fill_(-9e15)
            attn_matrix[adj_matrix==1] = attn_weights * (2**(-0.5))
            attn_matrix = F.softmax(attn_matrix, dim=1).squeeze(0)
        else:
            attn_matrix = to_dense_adj(edge_index, edge_attr=attn_weights).squeeze(0)

        ## Calculate final output
        x = torch.einsum("ij,ih->jh", attn_matrix, x2)
        
        if return_attn_matrix:
            return x, attn_matrix
        else:
            return x