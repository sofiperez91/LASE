import torch
import random
from torch_geometric.utils import to_dense_adj, erdos_renyi_graph

def big_bird_attention(window_size, random_prob, num_nodes):
    # Initialize attention matrix

    edge_index = erdos_renyi_graph(num_nodes, random_prob, directed=False)
    attention_matrix = to_dense_adj(edge_index, max_num_nodes=num_nodes).squeeze(0)
    # attention_matrix = torch.zeros(num_nodes, num_nodes)

    # Sliding window 
    for i in range(num_nodes):
        start = max(0, i - window_size)
        end = min(num_nodes, i + window_size + 1)
        attention_matrix[i, start:end] = 1

    # Random entries
    # for i in range(num_nodes):
    #     for j in range(num_nodes):
    #         if random.random() < random_prob:
    #             attention_matrix[i, j] = 1  
    #             attention_matrix[j, i] = 1  

    # Global entries
    # for i in range(global_size):
    #     attention_matrix[i, :] = 1  
    #     attention_matrix[:, i] = 1    

    return attention_matrix.nonzero().t().contiguous()

