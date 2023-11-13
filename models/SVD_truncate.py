# import numpy as np
# import scipy.linalg
# from torch_geometric.utils import to_dense_adj

# def embed_scipy(edge_index, K):
#     A = to_dense_adj(edge_index).squeeze().numpy()
#     UA, SA, VAt = scipy.sparse.linalg.svds(A,K)
#     XA = UA[:,0:K].dot(np.diag(np.sqrt(SA[0:K])))    
#     return XA

import numpy as np
import torch
import scipy.sparse
from torch_geometric.utils import to_dense_adj

def embed_scipy(edge_index, K, device='cpu'):
    A = to_dense_adj(edge_index).squeeze().to(device)
    # Convert to numpy and then to scipy sparse matrix for svds
    A_cpu = A.to("cpu").numpy()
    A_sparse = scipy.sparse.csr_matrix(A_cpu)

    # Perform sparse SVD on CPU using scipy
    UA, SA, VAt = scipy.sparse.linalg.svds(A_sparse, K)

    # Convert back to PyTorch tensors and move to GPU
    UA = torch.tensor(UA[:, :K].copy(), device=device).float()
    SA = torch.tensor(np.sqrt(SA[:K]), device=device).float()

    # Compute the final embedding
    XA = UA @ torch.diag(SA)

    return XA
