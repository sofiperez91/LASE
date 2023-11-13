import torch_geometric as pyg
import torch
from torch_geometric.utils import to_dense_adj, stochastic_blockmodel_graph
import numpy as np
import scipy



def RDPG_cost(x, A, mask, device = 'cpu'):
    M = to_dense_adj(mask).squeeze(0)
    return 0.5*torch.norm((x@x.T - A)*M)**2

def RPDG_gradient(x, A, mask, device = 'cpu'):
    M = to_dense_adj(mask).squeeze(0)
    return 4*(M*(x@x.T-A))@x

def GRDPG_cost(x, A, Q, mask):
    M = to_dense_adj(mask).squeeze(0)
    return 0.5*torch.norm((A - x@Q@x.T)*M)**2

def GRDPG_gradient(x, A, Q, mask):
    M = to_dense_adj(mask).squeeze(0)
    return 2*(((M.T+M)*(x@Q@x.T)) - ((M+M.T)*A))@x@Q

# def solve_linear_system(A,b,xx):
#     try:
#         result = scipy.linalg.solve(A,b)
#     except:
#         result = scipy.sparse.linalg.minres(A,b,xx)[0]    
#     return result


def solve_linear_system(A, b, xx):
    try:
        result = torch.linalg.solve(A, b)
    except:
        result = torch.linalg.cg(A, b, x0=xx)[0]

    return result

def RDPG_GD_fixstep(x, edge_index, mask, tol=1e-3, alpha=0.001, max_iter=10000):

    xd = x
    A = to_dense_adj(edge_index, max_num_nodes=x.shape[0]).squeeze()
    d = -RPDG_gradient(x, A, mask)
    k=0

    while (torch.norm(d) > tol) & (k<max_iter):

        xd = xd+alpha*d
        d = -RPDG_gradient(xd, A, mask)
        k=k+1

    cost = RDPG_cost(xd, A, mask)

    return xd, cost, k 


def RDPG_GD_Armijo(x, edge_index, M, tol=1e-3, max_iter=10000):

    A = to_dense_adj(edge_index).squeeze()
    b=0.3; sigma=0.1 # Armijo parameters
    t = 0.1
    xd=x
    k=0
    last_jump=1
    d = -RPDG_gradient(xd, A, M)
    tol = tol*(torch.norm(d))

    while (torch.norm(d) > tol) & (last_jump > 1e-16) & (k<max_iter):

        # Armijo 
        while (RDPG_cost(xd+t*d,A, M) > RDPG_cost(xd,A, M) - sigma*t*torch.norm(d)**2):
            t=b*t
            
        xd = xd+t*d
        last_jump = sigma*t*torch.norm(d)**2
        t=t/(b)
        k=k+1
        d = -RPDG_gradient(xd, A, M)
    
    return xd, RDPG_cost(xd, A, M), k


def coordinate_descent(edge_index, M, d, device='cpu', X=None, tol=1e-5):
    A = to_dense_adj(edge_index).squeeze(0).to(device)
    num_nodes=A.shape[0]

    if X is None:
        X = torch.rand((num_nodes, d)).to(device)
    else:
        X = X.clone().to(device)

    R = X.T@X
    fold = -1

    while (abs((fold - RDPG_cost(X, A, M, device))/fold) >tol):
        fold = RDPG_cost(X, A, M, device)
        for i in range(num_nodes):
            k=X[i,:].unsqueeze(0)
            R -= k.T@k
            X[i,:] = torch.tensor(solve_linear_system(R,(A[i,:]@X).T,X[i,:]), device=device)
            k=X[i,:].unsqueeze(0)
            R += k.T@k

    return X


def GRDPG_GD_Armijo(x, edge_index, Q, M, max_iter=100, tol=1e-3, b=0.3, sigma=0.1, t=0.1, verbose=False):
    A = to_dense_adj(edge_index).squeeze()
    b=0.3; sigma=0.1 # Armijo parameters
    t = 0.1
    xd=x
    k=0
    last_jump=1
    d = -GRDPG_gradient(xd, A, Q, M)
    tol = tol*(torch.norm(d))

    total_iter_steps = 0
    steps = []
    while (torch.norm(d) > tol) & (last_jump > 1e-16) & (k<max_iter):

        # Armijo 
        while (GRDPG_cost(xd+t*d,A,Q, M) > GRDPG_cost(xd,A,Q, M) - sigma*t*torch.norm(d)**2):
            t=b*t
            total_iter_steps+=1
            
        steps.append(t)    
        xd = xd+t*d
        last_jump = sigma*t*torch.norm(d)**2
        t=t/(b)
        k=k+1
        d = -GRDPG_gradient(xd, A, Q, M)
    
    if verbose:
        return xd, GRDPG_cost(xd, A, Q, M), k, steps, total_iter_steps
    else:
        return xd, GRDPG_cost(xd, A, Q, M), k 


def GRDPG_GD_fixstep(x, edge_index, Q, M, tol=1e-3, alpha=0.001, max_iter=10000):

    xd = x
    A = to_dense_adj(edge_index, max_num_nodes=x.shape[0]).squeeze()
    d = -GRDPG_gradient(xd, A, Q, M)
    k=0

    while (torch.norm(d) > tol) & (k<max_iter):

        xd = xd+alpha*d
        d = -GRDPG_gradient(xd, A, Q, M)
        k=k+1

    cost = GRDPG_cost(xd, A, Q, M)

    return xd, cost, k 