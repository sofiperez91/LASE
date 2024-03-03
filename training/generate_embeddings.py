import torch
from torch_geometric.utils import to_dense_adj
from graspologic.embed import AdjacencySpectralEmbed  

import sys
sys.path.append("../")
from models.RDPG_GD import GRDPG_GD_Armijo
from models.GLASE_unshared_normalized import gLASE 


def generate_embeddings(adj_matrix, mask, d):

    num_nodes = adj_matrix.shape[0]
    edge_index = torch.tensor(adj_matrix).nonzero().t().contiguous()
    
    ## Calculate Embeddings
    ## ASE 
    adj_matrix = to_dense_adj(edge_index.to('cpu'), max_num_nodes=num_nodes).squeeze(0)
    ase = AdjacencySpectralEmbed(n_components=d, diag_aug=True, algorithm='full')
    masked_adj = adj_matrix*mask
    x_ase = ase.fit_transform(masked_adj.numpy())
    x_ase = torch.from_numpy(x_ase)

    A = to_dense_adj(edge_index.to('cpu'), max_num_nodes=num_nodes).squeeze(0)

    u, V = torch.linalg.eig(A)

    list_q=[]
    for i in range(d):
        if u[i].numpy()>0:
            list_q.append(1)
        else:
            list_q.append(-1)
            
    q = torch.Tensor(list_q)
    Q=torch.diag(q)

    x_grdpg, cost, k  = GRDPG_GD_Armijo(x_ase, edge_index, Q, mask.nonzero().t().contiguous())
    x_grdpg = x_grdpg.detach()


    gd_steps = 20
    lr = 1e-2
    device = 'cuda'
    model = gLASE(d,d, gd_steps)
    model.to(device)


    epochs = 300

    ## Initialization
    for step in range(gd_steps):
        model.gd[step].lin1.weight.data = (torch.eye(d,d)*lr).to(device)#torch.nn.init.xavier_uniform_(model.gd[step].lin1.weight)*lr
        model.gd[step].lin2.weight.data = (torch.eye(d,d)*lr).to(device)#torch.nn.init.xavier_uniform_(model.gd[step].lin2.weight)*lr
        

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Define ATT mask
    edge_index_2 = torch.ones([num_nodes,num_nodes],).nonzero().t().contiguous().to(device)
    mask = mask.to(device)
    x_ase = x_ase.to(device)
    edge_index = edge_index.to(device)
    Q = Q.to(device)

    for epoch in range(epochs):
        # Train
        model.train()

        optimizer.zero_grad()
        out = model(x_ase, edge_index, edge_index_2, Q, mask.nonzero().t().contiguous())
        loss = torch.norm((out@Q@out.T - to_dense_adj(edge_index, max_num_nodes=num_nodes).squeeze(0))*mask)
        loss.backward() 
        optimizer.step() 
            
    loss = torch.norm((out@Q@out.T - to_dense_adj(edge_index, max_num_nodes=num_nodes).squeeze(0))*mask)

    x_glase = out.detach().to('cpu')
    x_ase = x_ase.to('cpu')


    masked_edge_index = masked_adj.nonzero().t().contiguous()
    
    return x_ase, x_grdpg, x_glase, masked_edge_index, edge_index_2, Q