import torch 

def get_x_init(num_nodes, d, alpha, beta, phi_min, phi_max):
    dim = d  # target dimensionality
    r = 1  # assuming a unit hypersphere
    
    # Generate num_nodes angles for each dimension, within specified ranges
    angles = [torch.linspace(phi_min, phi_max, num_nodes) for _ in range(dim - 1)]
    angles.insert(0, torch.linspace(alpha, beta, num_nodes))  # Insert theta angles at the beginning
    
    coords = torch.zeros((num_nodes, dim))
    
    # Compute coordinates in a loop
    for i in range(dim):
        coord = r
        for j in range(i):
            coord *= torch.sin(angles[j])
        if i < dim - 1:
            coord *= torch.cos(angles[i])
        coords[:, i] = coord
        
    indices = torch.randperm(coords.size(0))
    permuted_coords = coords[indices]
    
    return permuted_coords