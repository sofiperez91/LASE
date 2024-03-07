import sys
sys.path.append("../")

import warnings
warnings.filterwarnings("ignore")

import math
import torch
import torch_geometric as pyg
import numpy as np
from models.LASE_unshared_normalized import LASE 
from models.RDPG_GD import RDPG_GD_Armijo, coordinate_descent
from models.SVD_truncate import embed_scipy
from models.bigbird_attention import big_bird_attention
from graspologic.embed import AdjacencySpectralEmbed 
from torch_geometric.utils import to_dense_adj, stochastic_blockmodel_graph, erdos_renyi_graph
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from numpy import random
import time
from networkx import watts_strogatz_graph
from torch_geometric.utils.convert import from_networkx
from training.get_init import get_x_init
from scipy.stats import sem
import timeit

import argparse

parser = argparse.ArgumentParser(description='model')
parser.add_argument('--model', type=str, default='svd', help='[svd, gd, lase_full, lase_er05, lase_ws03, lase_ws01, lase_bb03, lase_bb01]')

args = parser.parse_args()
model = args.model

device = 'cuda'
d = 3
p = [
     [0.9, 0.2, 0.1],
     [0.2, 0.6, 0.2],
     [0.1, 0.2, 0.7]
]


# nodes = np.arange(5,55,5)*100
num_nodes =240
nodes = [num_nodes*2*i for i in range(1,11)]
n_array = [[120*2*i, 80*2*i, 40*2*i] for i in range(1,11)]

def measure_execution_time(func, *args, loops=10, repetitions=10):
    times = np.zeros(repetitions)
    for i in range(repetitions):
        loop_time = timeit.timeit(lambda: func(*args), number=loops)
        times[i] = loop_time / loops
    best_time = np.mean(times)
    mean_time = np.mean(times)
    stderr = sem(times)  
    return best_time, mean_time, stderr


if model == 'svd':

    svd_exec_time = np.zeros((len(nodes),2))

    for i, num_nodes in enumerate(nodes):
        print(num_nodes)
        n = n_array[i]
        edge_index = stochastic_blockmodel_graph(n, p)
        best_time, mean_time, std_time = measure_execution_time(embed_scipy, edge_index, d, device)
        svd_exec_time[i]=[mean_time, std_time]
        print(f"Best execution time over 10 repetitions: {best_time:.8f} seconds")
        
    with open('./results/svd_performance.pkl', 'wb') as f:
        pickle.dump(svd_exec_time, f)
    print(svd_exec_time)

if model == 'cgd':

    cgd_exec_time = np.zeros((len(nodes),2))

    for i, num_nodes in enumerate(nodes):
        print(num_nodes)
        n = n_array[i]
        edge_index = stochastic_blockmodel_graph(n, p).to(device)
        mask = (torch.ones([num_nodes,num_nodes],)-torch.eye(num_nodes)).nonzero().t().contiguous().to(device)
        best_time, mean_time, std_time = measure_execution_time(coordinate_descent, edge_index, mask, d, device)
        cgd_exec_time[i]=[mean_time, std_time]
        print(f"Best execution time over 5 repetitions: {best_time:.8f} seconds")

    with open('./results/coord_gd_performance.pkl', 'wb') as f:
        pickle.dump(cgd_exec_time, f)
        

if model == 'lase_er05':
    gd_steps = 5
    lase_ER05 = LASE(d, d, gd_steps)
    lase_ER05.load_state_dict(torch.load('../saved_models/lase_unshared_d3_normalized_unbalanced_rand_v2_ER05.pt'))
    lase_ER05.to(device)
    lase_ER05.eval()

    lase_ER05_exec_time = np.zeros((len(nodes),2))

    for i, num_nodes in enumerate(nodes):
        print(num_nodes)
        n = n_array[i]
        x = torch.rand((num_nodes, d)).to(device)
        edge_index = stochastic_blockmodel_graph(n, p).to(device)
        ER05 = erdos_renyi_graph(num_nodes, 0.5, directed=False).to(device)
        mask = (torch.ones([num_nodes,num_nodes],)-torch.eye(num_nodes)).nonzero().t().contiguous().to(device)
        best_time, mean_time, std_time = measure_execution_time(lase_ER05, x, edge_index, ER05, mask)
        lase_ER05_exec_time[i]=[mean_time, std_time]
        print(f"Best execution time over 5 repetitions: {best_time:.8f} seconds")
        
    with open('./results/lase_ER05_performance.pkl', 'wb') as f:
        pickle.dump(lase_ER05_exec_time, f)
        

if model == 'lase_ws03':
    gd_steps = 5
    lase_WS_03_01 = LASE(d, d, gd_steps)
    lase_WS_03_01.load_state_dict(torch.load('../saved_models/lase_unshared_d3_normalized_unbalanced_rand_v2_WS03.pt'))
    lase_WS_03_01.to(device)
    lase_WS_03_01.eval()
    
    lase_WS03_exec_time = np.zeros((len(nodes),2))

    for i, num_nodes in enumerate(nodes):
        print(num_nodes)
        n = n_array[i]
        x = torch.rand((num_nodes, d)).to(device)
        edge_index = stochastic_blockmodel_graph(n, p).to(device)    
        WS_0301 = from_networkx(watts_strogatz_graph(num_nodes, int(num_nodes*0.3), 0.1, seed=None)).edge_index.to(device)
        mask = (torch.ones([num_nodes,num_nodes],)-torch.eye(num_nodes)).nonzero().t().contiguous().to(device)
        best_time, mean_time, std_time = measure_execution_time(lase_WS_03_01, x, edge_index, WS_0301, mask)
        lase_WS03_exec_time[i]=[mean_time, std_time]
        print(f"Best execution time over 5 repetitions: {best_time:.8f} seconds")
        
    with open('./results/lase_WS03_performance.pkl', 'wb') as f:
        pickle.dump(lase_WS03_exec_time, f)       
        
if model == 'lase_ws01':
    gd_steps = 5
    lase_WS_01_01 = LASE(d, d, gd_steps)
    lase_WS_01_01.load_state_dict(torch.load('../saved_models/lase_unshared_d3_normalized_unbalanced_rand_v2_WS01.pt'))
    lase_WS_01_01.to(device)
    lase_WS_01_01.eval()
    
    lase_WS01_exec_time = np.zeros((len(nodes),2))

    for i, num_nodes in enumerate(nodes):
        print(num_nodes)
        n = n_array[i]
        x = torch.rand((num_nodes, d)).to(device)
        edge_index = stochastic_blockmodel_graph(n, p).to(device)    
        WS_0101 = from_networkx(watts_strogatz_graph(num_nodes, int(num_nodes*0.1), 0.1, seed=None)).edge_index.to(device)
        mask = (torch.ones([num_nodes,num_nodes],)-torch.eye(num_nodes)).nonzero().t().contiguous().to(device)
        best_time, mean_time, std_time = measure_execution_time(lase_WS_01_01, x, edge_index, WS_0101, mask)
        lase_WS01_exec_time[i]=[mean_time, std_time]
        print(f"Best execution time over 5 repetitions: {best_time:.8f} seconds")
        
    with open('./results/lase_WS01_performance.pkl', 'wb') as f:
        pickle.dump(lase_WS01_exec_time, f)    
    
if model == 'lase_bb03':
    gd_steps = 5
    lase_BB_03_01 = LASE(d, d, gd_steps)
    lase_BB_03_01.load_state_dict(torch.load('../saved_models/lase_unshared_d3_normalized_unbalanced_rand_v2_BB03.pt'))
    lase_BB_03_01.to(device)
    lase_BB_03_01.eval()
    
    lase_BB03_exec_time = np.zeros((len(nodes),2))

    for i, num_nodes in enumerate(nodes):
        print(num_nodes)
        n = n_array[i]
        x = torch.rand((num_nodes, d)).to(device)
        edge_index = stochastic_blockmodel_graph(n, p).to(device)
        BB_0301 = from_networkx(watts_strogatz_graph(num_nodes, int(num_nodes*0.125), 0.1, seed=None)).edge_index.to(device)
        mask = (torch.ones([num_nodes,num_nodes],)-torch.eye(num_nodes)).nonzero().t().contiguous().to(device)
        best_time, mean_time, std_time = measure_execution_time(lase_BB_03_01, x, edge_index, BB_0301, mask)
        lase_BB03_exec_time[i]=[mean_time, std_time]
        print(f"Best execution time over 5 repetitions: {best_time:.8f} seconds")
        
    with open('./results/lase_BB03_performance.pkl', 'wb') as f:
        pickle.dump(lase_BB03_exec_time, f)
        
if model == 'lase_bb01':
    gd_steps = 5
    lase_BB_01_01 = LASE(d, d, gd_steps)
    lase_BB_01_01.load_state_dict(torch.load('../saved_models/lase_unshared_d3_normalized_unbalanced_rand_v2_BB01.pt'))
    lase_BB_01_01.to(device)
    lase_BB_01_01.eval()
    
    lase_BB01_exec_time = np.zeros((len(nodes),2))

    for i, num_nodes in enumerate(nodes):
        print(num_nodes)
        n = n_array[i]
        x = torch.rand((num_nodes, d)).to(device)
        edge_index = stochastic_blockmodel_graph(n, p).to(device)
        BB_0101 = from_networkx(watts_strogatz_graph(num_nodes, int(num_nodes*0.025), 0.05, seed=None)).edge_index.to(device)
        mask = (torch.ones([num_nodes,num_nodes],)-torch.eye(num_nodes)).nonzero().t().contiguous().to(device)
        best_time, mean_time, std_time = measure_execution_time(lase_BB_01_01, x, edge_index, BB_0101, mask)
        lase_BB01_exec_time[i]=[mean_time, std_time]
        print(f"Best execution time over 5 repetitions: {best_time:.8f} seconds")
        
    with open('./results/lase_BB01_performance.pkl', 'wb') as f:
        pickle.dump(lase_BB01_exec_time, f)

if model == 'lase_full':
    gd_steps = 5
    lase_full = LASE(d, d, gd_steps)
    lase_full.load_state_dict(torch.load('../saved_models/lase_unshared_d3_normalized_unbalanced_rand_v2.pt'))
    lase_full.to(device)
    lase_full.eval()

    lase_full_exec_time = np.zeros((len(nodes),2))

    for i, num_nodes in enumerate(nodes):
        print(num_nodes)
        n = n_array[i]
        x = torch.rand((num_nodes, d)).to(device)
        edge_index = stochastic_blockmodel_graph(n, p).to(device)
        edge_index_2 = torch.ones([num_nodes,num_nodes],).nonzero().t().contiguous().to(device)
        mask = (torch.ones([num_nodes,num_nodes],)-torch.eye(num_nodes)).nonzero().t().contiguous().to(device)
        best_time, mean_time, std_time = measure_execution_time(lase_full, x, edge_index, edge_index_2, mask)
        lase_full_exec_time[i]=[mean_time, std_time]
        print(f"Best execution time over 5 repetitions: {best_time:.8f} seconds")
        
    with open('./results/lase_full_performance.pkl', 'wb') as f:
        pickle.dump(lase_full_exec_time, f)
        
        
