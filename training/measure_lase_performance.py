import sys
sys.path.append("../")

import warnings
warnings.filterwarnings("ignore")

import torch
import json

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
parser.add_argument('--model', type=str, default='svd', help='[svd, cgd, LASE_FULL, LASE_ER05, LASE_WS03, LASE_BB03]')
parser.add_argument('--dataset', type=str, default='sbm3_unbalanced_positive_v2')

args = parser.parse_args()
model = args.model
dataset = args.dataset

# Load the config file
with open('../data/data_config.json', 'r') as file:
    config = json.load(file)

device = 'cuda'
d = config[dataset]['d']
p = config[dataset]['p']
n = config[dataset]['n']
num_nodes = np.sum(n)

if d == 3:
    nodes = [num_nodes*2*i for i in range(1,11)]
    n_array = [[120*2*i, 80*2*i, 40*2*i] for i in range(1,11)]
elif d == 10:
    nodes = [num_nodes*i for i in range(1,8)]
    nodes.append(4125)
    n_array = [[80*i, 50*i, 80*i, 40*i, 50*i, 60*i, 50*i, 50*i, 40*i, 50*i] for i in range(1,8)]
    n_array.append([600, 375, 600, 300, 375, 450, 375, 375, 300, 375])


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
        
    with open(f'./results/svd_performance_{dataset}.pkl', 'wb') as f:
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

    with open(f'./results/coord_gd_performance_{dataset}.pkl', 'wb') as f:
        pickle.dump(cgd_exec_time, f)
        

if model == 'LASE_ER05':
    gd_steps = 5
    lase_ER05 = LASE(d, d, gd_steps)
    lase_ER05.load_state_dict(torch.load(f'../saved_models/lase_{dataset}_d{d}_normalized_random_{gd_steps}steps_ER05.pt'))
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
        
    with open(f'./results/lase_ER05_performance_{dataset}.pkl', 'wb') as f:
        pickle.dump(lase_ER05_exec_time, f)
        

if model == 'LASE_WS03':
    gd_steps = 5
    lase_WS_03_01 = LASE(d, d, gd_steps)
    lase_WS_03_01.load_state_dict(torch.load(f'../saved_models/lase_{dataset}_d{d}_normalized_random_{gd_steps}steps_WS03'))
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
        
    with open(f'./results/lase_WS03_performance_{dataset}.pkl', 'wb') as f:
        pickle.dump(lase_WS03_exec_time, f)       
     
    
if model == 'LASE_BB03':
    gd_steps = 5
    lase_BB_03_01 = LASE(d, d, gd_steps)
    lase_BB_03_01.load_state_dict(torch.load(f'../saved_models/lase_{dataset}_d{d}_normalized_random_{gd_steps}steps_BB03'))
    lase_BB_03_01.to(device)
    lase_BB_03_01.eval()
    
    lase_BB03_exec_time = np.zeros((len(nodes),2))

    for i, num_nodes in enumerate(nodes):
        print(num_nodes)
        n = n_array[i]
        x = torch.rand((num_nodes, d)).to(device)
        edge_index = stochastic_blockmodel_graph(n, p).to(device)
        BB_03_01 = big_bird_attention(int(num_nodes*0.125), 0.1, num_nodes).to(device)
        mask = (torch.ones([num_nodes,num_nodes],)-torch.eye(num_nodes)).nonzero().t().contiguous().to(device)
        best_time, mean_time, std_time = measure_execution_time(lase_BB_03_01, x, edge_index, BB_03_01, mask)
        lase_BB03_exec_time[i]=[mean_time, std_time]
        print(f"Best execution time over 5 repetitions: {best_time:.8f} seconds")
        
    with open(f'./results/lase_BB03_performance_{dataset}.pkl', 'wb') as f:
        pickle.dump(lase_BB03_exec_time, f)
        

if model == 'lase_full':
    gd_steps = 5
    lase_full = LASE(d, d, gd_steps)
    lase_full.load_state_dict(torch.load(f'../saved_models/lase_{dataset}_d{d}_normalized_random_{gd_steps}steps_FULL'))
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
        # print(lase_full_exec_time[i]) 
    with open(f'./results/lase_full_performance_{dataset}.pkl', 'wb') as f:
        pickle.dump(lase_full_exec_time, f)
        
        
