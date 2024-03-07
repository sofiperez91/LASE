import sys 
sys.path.append("../")

import warnings
warnings.filterwarnings("ignore")

from models.onu_fuctions import load_un_dataset, create_un_graphs, process_un_graph
import torch 

import torch
from torch_geometric.utils import to_dense_adj
from graspologic.embed import AdjacencySpectralEmbed  


from models.RDPG_GD import GRDPG_GD_Armijo
from models.GLASE_unshared_normalized import gLASE 
from torch_geometric.data import Data

import copy
import random

from training.run_link_prediction import link_prediction_onu_2, link_prediction_onu_Transformer, link_prediction_onu_GraphTransformer


votes_df = load_un_dataset('data/UNVotes-1.csv', unknown_votes=True)


# print("ORIGINAL FEATURES")

for year in votes_df.year.unique():
    countries = random.sample(votes_df[votes_df.year == year]['Country'].unique().tolist(), 6)
    # countries = ['ISR', 'GRB', 'NDL', 'CUB', 'TUR', 'VNM']
    print(f'{year};', end = " ")
    print(f"{countries};", end = " ") 
    # acc_gcn_array, acc_ase_array, acc_grdpg_array, acc_glase_array, acc_glase_e2e_array, selected_resolutions = link_prediction_onu_GraphTransformer(year = year, d=4, unknown_countries = countries, mask_threshold = 0.3, random_features=False, iter=10, with_e2e=True)
    acc_gcn_array, acc_ase_array, acc_grdpg_array, acc_glase_array, acc_glase_e2e_array, selected_resolutions = link_prediction_onu_Transformer(year = year, d=4, unknown_countries = countries, mask_threshold = 0.3, random_features=False, iter=10, with_e2e=True)
    # acc_gcn_array, acc_ase_array, acc_grdpg_array, acc_glase_array, acc_glase_e2e_array, selected_resolutions = link_prediction_onu_2(year = year, d=4, unknown_countries = countries, mask_threshold = 0.3, random_features=False, iter=10, with_e2e=True)
    # acc_gcn_array, acc_ase_array, acc_grdpg_array, acc_glase_array, selected_resolutions = link_prediction_onu_2(year = year, d=4, unknown_countries = countries, mask_threshold = 0.3, random_features=False, iter=10)
    print(f"{selected_resolutions};", end = " ") 
    print(f"{acc_gcn_array};", end = " ") 
    print(f"{acc_ase_array};", end = " ") 
    print(f"{acc_grdpg_array};", end = " ") 
    # print(f"{acc_glase_array}")  
    print(f"{acc_glase_array};", end = " ") 
    print(f"{acc_glase_e2e_array}") 


print("RANDOM FEATURES")

for year in votes_df.year.unique():
    countries = random.sample(votes_df[votes_df.year == year]['Country'].unique().tolist(), 6)
    # countries = ['ISR', 'GRB', 'NDL', 'CUB', 'TUR', 'VNM']
    print(f'{year};', end = " ")
    print(f"{countries};", end = " ") 
    # result = link_prediction_onu_2(year = year, d=4, unknown_countries = 30, mask_threshold = 0.9, random_features=True)
    # acc_gcn_array, acc_ase_array, acc_grdpg_array, acc_glase_array, acc_glase_e2e_array, selected_resolutions = link_prediction_onu_GraphTransformer(year = year, d=4, unknown_countries = countries, mask_threshold = 0.3, random_features=True, iter=10, with_e2e=True)    
    acc_gcn_array, acc_ase_array, acc_grdpg_array, acc_glase_array, acc_glase_e2e_array, selected_resolutions = link_prediction_onu_Transformer(year = year, d=4, unknown_countries = countries, mask_threshold = 0.3, random_features=True, iter=10, with_e2e=True)    
    # acc_gcn_array, acc_ase_array, acc_grdpg_array, acc_glase_array, acc_glase_e2e_array, selected_resolutions = link_prediction_onu_2(year = year, d=4, unknown_countries = countries, mask_threshold = 0.3, random_features=True, iter=10, with_e2e=True)
    # acc_gcn_array, acc_ase_array, acc_grdpg_array, acc_glase_array, selected_resolutions = link_prediction_onu_2(year = year, d=4, unknown_countries = countries, mask_threshold = 0.3, random_features=True, iter=10)
    print(f"{selected_resolutions};", end = " ") 
    print(f"{acc_gcn_array};", end = " ") 
    print(f"{acc_ase_array};", end = " ") 
    print(f"{acc_grdpg_array};", end = " ") 
    # print(f"{acc_glase_array}") 
    print(f"{acc_glase_array};", end = " ") 
    print(f"{acc_glase_e2e_array}") 

