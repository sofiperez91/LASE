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


from training.run_link_prediction import link_prediction_onu


votes_df = load_un_dataset('data/UNVotes-1.csv', unknown_votes=True)

print("ORIGINAL FEATURES")

for year in votes_df.year.unique():
    print(f'{year},', end =" ")
    result = link_prediction_onu(year = year, d=4, unknown_countries = 30, mask_threshold = 0.9, random_features=False)
    

print("RANDOM FEATURES")

for year in votes_df.year.unique():
    print(f'{year},', end =" ")
    result = link_prediction_onu(year = year, d=4, unknown_countries = 30, mask_threshold = 0.9, random_features=True)