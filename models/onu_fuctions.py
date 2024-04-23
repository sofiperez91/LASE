import os
import requests
from tqdm import tqdm
import pandas as pd
import pycountry_convert as pc
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from cycler import cycler
import torch 
import random
from typing import List

resolutions_issues = {'me': 'Palestinian conflict', 
                      'nu': 'Nuclear weapons and nuclear material', 
                      'di': 'Arms control and disarmament',
                      'co': 'Colonialism',
                      'hr': 'Human rights',
                      'ec': 'Economic Development',
                      'N/A': 'Not specified'}

resolutions_issues_color = {'me': 'salmon', 
                            'nu': 'yellow', 
                            'di': 'teal',
                            'co': 'orchid',
                            'hr': 'navy',
                            'ec': 'orange',
                            'N/A': 'black'}

continents_colors = {'North America': 'yellow',
                     'South America': 'forestgreen',
                     'Europe': 'royalblue',
                     'Africa': 'plum',
                     'Asia': 'darkorange',
                     'Oceania': 'firebrick'}

cycler_colors = ['royalblue','firebrick','forestgreen','olive']


def load_un_dataset(un_data_path, initial_year=1946, final_year=2018, remove_nonmembers=True, remove_nonpresent=False, unknown_votes=False):
    
    if os.path.isdir(os.path.dirname(un_data_path)):
        if not os.path.exists(un_data_path):
            download_un_dataset(un_data_path)
    else:
        raise Exception("Provided path for UN dataset is not reachable")
    
    # Load data    
    votes_df = pd.read_csv(un_data_path, low_memory=False, encoding='latin-1', index_col=0)
    # Keep only desired years
    votes_df = votes_df[votes_df.year>=initial_year]
    votes_df = votes_df[votes_df.year<=final_year]
    
    if remove_nonmembers:
        # Remove votes by nonmembers
        votes_df = votes_df[votes_df.vote!=9]
    
    if remove_nonpresent:
        # Remove votes by nonmembers
        votes_df = votes_df[votes_df.vote!=8]
        
    # Edges in graph represent an affirmative vote
    votes_df['weight'] = (votes_df.vote==1)
    
    if unknown_votes:
        # Voters preference is assumed unknown if it is an abstention or voter is not present
        votes_df['unknown'] =  (votes_df.vote==2) | (votes_df.vote==8)
    
    
    continents_dict = get_continents_dict(votes_df)
    votes_df['Continent'] = votes_df['Country'].map(continents_dict)
    aux = pd.get_dummies(votes_df['Continent']).astype(int)
    votes_df['continent_vector']=aux[['Africa','Asia','Europe','North America','Oceania','South America']].apply(lambda row: np.array(row), axis=1)
    votes_df['continent_vector'] = votes_df['continent_vector'].apply(lambda arr: np.concatenate((arr, np.zeros(6))))
    
    votes_df['res_features'] = votes_df[['me','nu','di','co','hr','ec']].apply(lambda row: np.array(row), axis=1)
    votes_df['res_features'] = votes_df['res_features'].apply(lambda arr: np.concatenate((np.zeros(6), arr)))
    

    return votes_df

        
def download_un_dataset(filename='UNVotes-1.csv', data_url='https://dataverse.harvard.edu/api/access/datafile/6358426'):
    # Code from https://gist.github.com/yanqd0/c13ed29e29432e3cf3e7c38467f42f51
    response = requests.get(data_url, stream=True)
    total_size = int(response.headers.get('content-length',0))
    with open(filename, "wb") as f, tqdm(desc='Downloading UN dataset', total=total_size, unit='B', unit_divisor=1024, unit_scale=True) as pbar:
        for un_data in response.iter_content(chunk_size=1024):
            size = f.write(un_data)
            pbar.update(size)
            
def get_continents_dict(votes_df):
    continents_dict = {}
    countries = votes_df.Country.unique()
    for country in countries: 
        try:
            continent_code = pc.country_alpha2_to_continent_code(pc.country_alpha3_to_country_alpha2(country))
            continents_dict[country] = pc.convert_continent_code_to_continent_name(continent_code)
        except:
            continue
            # print(pais)
            
    continents_dict['DDR'] = 'Europe'
    continents_dict['CSK'] = 'Europe'
    continents_dict['YUG'] = 'Europe'
    continents_dict['EAZ'] = 'Africa'
    continents_dict['YAR'] = 'Asia'
    continents_dict['TLS'] = 'Asia'
    
    return continents_dict

def get_countries_name_conversion_dict(votes_df):
    countries = votes_df.Countryname.unique()
    conversion_dict = {}
    for country in countries: 
        conversion_dict[country] = votes_df[votes_df.Countryname==country].Country.unique()[0]
        
    return conversion_dict

def create_un_graphs(votes_df):
    
    continents_dict = get_continents_dict(votes_df)
    conversion_dict = get_countries_name_conversion_dict(votes_df)
    
    all_graphs = {}
    
    edge_attr = ['weight', 'unknown'] if 'unknown' in votes_df.columns else 'weight'    
    
        
    g = nx.from_pandas_edgelist(votes_df,source='Countryname',target='resid',edge_attr=edge_attr,create_using=nx.DiGraph())
    if g.number_of_edges()>0:
        
        countries_list = votes_df.Countryname.unique()
        
        # Add country's code and continent as graph attributes
        countries_codes = {}
        countries_continents = {}
        nodes_colors = {}
        node_types = {}
        country_features = {}
        for country in countries_list:
            countries_codes[country] = conversion_dict[country]
            countries_continents[country] = continents_dict[conversion_dict[country]]
            nodes_colors[country] = continents_colors[countries_continents[country]]
            node_types[country] = "country"
            country_features[country] = votes_df[votes_df.Countryname==country]['continent_vector'].mean()
            
        nx.set_node_attributes(g, countries_codes, name='country code')
        nx.set_node_attributes(g, countries_continents, name='continent')
        nx.set_node_attributes(g, country_features, name='country_features') 
        
        # Add resolution's issue as graph attribute
        resolutions_list = votes_df.resid.unique()
        resolutions_issues_dict = {}
        important_resolutions_dict = {}
        resolutions_features = {}

        
        for resolution_id in resolutions_list:
            df_votes_sum = votes_df[votes_df.resid==resolution_id][['me','nu','di','co','hr','ec']].sum()
            if df_votes_sum.max()>0:
                resolutions_issues_dict[resolution_id] = df_votes_sum.idxmax()
            else:
                resolutions_issues_dict[resolution_id] = 'N/A'
                
            nodes_colors[resolution_id] = resolutions_issues_color[resolutions_issues_dict[resolution_id]]
            resolutions_features[resolution_id] = votes_df[votes_df.resid==resolution_id]['res_features'].mean()
            node_types[resolution_id] = "resolution"
            
            important_vote = votes_df[votes_df.resid==resolution_id]['importantvote'].max()
            if important_vote > 0:
                important_resolutions_dict[resolution_id] = True
            else:
                important_resolutions_dict[resolution_id] = False
        
        nx.set_node_attributes(g, resolutions_issues_dict,name='issue code')
        nx.set_node_attributes(g, nodes_colors, name='color')
        nx.set_node_attributes(g, important_resolutions_dict, name='important vote')
        nx.set_node_attributes(g, node_types, name='type') 
        nx.set_node_attributes(g, resolutions_features, name='res_features') 
        
            
    return g

def process_un_graph(graph, countries: int = 30, mask_threshold: float = 0.7):
    G = graph.to_undirected()
    
    # rename nodes
    mapping = {node: i for i, node in enumerate(G.nodes())}
    G_ = nx.relabel_nodes(G, mapping)

    country_indexes = []
    res_indexes = []
    __features = []

    for node, data in G_.nodes(data=True):
        if data["type"] == "country":
            country_indexes.append(node)
            __features.append(data['country_features'])
        else:
            res_indexes.append(node)
            __features.append(data['res_features'])


    unknown_edges = []

    for u, v, data in G_.edges(data=True):
        if data['unknown']:
            unknown_edges.append((u,v))
            unknown_edges.append((v,u))
        

    adj_matrix = nx.adjacency_matrix(G_).todense().astype(int)
    
    _features = np.array(__features)
    features = torch.tensor(_features)
    
    ## Create mask with unknown country votes

    num_nodes = adj_matrix.shape[0]

    mask = torch.ones([num_nodes,num_nodes]).squeeze(0)
    mask_unknown = torch.ones([num_nodes,num_nodes]).squeeze(0)

    random.seed(42)
    missing_countries = random.sample(country_indexes, countries)

    for i in missing_countries:
        votos = (torch.rand(1, num_nodes) < mask_threshold).int()
        mask[i,:] = votos
        mask[:,i] = votos
        
    for edge in unknown_edges:
        u, v = edge
        mask_unknown[u,v] = 0
        mask_unknown[v,u] = 0
        
        
    mask = mask*mask_unknown # Me aseguro que los unknown formen parte de la mask
    inverted_mask_matrix = (torch.ones([num_nodes,num_nodes]).squeeze(0) - mask)*mask_unknown # Me aseguro no predecir sobre los unknown
    
    
    return adj_matrix, country_indexes, res_indexes, features, missing_countries, mask, inverted_mask_matrix


def process_un_graph_2(graph, mask_countries: List[str], mask_threshold: float = 0.3):
    G = graph.to_undirected()
    
    # rename nodes
    mapping = {node: i for i, node in enumerate(G.nodes())}
    G_ = nx.relabel_nodes(G, mapping)

    country_indexes = []
    res_indexes = []
    __features = []

    for node, data in G_.nodes(data=True):
        if data["type"] == "country":
            country_indexes.append(node)
            __features.append(data['country_features'])
        else:
            res_indexes.append(node)
            __features.append(data['res_features'])


    unknown_edges = []

    for u, v, data in G_.edges(data=True):
        if data['unknown']:
            unknown_edges.append((u,v))
            unknown_edges.append((v,u))
        

    adj_matrix = nx.adjacency_matrix(G_).todense().astype(int)
    
    _features = np.array(__features)
    features = torch.tensor(_features)
    
    ## Create mask with unknown country votes

    num_nodes = adj_matrix.shape[0]

    mask = torch.ones([num_nodes,num_nodes]).squeeze(0)
    mask_unknown = torch.ones([num_nodes,num_nodes]).squeeze(0)

    random.seed(42)
    selected_resolutions = random.sample(res_indexes, int(len(res_indexes)*mask_threshold))
    
    mask_nodes = []
    for node, data in G_.nodes(data=True):
        if data['type'] == 'country':
            if data['country code'] in mask_countries:
                mask_nodes.append(node)

    
    for i in mask_nodes:
        for j in selected_resolutions:
            mask[i,j] = 0
            mask[j,i] = 0
        
    for edge in unknown_edges:
        u, v = edge
        mask_unknown[u,v] = 0
        mask_unknown[v,u] = 0
        
        
    mask = mask*mask_unknown # Me aseguro que los unknown formen parte de la mask
    inverted_mask_matrix = (torch.ones([num_nodes,num_nodes]).squeeze(0) - mask)*mask_unknown # Me aseguro no predecir sobre los unknown
    
    
    return adj_matrix, country_indexes, res_indexes, unknown_edges, features, mask_nodes, mask, selected_resolutions, inverted_mask_matrix, mask_unknown