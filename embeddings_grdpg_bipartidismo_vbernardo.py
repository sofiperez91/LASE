#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 18:05:45 2020

@author: flarroca
"""

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from graspologic import simulations, embed
import ase_opt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


plt.close(fig='all')
plt.rcParams['lines.linewidth'] = 3
plt.rcParams['text.usetex'] = True
plt.rcParams['font.size'] = 28
plt.rcParams['lines.markersize'] = 15
plt.rcParams['axes.grid'] = True

#import netgraph




##############################
## Directed SBM - simula bipartidismo
##############################

plt.close(fig='all')

def grafo_senadores(nrep = 50, ndem = 50, no_presentes=[],
                    lrep = 50, ldem = 200, lmix = 40, 
                    p_rep_rep = 0.9, p_rep_dem = 0.01, p_rep_mix = 0.2, 
                    p_dem_rep = 0.1, p_dem_dem = 0.8, p_dem_mix = 0.3):
    """
    Genera un grafo con senadores democratas y republicanos. Es un SBM directivo
    con probabilidades de votacion indicadas en cada parametro. Las leyes mix 
    son leyes que tienden a votar en ambos partidos. 
    """
    
    n = [nrep, ndem, lrep, ldem, lmix]
    nodes_dict = {}
    labels_dict = {}
    markers_dict = {}
    
    for rep in range(nrep):
        nodes_dict[rep] = 0
        labels_dict[rep] = 'Party 1'
        markers_dict[rep] = 'o'
    for dem in range(ndem):
        nodes_dict[(dem+nrep)] = 9
        labels_dict[(dem+nrep)] = 'Party 2'
        markers_dict[(dem+nrep)] = 'o'
    for lr in range(lrep):
        nodes_dict[(ndem+nrep+lr)] = 3
        labels_dict[(ndem+nrep+lr)] = 'Laws party 1'
        markers_dict[(ndem+nrep+lr)] = '^'
    for ld in range(ldem):
        nodes_dict[(ndem+nrep+lrep+ld)] = 7
        labels_dict[(ndem+nrep+lrep+ld)] = 'Laws party 2'
        markers_dict[(ndem+nrep+lrep+ld)] = '^'
    for lm in range(lmix):
        nodes_dict[(ndem+nrep+lrep+ldem+lm)] = 5
        labels_dict[(ndem+nrep+lrep+ldem+lm)] = 'Mixed laws'
        markers_dict[(ndem+nrep+lrep+ldem+lm)] = '^'
    
    if no_presentes:
        for sen in no_presentes:
            markers_dict[sen] = 'P'
            labels_dict[sen] = 'Not present (' + labels_dict[sen] + ')'
            nodes_dict[sen] = nodes_dict[sen] - 1
            
    
    p = [[0, 0, p_rep_rep, p_rep_dem, p_rep_mix],
          [0, 0, p_dem_rep, p_dem_dem, p_dem_mix],
          [0, 0, 0, 0, 0], # las leyes no votan
          [0, 0, 0, 0, 0], 
          [0, 0, 0, 0, 0]]
    
    weights = simulations.sbm(n=n, p=p, directed=True)
    # gy.plot.heatmap(weights)
    g = nx.from_numpy_array(weights)
    
    nx.set_node_attributes(g, nodes_dict,'category')
    nx.set_node_attributes(g, labels_dict,'text_labels')
    #nx.set_node_attributes(g, markers_dict,'scatter_marker')
    return (g, weights)

# I generate the original graph
# Some senators did not vote on certain laws
nrep = 50
ndem = 50
num_senadores = nrep + ndem
lrep = 100
ldem = 100
lmix = 30
num_leyes = lrep + ldem + lmix
num_senadores_no_presentes_rep = 5
num_senadores_no_presentes_dem = 2
senadores_no_presentes = list(range(num_senadores_no_presentes_rep)) + list(range(ndem,ndem+num_senadores_no_presentes_dem))
(g1, weights) = grafo_senadores(nrep=nrep, ndem=ndem, lrep=lrep, ldem=ldem, lmix=lmix,no_presentes=senadores_no_presentes)

presentes = np.ones_like(weights)
proba_presente = 0.3
for sen in senadores_no_presentes:
    presentes[sen,:] = presentes[sen,:]*[np.random.rand(1,weights.shape[0])<proba_presente]
presentes = np.triu(presentes)

# I'll take that as a no
weights = weights*presentes
weights = weights + weights.T

# Embedding with graspologic
ase = embed.AdjacencySpectralEmbed(n_elbows = 2, diag_aug=True)
Xhat1 = ase.fit_transform(weights)


fig, (ax1,ax2) = plt.subplots(nrows=1, ncols=2,sharex=True,sharey=True)

dims=[1,3]
ax1.scatter(Xhat1[num_senadores_no_presentes_rep:nrep,dims[0]-1],Xhat1[num_senadores_no_presentes_rep:nrep,dims[1]-1],c='royalblue',marker='o',label='Party 1')
ax1.scatter(Xhat1[:num_senadores_no_presentes_rep,dims[0]-1],Xhat1[:num_senadores_no_presentes_rep,dims[1]-1],c='gold',marker='X',label='Not present (Party 1)')
ax1.scatter(Xhat1[nrep+num_senadores_no_presentes_dem:num_senadores,dims[0]-1],Xhat1[nrep+num_senadores_no_presentes_dem:num_senadores,dims[1]-1],c='firebrick',marker='o',label='Party 2')
ax1.scatter(Xhat1[nrep:nrep+num_senadores_no_presentes_dem,dims[0]-1],Xhat1[nrep:nrep+num_senadores_no_presentes_dem,dims[1]-1],c='limegreen',marker='X',label='Not present (Party 2)')
ax1.scatter(Xhat1[num_senadores:num_senadores+lrep,dims[0]-1],Xhat1[num_senadores:num_senadores+lrep,dims[1]-1],c='cornflowerblue',marker='^',label='Party 1 laws')
ax1.scatter(Xhat1[num_senadores+lrep:num_senadores+lrep+ldem,dims[0]-1],Xhat1[num_senadores+lrep:num_senadores+lrep+ldem,dims[1]-1],c='indianred',marker='^',label='Party 2 laws')
ax1.scatter(Xhat1[num_senadores+lrep+ldem:,dims[0]-1],Xhat1[num_senadores+lrep+ldem:,dims[1]-1],c='darkorange',marker='^',label='Bipartisan laws')


# Embedding with gradient descent, starting from the one obtained with graspologic
nt = Xhat1.shape[0]
#I'll consider only present voters
M = (np.ones(nt) - np.eye(nt))*(presentes+presentes.T)
Q = np.diag((-1)**np.arange(Xhat1.shape[1]))
X_gd = ase_opt.ase_gd_GRPDG(weights,Xhat1,Q,M)

ax2.scatter(X_gd[num_senadores_no_presentes_rep:nrep,dims[0]-1],X_gd[num_senadores_no_presentes_rep:nrep,dims[1]-1],c='royalblue',marker='o',label='Party 1 members')
ax2.scatter(X_gd[:num_senadores_no_presentes_rep,dims[0]-1],X_gd[:num_senadores_no_presentes_rep,dims[1]-1],c='gold',marker='X',label='Missing data (Party 1)')
ax2.scatter(X_gd[nrep+num_senadores_no_presentes_dem:num_senadores,dims[0]-1],X_gd[nrep+num_senadores_no_presentes_dem:num_senadores,dims[1]-1],c='firebrick',marker='o',label='Party 2 members')
ax2.scatter(X_gd[nrep:nrep+num_senadores_no_presentes_dem,dims[0]-1],X_gd[nrep:nrep+num_senadores_no_presentes_dem,dims[1]-1],c='limegreen',marker='X',label='Missing data (Party 2)')
ax2.scatter(X_gd[num_senadores:num_senadores+lrep,dims[0]-1],X_gd[num_senadores:num_senadores+lrep,dims[1]-1],c='cornflowerblue',marker='^',label='Party 1 laws')
ax2.scatter(X_gd[num_senadores+lrep:num_senadores+lrep+ldem,dims[0]-1],X_gd[num_senadores+lrep:num_senadores+lrep+ldem,dims[1]-1],c='indianred',marker='^',label='Party 2 laws')
ax2.scatter(X_gd[num_senadores+lrep+ldem:,dims[0]-1],X_gd[num_senadores+lrep+ldem:,dims[1]-1],c='darkorange',marker='^',label='Bipartisan laws')


plt.legend(loc='upper center', bbox_to_anchor=(-0.05, -0.05),fancybox=True, shadow=True, ncol=4, fontsize=36, handletextpad=0.01,columnspacing=0.5,borderpad=0.1)
fig.subplots_adjust(left=0.06,right=0.98,top=0.97,bottom=0.24,hspace=0.06,wspace=0.02)

#
# # PCA for visualization
# X_all = np.vstack((Xhat1,X_gd))
# Xtsne = PCA(n_components=2).fit_transform(X_all)
# Xtsne_gp,Xtsne_gd = np.split(Xtsne,2)
# #Xtsne_gp = TSNE(n_components=2).fit_transform(Xhat1)
# #Xtsne_gd = TSNE(n_components=2).fit_transform(X_gd)
#
# fig, (ax1,ax2) = plt.subplots(nrows=2, ncols=1,sharex=True,sharey=True)
#
# dims=[1,2]
# ax1.scatter(Xtsne_gp[num_senadores_no_presentes_rep:nrep,dims[0]-1],Xtsne_gp[num_senadores_no_presentes_rep:nrep,dims[1]-1],c='royalblue',marker='o',label='Party 1')
# ax1.scatter(Xtsne_gp[:num_senadores_no_presentes_rep,dims[0]-1],Xtsne_gp[:num_senadores_no_presentes_rep,dims[1]-1],c='gold',marker='X',label='Not present (Party 1)')
# ax1.scatter(Xtsne_gp[nrep+num_senadores_no_presentes_dem:num_senadores,dims[0]-1],Xtsne_gp[nrep+num_senadores_no_presentes_dem:num_senadores,dims[1]-1],c='firebrick',marker='o',label='Party 2')
# ax1.scatter(Xtsne_gp[nrep:nrep+num_senadores_no_presentes_dem,dims[0]-1],Xtsne_gp[nrep:nrep+num_senadores_no_presentes_dem,dims[1]-1],c='limegreen',marker='X',label='Not present (Party 2)')
# ax1.scatter(Xtsne_gp[num_senadores:num_senadores+lrep,dims[0]-1],Xtsne_gp[num_senadores:num_senadores+lrep,dims[1]-1],c='cornflowerblue',marker='^',label='Party 1 laws')
# ax1.scatter(Xtsne_gp[num_senadores+lrep:num_senadores+lrep+ldem,dims[0]-1],Xtsne_gp[num_senadores+lrep:num_senadores+lrep+ldem,dims[1]-1],c='indianred',marker='^',label='Party 2 laws')
# ax1.scatter(Xtsne_gp[num_senadores+lrep+ldem:,dims[0]-1],Xtsne_gp[num_senadores+lrep+ldem:,dims[1]-1],c='darkorange',marker='^',label='Bipartisan laws')
#
# ax2.scatter(Xtsne_gd[num_senadores_no_presentes_rep:nrep,dims[0]-1],Xtsne_gd[num_senadores_no_presentes_rep:nrep,dims[1]-1],c='royalblue',marker='o',label='Party 1 members')
# ax2.scatter(Xtsne_gd[:num_senadores_no_presentes_rep,dims[0]-1],Xtsne_gd[:num_senadores_no_presentes_rep,dims[1]-1],c='gold',marker='X',label='Missing data (Party 1)')
# ax2.scatter(Xtsne_gd[nrep+num_senadores_no_presentes_dem:num_senadores,dims[0]-1],Xtsne_gd[nrep+num_senadores_no_presentes_dem:num_senadores,dims[1]-1],c='firebrick',marker='o',label='Party 2 members')
# ax2.scatter(Xtsne_gd[nrep:nrep+num_senadores_no_presentes_dem,dims[0]-1],Xtsne_gd[nrep:nrep+num_senadores_no_presentes_dem,dims[1]-1],c='limegreen',marker='X',label='Missing data (Party 2)')
# ax2.scatter(Xtsne_gd[num_senadores:num_senadores+lrep,dims[0]-1],Xtsne_gd[num_senadores:num_senadores+lrep,dims[1]-1],c='cornflowerblue',marker='^',label='Party 1 laws')
# ax2.scatter(Xtsne_gd[num_senadores+lrep:num_senadores+lrep+ldem,dims[0]-1],Xtsne_gd[num_senadores+lrep:num_senadores+lrep+ldem,dims[1]-1],c='indianred',marker='^',label='Party 2 laws')
# ax2.scatter(Xtsne_gd[num_senadores+lrep+ldem:,dims[0]-1],Xtsne_gd[num_senadores+lrep+ldem:,dims[1]-1],c='darkorange',marker='^',label='Bipartisan laws')
#
#
# plt.legend(loc='upper center', bbox_to_anchor=(0.48, -0.07),fancybox=True, shadow=True, ncol=4)
# plt.suptitle('PCA')
# fig.subplots_adjust(left=0.07,right=0.98,top=0.93,bottom=0.20,hspace=0.06)
#
# # t-distributed stochastic neighbor embedding for visualization
# X_all = np.vstack((Xhat1,X_gd))
# Xtsne = TSNE(n_components=2).fit_transform(X_all)
# Xtsne_gp,Xtsne_gd = np.split(Xtsne,2)
# #Xtsne_gp = TSNE(n_components=2).fit_transform(Xhat1)
# #Xtsne_gd = TSNE(n_components=2).fit_transform(X_gd)
#
# fig, (ax1,ax2) = plt.subplots(nrows=2, ncols=1,sharex=True,sharey=True)
#
# dims=[1,2]
# ax1.scatter(Xtsne_gp[num_senadores_no_presentes_rep:nrep,dims[0]-1],Xtsne_gp[num_senadores_no_presentes_rep:nrep,dims[1]-1],c='royalblue',marker='o',label='Party 1')
# ax1.scatter(Xtsne_gp[:num_senadores_no_presentes_rep,dims[0]-1],Xtsne_gp[:num_senadores_no_presentes_rep,dims[1]-1],c='gold',marker='X',label='Not present (Party 1)')
# ax1.scatter(Xtsne_gp[nrep+num_senadores_no_presentes_dem:num_senadores,dims[0]-1],Xtsne_gp[nrep+num_senadores_no_presentes_dem:num_senadores,dims[1]-1],c='firebrick',marker='o',label='Party 2')
# ax1.scatter(Xtsne_gp[nrep:nrep+num_senadores_no_presentes_dem,dims[0]-1],Xtsne_gp[nrep:nrep+num_senadores_no_presentes_dem,dims[1]-1],c='limegreen',marker='X',label='Not present (Party 2)')
# ax1.scatter(Xtsne_gp[num_senadores:num_senadores+lrep,dims[0]-1],Xtsne_gp[num_senadores:num_senadores+lrep,dims[1]-1],c='cornflowerblue',marker='^',label='Party 1 laws')
# ax1.scatter(Xtsne_gp[num_senadores+lrep:num_senadores+lrep+ldem,dims[0]-1],Xtsne_gp[num_senadores+lrep:num_senadores+lrep+ldem,dims[1]-1],c='indianred',marker='^',label='Party 2 laws')
# ax1.scatter(Xtsne_gp[num_senadores+lrep+ldem:,dims[0]-1],Xtsne_gp[num_senadores+lrep+ldem:,dims[1]-1],c='darkorange',marker='^',label='Bipartisan laws')
#
# ax2.scatter(Xtsne_gd[num_senadores_no_presentes_rep:nrep,dims[0]-1],Xtsne_gd[num_senadores_no_presentes_rep:nrep,dims[1]-1],c='royalblue',marker='o',label='Party 1 members')
# ax2.scatter(Xtsne_gd[:num_senadores_no_presentes_rep,dims[0]-1],Xtsne_gd[:num_senadores_no_presentes_rep,dims[1]-1],c='gold',marker='X',label='Missing data (Party 1)')
# ax2.scatter(Xtsne_gd[nrep+num_senadores_no_presentes_dem:num_senadores,dims[0]-1],Xtsne_gd[nrep+num_senadores_no_presentes_dem:num_senadores,dims[1]-1],c='firebrick',marker='o',label='Party 2 members')
# ax2.scatter(Xtsne_gd[nrep:nrep+num_senadores_no_presentes_dem,dims[0]-1],Xtsne_gd[nrep:nrep+num_senadores_no_presentes_dem,dims[1]-1],c='limegreen',marker='X',label='Missing data (Party 2)')
# ax2.scatter(Xtsne_gd[num_senadores:num_senadores+lrep,dims[0]-1],Xtsne_gd[num_senadores:num_senadores+lrep,dims[1]-1],c='cornflowerblue',marker='^',label='Party 1 laws')
# ax2.scatter(Xtsne_gd[num_senadores+lrep:num_senadores+lrep+ldem,dims[0]-1],Xtsne_gd[num_senadores+lrep:num_senadores+lrep+ldem,dims[1]-1],c='indianred',marker='^',label='Party 2 laws')
# ax2.scatter(Xtsne_gd[num_senadores+lrep+ldem:,dims[0]-1],Xtsne_gd[num_senadores+lrep+ldem:,dims[1]-1],c='darkorange',marker='^',label='Bipartisan laws')
#
#
# plt.legend(loc='upper center', bbox_to_anchor=(0.48, -0.07),fancybox=True, shadow=True, ncol=4)
# plt.suptitle('t-SNE')
# fig.subplots_adjust(left=0.07,right=0.98,top=0.93,bottom=0.20,hspace=0.06)

plt.show()