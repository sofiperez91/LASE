#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 18:05:45 2020

@author: flarroca
"""

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import scipy
import random
import collections

#import netgraph

import graspologic as gy
import funciones_aux

import ase_opt

plt.close(fig='all')



# ##############################
# ## SBM con valor propios negativos
# ##############################

# plt.close(fig='all')

# n = [100, 900]
# p = [[0.2, 0.5],
#       [0.5, 0.2]]

# weights = gy.simulations.sbm(n=n, p=p, directed=False)
# g = nx.from_numpy_array(weights)

# # asimetrico
# ase = gy.embed.AdjacencySpectralEmbed(n_elbows = 2, diag_aug=True)
# Xhat = ase.fit_transform(weights)

# # (Xhatl, Xhatr) = funciones_aux.normalizar_rdpg_directivo(Xhatl,Xhatr)

# gy.plot.heatmap(weights)
# funciones_aux.scatter_anotado([g],[Xhat], polar=False)
# plt.title('Xhat')
# plt.show()

# funciones_aux.scatter_anotado([g],[Xhat], polar=True)
# plt.title('Xhat')
# plt.show()

# simetrico

# ase = gy.embed.AdjacencySpectralEmbed(n_components = 4, diag_aug=True)
# Xhat = ase.fit_transform(weights+weights.T)

# gy.plot.heatmap(weights+weights.T)
# funciones_aux.scatter_anotado([g],[Xhat],dims=(1,2))
# plt.title('todo')
# plt.show()

# # (w,v) = scipy.sparse.linalg.eigs(weights+diagonal, k=Xhat.shape[1],which='LM')
# # w = w[np.argsort(-abs(w))]
# # #    print(w)
# # # I use gRDPG
# estimado = Xhatl@Xhatr.T
# gy.plot.heatmap(estimado,title='estimado')
# # gy.plot.heatmap(g,title='real')

# # g = nx.from_numpy_array(weights+diagonal)
# # funciones_aux.scatter_anotado([g],[Xhat],dims=(1,2))

# # plt.figure()
# # plt.hist(estimado.flatten(),100)
# # plt.title('histograma de la estimacion')

# ###############################
# # usuarios que le gustan dos cosas
# ###############################

# plt.close(fig='all')

# n = [100, 50, 50]
# p = [[0, 0.9, 0.3],
#       [0, 0, 0], 
#       [0, 0, 0]]

# weights = gy.simulations.sbm(n=n, p=p, directed=True)
# g = nx.from_numpy_array(weights)

# # asimetrico
# ase = gy.embed.AdjacencySpectralEmbed(n_components = 2, diag_aug=True)
# [Xhatl, Xhatr] = ase.fit_transform(weights)

# (Xhatl, Xhatr) = funciones_aux.normalizar_rdpg_directivo(Xhatl,Xhatr)

# # gy.plot.heatmap(weights)
# funciones_aux.scatter_anotado([g],[Xhatl])
# plt.title('out - 50/50 ')
# plt.show()
# funciones_aux.scatter_anotado([g],[Xhatr])
# plt.title('in - 50/50')
# plt.show()

# n = [100, 100, 100]
# p = [[0, 0.9, 0.3],
#       [0, 0, 0], 
#       [0, 0, 0]]

# weights = gy.simulations.sbm(n=n, p=p, directed=True)
# g = nx.from_numpy_array(weights)

# # asimetrico
# ase = gy.embed.AdjacencySpectralEmbed(n_components = 2, diag_aug=True)
# [Xhatl, Xhatr] = ase.fit_transform(weights)

# (Xhatl, Xhatr) = funciones_aux.normalizar_rdpg_directivo(Xhatl,Xhatr)

# # gy.plot.heatmap(weights)
# funciones_aux.scatter_anotado([g],[Xhatl])
# plt.title('out - 50/100 ')
# plt.show()
# funciones_aux.scatter_anotado([g],[Xhatr])
# plt.title('in - 50/100')
# plt.show()

##############################
## Directed SBM - simula bipartidismo
##############################

plt.close(fig='all')

def grafo_senadores(nrep = 50, ndem = 50, lrep = 50, ldem = 200, lmix = 40, 
                    p_rep_rep = 0.9, p_rep_dem = 0.01, p_rep_mix = 0.2, 
                    p_dem_rep = 0.1, p_dem_dem = 0.8, p_dem_mix = 0.3):
    """
    Genera un grafo con senadores democratas y republicanos. Es un SBM directivo
    con probabilidades de votacion indicadas en cada parametro. Las leyes mix 
    son leyes que tienden a votar en ambos partidos. 
    """
    
    n = [nrep, ndem, lrep, ldem, lmix]
    nodes_dict = {}
    
    for rep in range(nrep):
        nodes_dict[rep] = 0
    for dem in range(ndem):
        nodes_dict[(dem+nrep)] = 9
    for lr in range(lrep):
        nodes_dict[(ndem+nrep+lr)] = 3
    for ld in range(ldem):
        nodes_dict[(ndem+nrep+lrep+ld)] = 7
    for lm in range(lmix):
        nodes_dict[(ndem+nrep+lrep+ldem+lm)] = 5
    
    p = [[0, 0, p_rep_rep, p_rep_dem, p_rep_mix],
          [0, 0, p_dem_rep, p_dem_dem, p_dem_mix],
          [0, 0, 0, 0, 0], # las leyes no votan
          [0, 0, 0, 0, 0], 
          [0, 0, 0, 0, 0]]
    
    weights = gy.simulations.sbm(n=n, p=p, directed=True)
    # gy.plot.heatmap(weights)
    g = nx.from_numpy_array(weights)
    
    nx.set_node_attributes(g, nodes_dict,'category')
    return (g, weights)

# I generate the original graph
(g1, weights) = grafo_senadores(lmix=200,lrep=200)

# Some senators did not vote on certain laws
presentes = np.ones_like(weights)
senadores_no_presentes = [0, 1, 2]
proba_presente = 0.3
for sen in senadores_no_presentes:
    presentes[sen,:] = presentes[sen,:]*[np.random.rand(1,weights.shape[0])<proba_presente]
presentes = np.triu(presentes)

# I'll take that as a no
weights = weights*presentes

weights = weights + weights.T
# simetrico
ase = gy.embed.AdjacencySpectralEmbed(n_elbows = 2, diag_aug=True)
Xhat1 = ase.fit_transform(weights)
# Xhatl1, Xhatr1) = funciones_aux.normalizar_rdpg_directivo(Xhatl1,Xhatr1)

gy.plot.heatmap(weights)
funciones_aux.scatter_anotado([g1],[Xhat1], dims=[2,4], polar=True)
plt.title('2,4')
plt.show()

funciones_aux.scatter_anotado([g1],[Xhat1], dims=[1,3], polar=False)
plt.title('1,3')
plt.show()

nt = Xhat1.shape[0]
#I'll consider only present voters
M = (np.ones(nt) - np.eye(nt))*(presentes+presentes.T)
Q = np.diag((-1)**np.arange(Xhat1.shape[1]))
X_gd = ase_opt.ase_gd_GRPDG(weights,Xhat1,Q,M)
funciones_aux.scatter_anotado([g1],[X_gd], dims=[1,3], polar=False)
plt.title('1,3 - GD')
plt.show()

# # simetrico

# ase = gy.embed.AdjacencySpectralEmbed(n_components = 4, diag_aug=True)
# Xhat1 = ase.fit_transform(weights+weights.T)

# gy.plot.heatmap(weights+weights.T)
# funciones_aux.scatter_anotado([g],[Xhat],dims=(1,3))
# plt.title('todo')
# plt.show()

# ##############################
# ## Directed SBM - simula bipartidismo en el tiempo
# ##############################

# # plt.close(fig='all')
# (g2, weights) = grafo_senadores(nrep = 15, ndem = 85)

# # asimetrico
# ase = gy.embed.AdjacencySpectralEmbed(n_components = 2, diag_aug=True)
# [Xhatl2, Xhatr2] = ase.fit_transform(weights)
# (Xhatl2, Xhatr2) = funciones_aux.normalizar_rdpg_directivo(Xhatl2,Xhatr2)

# gy.plot.heatmap(weights)
# funciones_aux.scatter_anotado([g1, g2],[Xhatl1, Xhatl2])
# plt.title('left')
# plt.show()
# funciones_aux.scatter_anotado([g1, g2],[Xhatr1, Xhatr2])
# plt.title('right')
# plt.show()

# # simetrico

# ase = gy.embed.AdjacencySpectralEmbed(n_components = 4, diag_aug=True)
# Xhat2 = ase.fit_transform(weights+weights.T)

# gy.plot.heatmap(weights+weights.T)
# funciones_aux.scatter_anotado([g1, g2],[Xhat1, Xhat2],dims=(1,3))
# plt.title('todo')
# plt.show()