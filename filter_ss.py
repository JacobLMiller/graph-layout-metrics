import networkx as nx
import numpy as np 
import os

for fname in os.listdir("SS_graphs"):
    E = np.loadtxt("SS_graphs/" + fname)
    if isinstance(E[0], np.float64): continue
    G = nx.Graph()
    G.add_edges_from(E.tolist())
    if list(nx.isolates(G)): print(f"{fname} has {len(list(nx.isolates(G)))} isolates")
    G.remove_nodes_from(nx.isolates(G))
    if not nx.is_connected(G): print(f"{fname} is not connected")
    