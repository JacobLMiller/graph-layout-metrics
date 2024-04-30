import networkx as nx
import numpy as np
import os
from modules import graph_io
from tqdm import tqdm

# download from https://visdunneright.github.io/gd_benchmark_sets/
# place extracted "rome" directory into repository

if not os.path.isdir("Rome_graphs"):
    os.makedirs("Rome_graphs")

for root, dirs, files in os.walk("rome/data/"):
    for name in tqdm(files):
        if name.split('.')[-1] != 'json':
            print(f"not json: {name}")
            continue
        try:
            G = graph_io.json_to_graph(f'rome/data/{name}')
        except:
            print(f"Error indexing: {name}")
        Elist = [[u, v] for u, v in G.edges()]
        if nx.is_connected(G) and G.number_of_nodes() <= 2000:
            fname = '.'.join(name.split('.')[:-1])
            np.savetxt(f"Rome_graphs/{fname}.txt", Elist, fmt='%d')
        elif nx.is_connected(G):
            print(f"Too many nodes: {name} - {G.number_of_nodes()}")
        else:
            print(f"Not connected: {name}")
