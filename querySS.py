from ssgetpy import search,fetch
import networkx as nx

matcollection = search(rowbounds=(None,1000), colbounds=(None,1000), limit=5000)

print(matcollection[0])

matcollection.download(format="MM", destpath="SS_out/", extract=True)

from scipy.io import mmread
import scipy.sparse as sp
import numpy as np
import os 
if not os.path.isdir("SS_graphs"): os.makedirs("SS_graphs")
for root, dirs, files in os.walk("SS_out/"):
    for name in files:
        if name.split(".")[1] != "mtx": continue
        print(os.path.join(root,name))
        A = mmread(os.path.join(root, name))
        E = {tuple(sorted([u,v])) for u,v,w in zip(*sp.find(A)) if u != v}
        Elist = list(E)
        if len(Elist) < 2: continue

        G = nx.Graph()
        G.add_edges_from(Elist)
        G.remove_nodes_from(nx.isolates(G))
        G = nx.convert_node_labels_to_integers(G)
        Elist = [[u,v] for u,v in G.edges()]
        if nx.is_connected(G) and G.number_of_nodes() <= 2000:
            fname = name.split(".")[0]
            np.savetxt(f"SS_graphs/{fname}.txt",Elist,fmt='%d')

# A = mmread("SS_out/ash85/ash85.mtx")
# print(A)