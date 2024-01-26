import s_gd2
import networkx as nx 
import numpy as np 


G = nx.lattice.grid_graph((10,10))
G = nx.convert_node_labels_to_integers(G)

I = [u for u,v in G.edges()]
J = [v for u,v in G.edges()]
pos = s_gd2.layout_convergent(I,J)

####
from metrics import Metrics
import tqdm as tqdm

M = Metrics(G,pos)

results = list()
interval = np.linspace(1e-8, 5, 500)
for a in tqdm.tqdm(interval):
    L_prime = a * pos 
    stress = Metrics(G,L_prime).compute_stress()
    results.append(stress)

import pylab as plt 
plt.plot(interval, results)
plt.show()