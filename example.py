import networkx as nx
import numpy as np
import pylab as plt
import os
import tqdm
from modules import graph_io
from modules.metrics import Metrics

# Lets look at how stress behaves on some random layouts

# Setup output directory
if not os.path.isdir("outputs"):
    os.makedirs("outputs")

# All graphs in the graphs directory
graph_corpus = list(graph_io.get_corpus_file_names())

for gname in tqdm.tqdm(graph_corpus):
    G, X = graph_io.load_graph_with_embedding(gname, "random")

    M = Metrics(G, X)

    # A range of alphas between 1e-12 and 20, evenly spaced with 500 samples
    alpha_spectrum = np.linspace(1e-12, 20, 20)

    stress = np.zeros_like(alpha_spectrum)
    for i, alpha in enumerate(alpha_spectrum):
        M.setX(alpha * X)
        stress[i] = M.compute_stress()

    plt.plot(alpha_spectrum, stress, label="stress")
    plt.legend()
    plt.xlabel("alpha scale factor")
    plt.ylabel("stress")
    plt.suptitle(f"stress for {gname}")

    plt.savefig(f"outputs/{gname}_random_stress.png")
    plt.clf()
