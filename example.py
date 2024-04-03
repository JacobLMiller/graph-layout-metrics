import networkx as nx
import numpy as np
import pylab as plt
import os
import tqdm
from modules import graph_io
from modules.metrics import Metrics
from modules.normalization import Normalize

# Lets look at how stress behaves on some random layouts

# Setup output directory
if not os.path.isdir("outputs"):
    os.makedirs("outputs")

# All graphs in the graphs directory
graph_corpus = list(graph_io.get_corpus_file_names())

for gname in tqdm.tqdm(graph_corpus):
    G, X = graph_io.load_graph_with_embedding(gname, "random")

    M = Metrics(G,X)
    N = Normalize(G, X)

    normal_factors = {
        "Identity": N.identity(),
        "Unit Norm": N.unit_norm(),
        "Optimize": None
    }

    #A range of alphas between 1e-12 and 20, evenly spaced with 500 samples
    alpha_spectrum = np.linspace(1e-12, 10, 500)

    stress = np.zeros_like(alpha_spectrum)
    for i, alpha in enumerate(alpha_spectrum):
        M.setX(alpha * X)
        stress[i] = M.compute_stress_norm()

    normal_factors["Optimize"] = alpha_spectrum[ np.argmin(stress) ]
    print(normal_factors)

    plt.plot(alpha_spectrum, stress, label="stress")

    y_min = np.min(stress)
    y_max = np.max(stress)
    interp_space = 20
    for name, val in normal_factors.items():
        print([val] * interp_space)
        plt.plot([val] * interp_space, np.linspace(y_min,y_max,interp_space), label=name)

    plt.legend()
    plt.xlabel("alpha scale factor")
    plt.ylabel("stress")
    plt.suptitle(f"stress for {gname}")

    plt.show()
    # plt.savefig(f"outputs/{gname}_random_stress.png")
    plt.clf()
    break
