import random
import networkx as nx
import numpy as np
import pylab as plt
import os
import tqdm
from modules import graph_io
from modules.metrics import Metrics
from modules.metrics_h import MetricsH
from sklearn.isotonic import IsotonicRegression


def main():
    # Setup output directory
    if not os.path.isdir("outputs"):
        os.makedirs("outputs")

    # All graphs in the graphs directory
    graph_corpus = list(graph_io.get_corpus_file_names())

    # for gname in tqdm.tqdm(graph_corpus):
    for gname in graph_corpus:
        for emb in ["random", "stress", "tsnet"]:
            G, X = graph_io.load_graph_with_embedding(gname, emb)
            # G, X = graph_io.load_graph_with_embedding("spx_teaser", "tsnet")

            if len(X) >= 1000:
                break

            # M = Metrics(G, X)
            M1 = MetricsH(G, X)
            M2 = MetricsH(G, X * 2)

            print(gname, emb, "kruskal 1x:", M1.compute_stress_kruskal())
            print(gname, emb, "kruskal 2x:", M2.compute_stress_kruskal())
            
            # TODO OPTIMIZE, FIND AND COMPARE OPTIMAL ALPHA
            # A range of alphas between 1e-12 and 20, evenly spaced
            # alpha_spectrum = np.linspace(2e-12, 32, 50)

            # plot_shepard_diagram(M, gname, emb)
            # plot_metric("stress_kruskal", M.compute_stress_kruskal, M, X, alpha_spectrum, gname, emb)
            # plot_metric("stress_norm", M.compute_stress_norm, M, X, alpha_spectrum, gname, emb)
            # plot_metric("neighborhood_preservation", M.compute_neighborhood, M, X, alpha_spectrum, gname, emb)
            # plot_metric("edge_uniformity", M.compute_ideal_edge_avg, M, X, alpha_spectrum, gname, emb)
            # plot_metric("edge_uniformity_indep", M.compute_ideal_edge_avg_indep, M, X, alpha_spectrum, gname, emb)

            # TODO compare with other stress algos, compare for all layouts
    


def plot_metric(mstr, mfn, M, X, alphas, gname, emb):
    metric = np.zeros_like(alphas)

    if not os.path.isdir(f"outputs/{mstr}"):
        os.makedirs(f"outputs/{mstr}")

    for i, alpha in enumerate(alphas):
        M.setX(alpha * X)
        metric[i] = mfn()

    plt.plot(alphas, metric, label=mstr)
    plt.legend()
    plt.xlabel("alpha scale factor")
    plt.ylabel(mstr)
    plt.suptitle(f"{mstr} for {gname}")

    plt.savefig(f"outputs/{mstr}/{gname}_{emb}_{mstr}.png")  # emb = 'random'
    plt.clf()


def plot_shepard_diagram(M, gname, emb):
    sv = M.shepard_vals()

    if not os.path.isdir(f"outputs/shepard_diagrams"):
        os.makedirs(f"outputs/shepard_diagrams")

    dij = [i[0] for i in sv]
    xij = [i[1] for i in sv]

    isoreg = IsotonicRegression().fit(dij, xij)
    plt.scatter(xij, dij, s=5, c='dimgray')
    plt.plot(isoreg.predict(dij), dij, c='darkred')

    plt.xlabel("x_ij")
    plt.ylabel("d_ij")
    plt.suptitle(f"shepard diagram for {gname}_{emb}")

    plt.savefig(f"outputs/shepard_diagrams/{gname}_{emb}_shepard.png")
    plt.clf()


if __name__ == '__main__':
    main()
