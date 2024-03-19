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

    create_kruskal_graphs(graph_corpus)

    # for gname in graph_corpus:
    for gname in tqdm.tqdm(graph_corpus):
        for emb in ["random", "stress", "tsnet"]:
            G, X = graph_io.load_graph_with_embedding(gname, emb)

            if len(X) >= 2000:
                break

            # spectrum = np.linspace(0.1, 20, 100)
            M = MetricsH(G, X)

            # kruskal_vals = list()
            # for alpha in spectrum:
            #     M.setX(alpha * X)
            #     kruskal_vals.append(M.compute_stress_kruskal())

            # plt.plot(spectrum, kruskal_vals)
            # plt.savefig(f"outputs/shepard_diagrams/{gname}_{emb}_shepard.png")
            # plt.clf()

            # TODO OPTIMIZE, FIND AND COMPARE OPTIMAL ALPHA

            # plot_shepard_diagram(M, gname, emb)
            # plot_metric("stress_kruskal", M.compute_stress_kruskal, M, X, alpha_spectrum, gname, emb)
            # plot_metric("stress_norm", M.compute_stress_norm, M, X, alpha_spectrum, gname, emb)
            # plot_metric("neighborhood_preservation", M.compute_neighborhood, M, X, alpha_spectrum, gname, emb)
            # plot_metric("edge_uniformity", M.compute_ideal_edge_avg, M, X, alpha_spectrum, gname, emb)
            # plot_metric("edge_uniformity_indep", M.compute_ideal_edge_avg_indep, M, X, alpha_spectrum, gname, emb)

            # TODO compare with other stress algos, compare for all layouts


def create_kruskal_graphs(corpus):
    for gname in tqdm.tqdm(corpus):
        kruskal_vals = list()
        for emb in ["random", "stress", "tsnet"]:
            G, X = graph_io.load_graph_with_embedding(gname, emb)

            # if len(X) >= 2000:
            #     break

            M = MetricsH(G, X)

            kruskal_vals.append(M.compute_stress_kruskal())

        if (len(kruskal_vals) != 3):
            continue

        plt.bar(['random', 'stress', 'tsnet'], kruskal_vals,
                color=['tab:red', 'tab:blue', 'tab:green'])
        plt.ylabel('kruskal stress measure')
        plt.ylim(0, 1)
        plt.suptitle(f'kruskal stress for {gname}')

        if not os.path.isdir("outputs/stress_kruskal"):
            os.makedirs("outputs/stress_kruskal")

        plt.savefig(f"outputs/stress_kruskal/{gname}_kruskal.png")
        plt.clf()


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

    from sklearn.metrics import pairwise_distances
    output_dists = pairwise_distances(M.X)
    output_dists = output_dists[np.triu_indices(output_dists.shape[0], 1)]

    input_dists = M.D[np.triu_indices(M.D.shape[0], 1)]

    sorted_indices = np.argsort(input_dists)
    input_dists = input_dists[sorted_indices]
    output_dists = output_dists[sorted_indices]

    dij = [i[0] for i in sv]
    xij = [i[1] for i in sv]

    isoreg = IsotonicRegression().fit(dij, xij)
    plt.scatter(xij, dij, s=5, c='dimgray')
    plt.plot(isoreg.predict(dij), dij, c='darkred')
    plt.plot(IsotonicRegression().fit(input_dists, output_dists).predict(
        input_dists), input_dists, c="blue")

    plt.xlabel("x_ij")
    plt.ylabel("d_ij")
    plt.suptitle(f"shepard diagram for {gname}_{emb}")

    plt.savefig(f"outputs/shepard_diagrams/{gname}_{emb}_shepard.png")
    plt.clf()


if __name__ == '__main__':
    main()

    # test = np.array([
    #     [1,2,3],
    #     [4,5,6],
    #     [7,8,9]
    # ])
    # print(np.triu(test))
    # print()
    # print(np.triu(test, 1))
