import networkx as nx 
import numpy as np 
import pylab as plt
import os
import tqdm
from modules import graph_io
from modules.metrics import Metrics

def main():
    #Setup output directory
    if not os.path.isdir("outputs"):
        os.makedirs("outputs")

    #All graphs in the graphs directory
    graph_corpus = list(graph_io.get_corpus_file_names())

    for gname in tqdm.tqdm(graph_corpus):
        G, X = graph_io.load_graph_with_embedding(gname, "random")

        M = Metrics(G,X)

        #A range of alphas between 1e-12 and 20, evenly spaced with 500 samples
        alpha_spectrum = np.geomspace(1/128, 128, 15)#BALALALAA

        # plot_stress(M, X, alpha_spectrum, gname)
        # plot_neighborhood(M, X, alpha_spectrum, gname)
        plot_edge_uniformity(M, X, alpha_spectrum, gname)

def plot_stress(M: Metrics, X, alpha_spectrum: np.ndarray, gname: str):
    metric = np.zeros_like(alpha_spectrum)
    mstr = "stress"

    if not os.path.isdir(f"outputs/{mstr}"):
        os.makedirs(f"outputs/{mstr}")

    for i, alpha in enumerate(alpha_spectrum):
        M.setX(alpha * X)
        metric[i] = M.compute_stress()

    ax = plt.gca()
    ax.set_xscale('log')
    ax.set_yscale('log')

    plt.plot(alpha_spectrum, metric, label=mstr)
    plt.legend()
    plt.xlabel("alpha scale factor")
    plt.ylabel(mstr)
    plt.suptitle(f"{mstr} for {gname}")

    plt.savefig(f"outputs/{mstr}/{gname}_random_{mstr}.png")
    plt.clf()

def plot_neighborhood(M: Metrics, X, alpha_spectrum: np.ndarray, gname: str):
    metric = np.zeros_like(alpha_spectrum)
    mstr = "neighborhood_preservation"

    if not os.path.isdir(f"outputs/{mstr}"):
        os.makedirs(f"outputs/{mstr}")

    for i, alpha in enumerate(alpha_spectrum):
        M.setX(alpha * X)
        metric[i] = M.compute_neighborhood()

    ax = plt.gca()
    ax.set_xscale('log')
    ax.set_yscale('log')

    plt.plot(alpha_spectrum, metric, label=mstr)
    plt.legend()
    plt.xlabel("alpha scale factor")
    plt.ylabel(mstr)
    plt.suptitle(f"{mstr} for {gname}")

    plt.savefig(f"outputs/{mstr}/{gname}_random_{mstr}.png")
    plt.clf()

def plot_edge_uniformity(M: Metrics, X, alpha_spectrum: np.ndarray, gname: str):
    metric = np.zeros_like(alpha_spectrum)
    mstr = "edge_uniformity"

    if not os.path.isdir(f"outputs/{mstr}"):
        os.makedirs(f"outputs/{mstr}")

    for i, alpha in enumerate(alpha_spectrum):
        M.setX(alpha * X)
        metric[i] = M.compute_ideal_edge_avg()

    ax = plt.gca()
    ax.set_xscale('log')
    ax.set_yscale('log')

    plt.plot(alpha_spectrum, metric, label=mstr)
    plt.legend()
    plt.xlabel("alpha scale factor")
    plt.ylabel(mstr)
    plt.suptitle(f"{mstr} for {gname}")

    plt.savefig(f"outputs/{mstr}/{gname}_random_{mstr}.png")
    plt.clf()

if __name__ == '__main__':
    main()