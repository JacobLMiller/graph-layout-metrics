from modules import graph_io
from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.metrics import pairwise_distances


def degree_info(G):
    degrees = dict(G.degree())
    min_degree = min(degrees.values())
    max_degree = max(degrees.values())

    return (min_degree, max_degree)


def graph_density(G):
    nedges = G.number_of_edges()
    nnodes = G.number_of_nodes()

    return nedges / (nnodes * (nnodes - 1) / 2)


def compute_stats(values):
    mean = np.mean(values)
    min_val = np.min(values)
    max_val = np.max(values)
    std_dev = np.std(values)

    return [mean, std_dev, min_val, max_val]


if __name__ == "__main__":
    nnodes_rome = []
    nedges_rome = []
    mindeg_rome = []
    maxdeg_rome = []
    density_rome = []
    maxpath_rome = []

    nnodes_ss = []
    nedges_ss = []
    mindeg_ss = []
    maxdeg_ss = []
    density_ss = []
    maxpath_ss = []

    algs = ['stress', 'neato', 'sfdp', 'twopi', 'tsnet', 'random']
    maxdist_rome = {k: [] for k in algs}
    maxdist_ss = {k: [] for k in algs}

    for count, G in enumerate(tqdm(graph_io.get_corpus_Rome())):
        nedges_rome.append(G.number_of_edges())
        nnodes_rome.append(G.number_of_nodes())
        mindeg, maxdeg = degree_info(G)
        mindeg_rome.append(mindeg)
        maxdeg_rome.append(maxdeg)
        density_rome.append(graph_density(G))
        maxpath_rome.append(np.max(graph_io.get_apsp(G)))
        for alg in algs:
            pos = graph_io.load_embedding(G, alg)
            maxdist = np.max(pairwise_distances(pos))
            maxdist_rome[alg].append(maxdist)

    for count, G in enumerate(tqdm(graph_io.get_corpus_SS())):
        nedges_ss.append(G.number_of_edges())
        nnodes_ss.append(G.number_of_nodes())
        mindeg, maxdeg = degree_info(G)
        mindeg_ss.append(mindeg)
        maxdeg_ss.append(maxdeg)
        density_ss.append(graph_density(G))
        maxpath_ss.append(np.max(graph_io.get_apsp(G)))
        for alg in algs:
            pos = graph_io.load_embedding(G, alg)
            maxdist = np.max(pairwise_distances(pos))
            maxdist_ss[alg].append(maxdist)

    nedges_rome = np.array(nedges_rome)
    nnodes_rome = np.array(nnodes_rome)
    mindeg_rome = np.array(mindeg_rome)
    maxdeg_rome = np.array(maxdeg_rome)
    density_rome = np.array(density_rome)
    maxpath_rome = np.array(maxpath_rome)

    nedges_ss = np.array(nedges_ss)
    nnodes_ss = np.array(nnodes_ss)
    mindeg_ss = np.array(mindeg_ss)
    maxdeg_ss = np.array(maxdeg_ss)
    density_ss = np.array(density_ss)
    maxpath_ss = np.array(maxpath_ss)

    maxdist_rome = {k: np.array(v) for k, v in maxdist_rome.items()}
    maxdist_ss = {k: np.array(v) for k, v in maxdist_ss.items()}

    nedges_rome = compute_stats(nnodes_rome)
    nnodes_rome = compute_stats(nedges_rome)
    mindeg_rome = compute_stats(mindeg_rome)
    maxdeg_rome = compute_stats(maxdeg_rome)
    density_rome = compute_stats(density_rome)
    maxpath_rome = compute_stats(maxpath_rome)

    nedges_ss = compute_stats(nnodes_ss)
    nnodes_ss = compute_stats(nedges_ss)
    mindeg_ss = compute_stats(mindeg_ss)
    maxdeg_ss = compute_stats(maxdeg_ss)
    density_ss = compute_stats(density_ss)
    maxpath_ss = compute_stats(maxpath_ss)

    maxdist_rome = {k: compute_stats(v) for k, v in maxdist_rome.items()}
    maxdist_ss = {k: compute_stats(v) for k, v in maxdist_ss.items()}

    stats_rome = [nedges_rome,
                  nnodes_rome,
                  mindeg_rome,
                  maxdeg_rome,
                  density_rome,
                  maxpath_rome] + list(maxdist_rome.values())

    stats_ss = [nedges_ss,
                nnodes_ss,
                mindeg_ss,
                maxdeg_ss,
                density_ss,
                maxpath_ss] + list(maxdist_ss.values())
    df_rome = pd.DataFrame(
        stats_rome, columns=['Mean', 'Standard Deviation', 'Minimum', 'Maximum'])
    df_ss = pd.DataFrame(
        stats_ss, columns=['Mean', 'Standard Deviation', 'Minimum', 'Maximum'])

    df_rome[''] = ['# Nodes',
                   '# Edges',
                   'Min Degree',
                   'Max Degree',
                   'Density',
                   'Max Graph Distance'] + [f"{alg} Max Drawing Distance" for alg in algs]
    df_ss[''] = ['# Nodes',
                 '# Edges',
                 'Min Degree',
                 'Max Degree',
                 'Density',
                 'Max Graph Distance'] + [f"{alg} Max Drawing Distance" for alg in algs]
    df_rome = df_rome.set_index('')
    df_ss = df_ss.set_index('')

    df_rome.to_csv("results/rome-graph-stats.csv")
    df_ss.to_csv("results/ss-graph-stats.csv")
