import numpy as np
import networkx as nx
from sklearn.metrics import pairwise_distances
from sklearn.isotonic import IsotonicRegression
from modules import graph_io


def optimize_scale(X, D, func):
    from scipy.optimize import minimize_scalar
    def f(a): return func(a * X, D)
    min_a = minimize_scalar(f)
    return func(min_a.x * X, D)


def dist(u, v):
    return np.sqrt(np.sum(np.square(u - v)))


class MetricsH():

    def __init__(self, G: nx.Graph, pos):
        """
        G: networkx graph 
        pos: 2 dimensional embedding of graph G
            can be either a dictionary or numpy array
        """
        self.G = nx.convert_node_labels_to_integers(G)
        self.N = G.number_of_nodes()
        self.name = self.G["gname"] if "gname" in self.G else "tmp"

        if isinstance(pos, dict):
            X = np.zeros((G.number_of_nodes(), 2))
            for i, (v, p) in enumerate(pos.items()):
                X[i] = p
            self.X = X
        elif isinstance(pos, np.ndarray):
            self.X = pos

        self.D = graph_io.get_apsp(self.G)

        self.xij = None

    def setX(self, X):
        self.X = X

    def compute_stress_norm(self):
        """
        Computes \sum_{i,j} ( (||X_i - X_j|| - D_{i,j}) / D_{i,j})^2
        """

        X = self.X
        D = self.D

        xij = []
        for i in range(len(X)):
            for j in range(i+1, len(X)):
                xij.append(((X[j][0] - X[i][0]) ** 2 +
                           (X[j][1] - X[j][1]) ** 2) ** 0.5)

        dij = []
        for i in range(len(D)):
            for j in range(i+1, len(D[0])):
                dij.append(D[i][j])

        # print("xij", xij[:10], xij[-10:], len(xij))
        # print("dij", dij[:10], dij[-10:], len(dij))

        sij = []
        for k in range(len(xij)):
            sij.append(((xij[k] - dij[k]) / (max(dij[k], 1e-15))) ** 2)

        return sum(sij)

    def shepard_vals(self):
        X = self.X
        D = self.D

        sij = []
        for i in range(len(X)):
            for j in range(i+1, len(X)):
                xij = ((X[j][0] - X[i][0]) ** 2 +
                       (X[j][1] - X[i][1]) ** 2) ** 0.5
                sij.append((D[i][j], xij))

        return sorted(sij)

    # def get_pairwise(self):
    #     if isinstance(self.xij, np.ndarray):
    #         return self.xij
    #     self.xij = pairwise_distances(self.X)
    #     return self.xij

    def compute_stress_kruskal(self):
        output_dists = pairwise_distances(self.X)
        xij = output_dists[np.triu_indices(output_dists.shape[0])]

        dij = self.D[np.triu_indices(self.D.shape[0])]

        sorted_indices = np.argsort(dij)
        dij = dij[sorted_indices]
        xij = xij[sorted_indices]

        hij = IsotonicRegression().fit(dij, xij).predict(dij)

        raw_stress = np.sum(np.square(xij - hij))
        norm_factor = np.sum(np.square(xij))

        kruskal_stress = np.sqrt(raw_stress / norm_factor)
        return kruskal_stress

    def compute_neighborhood(self, rg=2):
        """
        How well do the local neighborhoods represent the theoretical neighborhoods?
        Closer to 1 is better.
        Measure of percision: ratio of true positives to true positives+false positives
        """
        X = self.X
        d = self.D

        def get_k_embedded(X, k_t):
            dist_mat = pairwise_distances(X)
            return [np.argsort(dist_mat[i])[1:len(k_t[i])+1] for i in range(len(dist_mat))]

        k_theory = [np.where((d[i] <= rg) & (d[i] > 0))[0]
                    for i in range(len(d))]

        k_embedded = get_k_embedded(X, k_theory)

        NP = 0
        for i in range(len(X)):
            if len(k_theory[i]) <= 0:
                continue
            intersect = np.intersect1d(k_theory[i], k_embedded[i]).size
            jaccard = intersect / (2*k_theory[i].size - intersect)

            NP += jaccard

        return NP / len(X)

    def compute_ideal_edge_avg(self):
        X = self.X
        edge_lengths = np.array([dist(X[i], X[j]) for i, j in self.G.edges()])

        avg = edge_lengths.mean()

        return np.sum(np.square((edge_lengths - avg) / avg))

    def compute_ideal_edge_avg_indep(self):
        X = self.X
        edge_lengths = np.array([dist(X[i], X[j]) for i, j in self.G.edges()])

        return np.sum(np.square((edge_lengths - 1) / 1))
