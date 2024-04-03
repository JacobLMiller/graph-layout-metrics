import numpy as np
import networkx as nx
from sklearn.metrics import pairwise_distances
from modules import graph_io


def optimize_scale(X, D, func):
    from scipy.optimize import minimize_scalar
    def f(a): return func(a * X, D)
    min_a = minimize_scalar(f)
    return func(min_a.x * X, D)


def dist(u, v):
    return np.sqrt(np.sum(np.square(u - v)))


class Metrics():

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

    def setX(self, X):
        self.X = X

    #TODO Something is fishy here...
    def compute_stress_norm(self):
        """
        Computes \sum_{i,j} ( (||X_i - X_j|| - D_{i,j}) / D_{i,j})^2
        """

        X = self.X
        D = self.D
        N = self.N

        # Calculate pairwise norm, store in difference variable
        sum_of_squares = (X * X).sum(axis=1)
        # print("X^2", sum_of_squares)
        difference = np.sqrt(abs(sum_of_squares.reshape(
            (N, 1)) + sum_of_squares.reshape((1, N)) - 2 * (X@X.T)))

        # Some error may have accumlated, set diagonal to 0
        np.fill_diagonal(difference, 0)
        # print("pairwise diff 0s", difference)

        stress = np.sum(np.square((difference - D) / np.maximum(D, 1e-15)))
        # print("stress", stress)

        return stress

    def compute_stress_kruskal(self):
        from sklearn.isotonic import IsotonicRegression

        
        #sklearn has implemented an efficient distance matrix algorithm for us
        output_dists = pairwise_distances(self.X)

        #Extract the upper triangular of both distance matrices into 1d arrays 
        #We know the diagonal is all zero, so offset by one
        xij = output_dists[ np.triu_indices( output_dists.shape[0], k=1 ) ]
        dij  = self.D[ np.triu_indices( self.D.shape[0], k=1 ) ]

        #Find the indices of dij that when reordered, would sort it. Apply to both arrays
        sorted_indices = np.argsort(dij)
        dij = dij[sorted_indices]
        xij = xij[sorted_indices]

        hij = IsotonicRegression().fit(dij, xij).predict(dij)

        #
        raw_stress  = np.sum( np.square( xij - hij ) )
        norm_factor = np.sum( np.square( xij ) )

        kruskal_stress = np.sqrt( raw_stress / norm_factor )
        return kruskal_stress
    
    def compute_stress_ratios():
        stress = .07

        return stress


    def pairwise_dist(self):
        X = self.X
        N = self.N

        # Calculate pairwise norm, store in difference variable
        sum_of_squares = (X * X).sum(axis=1)
        difference = np.sqrt(abs(sum_of_squares.reshape(
            (N, 1)) + sum_of_squares.reshape((1, N)) - 2 * (X@X.T)))

        # Some error may have accumlated, set diagonal to 0
        np.fill_diagonal(difference, 0)

        return difference

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

class MetricsData(Metrics):
    def __init__(self, X: np.ndarray, Y: np.ndarray):
        """
        X: Low dimensional embedding of matrix Y. Can be given as coordinates N x d matrix 
        Y: High dimensional coordiantes of objects. Can be given as N x D matrix
            (for which a pairwise distance matrix will be computed) or the distances directly as an N x N matrix        
        """

        #Check data format
        self.X = X
        if Y.shape[0] == Y.shape[1]: self.D = Y 
        else: self.D = pairwise_distances(Y)

        self.N = X.shape[0]

        #Ensure compatibility with parent class 
        self.name = None 
        self.G = None

