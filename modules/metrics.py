import numpy as np 
import networkx as nx
from sklearn.metrics import pairwise_distances
from modules import graph_io

def optimize_scale(X,D,func):
    from scipy.optimize import minimize_scalar
    f = lambda a: func(a * X, D)
    min_a = minimize_scalar(f)
    return func(min_a.x * X, D)

def dist(u,v):
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
            for i,(v, p) in enumerate(pos.items()):
                X[i] = p
            self.X = X
        elif isinstance(pos, np.ndarray):
            self.X = pos

        self.D = graph_io.get_apsp(self.G)

    def setX(self,X):
        self.X = X

    def compute_stress(self):
        """
        Computes \sum_{i,j} (||X_i - X_j || - D_{i,j} )^2
        """

        X = self.X
        D = self.D 
        N = self.N

        #Calculate pairwise norm, store in difference variable
        sum_of_squares = (X * X).sum(axis=1)
        difference = np.sqrt( abs( sum_of_squares.reshape((N,1)) + sum_of_squares.reshape((1,N)) - 2 * (X@X.T) ))

        #Some error may have accumlated, set diagonal to 0 
        np.fill_diagonal(difference, 0)

        stress = np.sum( np.square( (difference - D) / np.maximum(D, 1e-15) ) )

        return stress



    def compute_neighborhood(self,rg = 2):
        """
        How well do the local neighborhoods represent the theoretical neighborhoods?
        Closer to 1 is better.
        Measure of percision: ratio of true positives to true positives+false positives
        """
        X = self.X 
        d = self.D

        def get_k_embedded(X,k_t):
            dist_mat = pairwise_distances(X)
            return [np.argsort(dist_mat[i])[1:len(k_t[i])+1] for i in range(len(dist_mat))]

        k_theory = [np.where((d[i] <= rg) & (d[i] > 0))[0] for i in range(len(d))]

        k_embedded = get_k_embedded(X,k_theory)


        NP = 0
        for i in range(len(X)):
            if len(k_theory[i]) <= 0:
                continue
            intersect = np.intersect1d(k_theory[i],k_embedded[i]).size
            jaccard = intersect / (2*k_theory[i].size - intersect)

            NP += jaccard

        return NP / len(X)
    
    def compute_ideal_edge_avg(self):
        X = self.X
        edge_lengths = np.array([dist(X[i], X[j]) for i,j in self.G.edges()])

        avg = edge_lengths.mean()

        return np.sum(np.square( (edge_lengths - avg) / avg ))

