import numpy as np 
import networkx as nx
from sklearn.metrics import pairwise_distances

def optimize_scale(X,D,func):
    from scipy.optimize import minimize_scalar
    f = lambda a: func(a * X, D)
    min_a = minimize_scalar(f)
    return func(min_a.x * X, D)

class Metrics():

    def __init__(self, G: nx.Graph, pos):
        """
        G: networkx graph 
        pos: 2 dimensional embedding of graph G
            can be either a dictionary or numpy array
        """
        self.G = nx.convert_node_labels_to_integers(G) 
        self.N = G.number_of_nodes()

        if isinstance(pos, dict):
            X = np.zeros((G.number_of_nodes(), 2))
            for i,(v, p) in enumerate(pos.items()):
                X[i] = p
            self.X = X
        elif isinstance(pos, np.ndarray):
            self.X = pos

        self.apsp()

        
    def apsp(self):
        d = np.zeros((self.N, self.N))

        dists = dict(nx.all_pairs_shortest_path_length(self.G))
        for u in dists:
            for v in range(u):
                d[u,v] = dists[u][v]
                d[v,u] = dists[u][v]
        self.D = d

    def compute_stress(self):
        """
        X: N x p array (with p dimension of projection) that represents positions in low dimensional space 
        D: N x N distance array from high dimensional space 
        alpha: Scalar to stretch or shrink X by to compute stress

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



    def get_neighborhood(self,X,d,rg = 2):
        """
        How well do the local neighborhoods represent the theoretical neighborhoods?
        Closer to 1 is better.
        Measure of percision: ratio of true positives to true positives+false positives
        """
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