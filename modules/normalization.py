"""
Contains various normalization schemes for input/output pairs 
"""
import numpy as np
from sklearn.metrics import pairwise_distances
from scipy.optimize import minimize_scalar

def compute_stress(X, D):
    """
    Computes \sum_{i,j} (||X_i - X_j || - D_{i,j} )^2
    """
    N = X.shape[0]

    #Calculate pairwise norm, store in difference variable
    sum_of_squares = (X * X).sum(axis=1)
    difference = np.sqrt( abs( sum_of_squares.reshape((N,1)) + sum_of_squares.reshape((1,N)) - 2 * (X@X.T) ))

    #Some error may have accumlated, set diagonal to 0 
    np.fill_diagonal(difference, 0)

    stress = np.sum( np.square( (difference - D) / np.maximum(D, 1e-15) ) )

    return stress


class Normalize():
    def __init__(self, D, X):
        self.D = D 
        self.X = X 
        self.alpha = 1.0

    def compute_alpha(self,new):
        scalar_matrix = np.divide(self.X, new)
        return scalar_matrix[0,0]

    def identity(self):
        self.alpha = 1.0
        return self.alpha
    
    def unit_square(self):
        """
        x - min(x) / max(x) - min(x)
        """
        left = np.min(self.X,axis=0)
        right = np.max(self.X, axis=0)
        
        ret = (self.X - left) / (right - left)

        self.alpha = self.compute_alpha(ret)

        return self.alpha
    
    def unit_norm(self):
        X = self.X
        ret = X / np.max(np.linalg.norm(X, axis=1))
        self.compute_alpha(ret)
        return self.alpha
    
    def find_min(self):
        from scipy.optimize import minimize_scalar
        stress = lambda a: compute_stress(a*self.X, self.D)
        
        min_a = minimize_scalar(stress, bounds=(1e-15,20), method="bounded").x
        
        self.alpha = min_a

        return min_a 



if __name__ == "__main__":
    import graph_io as graph_io
    G, X = graph_io.load_graph_with_embedding("can_96", "random")
    D = graph_io.get_apsp(G)

    N = Normalize(D,X)
    N.find_min()

    print(N.alpha)


