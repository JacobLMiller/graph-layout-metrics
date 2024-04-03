import numpy as np 
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris

from modules.metrics import MetricsData

if __name__ == "__main__":

    #Example HD data
    Y = load_iris().data

    #Reduce to 2 dimensions    
    X = PCA(n_components=2).fit_transform(Y)

    #Instantiate Metrics class 
    M = MetricsData(X, Y)
    print(M.compute_stress_kruskal())