import networkx as nx 
import numpy as np 
from modules import graph_io
from modules.metrics import Metrics
from layouts import Layout
import json

class Experiment():
    def __init__(self, metric_function, metric_name):

        """
        metric_function(G,x) must be a function with two parameters, 
        G: a networkx graph
        x: A dict/matrix of positions for G
        """
        self.metric_function = metric_function 
        self.metric_name = metric_name
        self.results = dict()
    
    def conduct_experiment(self, algs=["stress", "tsnet", "random", "sfdp", "neato", "twopi"], limit=None, size_limit=1000):
        #Calc length to show progress bar:
        from os import listdir
        total = len(listdir("SS_graphs"))

        from tqdm import tqdm
        for count, G in enumerate(tqdm(graph_io.get_corpus_SS(size_limit), total=total)):

            layouts = [graph_io.load_embedding(G, alg) for alg in algs]

            try:
                stresses = {algs[i]: self.metric_function(G, x) for i,x in enumerate(layouts)}
                stresses['order'] = sorted(stresses.keys(), key=stresses.get)

                self.results[G.graph["gname"]] = stresses
            except:
                print(f"\nBad Graph: {G.graph['gname']}")
            
            if limit and count >= limit: break

    def write_results(self,fname=None):
        if fname is None: 
            fname = f"results/{self.metric_name}-results.json"
        with open(fname, 'w') as fdata:
            json.dump(self.results, fdata, indent=4)

if __name__ == "__main__":
    """
    Template for collecting data about stress measures
    """

    experiment = Experiment(
        lambda G,x: Metrics(G,x).compute_stress_raw(), #Given graph G and pos matrix x, computes normalized stress
        "raw"                                          # str name of stress metric, sets default of output json
    )
    experiment.conduct_experiment()             #Only run the first 10 for example (default is to run all)
    experiment.write_results()                          #Write out the results to json. Default is to results/{name}-results.json

