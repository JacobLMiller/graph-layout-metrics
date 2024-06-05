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
        self.runtime = 0
        self.ratios_runtimes = {}

    def conduct_experiment(self, algs=["stress", "tsnet", "random", "sfdp", "neato", "twopi"], limit=None, size_limit=1000):
        # Calc length to show progress bar:
        from os import listdir
        # total = len(listdir("SS_graphs"))
        total = len(listdir("Rome_graphs"))

        import time

        from tqdm import tqdm
        # for count, G in enumerate(tqdm(graph_io.get_corpus_SS(size_limit), total=total)):
        for count, G in enumerate(tqdm(graph_io.get_corpus_Rome(size_limit), total=total)):

            start = time.time()
            layouts = [graph_io.load_embedding(G, alg) for alg in algs]

            try:
                stresses = {f"{alg}": self.metric_function(
                    G, x) for alg, x in zip(algs, layouts)}
                stresses['order'] = sorted(stresses.keys(), key=stresses.get)

                self.results[G.graph["gname"]] = stresses
            except:
                print(f"\nBad Graph: {G.graph['gname']}")

            total_time = time.time() - start
            self.runtime += total_time
            self.ratios_runtimes[G.graph["gname"]] = {
                "nnodes": G.number_of_nodes(), "nedges": G.number_of_edges(), "time": total_time}

            if limit and count >= limit:
                break

    def write_results(self, fname=None):
        if fname is None:
            fname = f"results/{self.metric_name}-results.json"
        with open(fname, 'w') as fdata:
            json.dump(self.results, fdata, indent=4)


if __name__ == "__main__":
    """
    Template for collecting data about stress measures
    """
    mets = [
        # (lambda G,x: Metrics(G,x).compute_stress_sheppardscale(), "sheppardscale"),
        # (lambda G,x: Metrics(G,x).compute_stress_sheppard(), "sheppard"),
        # (lambda G,x: Metrics(G,x).compute_stress_minopt(), "minopt"),
        # (lambda G,x: Metrics(G,x).compute_stress_norm(), "normalized-stress"),
        # (lambda G,x: Metrics(G,x).compute_stress_raw(), "raw"),
        # (lambda G,x: Metrics(G,x).compute_stress_unitball(), "unitball"),
        # (lambda G,x: Metrics(G,x).compute_stress_kruskal(), "kruskal"),
        # (lambda G,x: Metrics(G,x).compute_stress_kk(), "kk-rome")
        (lambda G, x: Metrics(G, x).compute_stress_ratios(), "ratios")
    ]

    # for gorilla in range(9):
    for m in mets:
        experiment = Experiment(*m)
        # Only run the first 10 for example (default is to run all)
        experiment.conduct_experiment(
            algs=["stress", "tsnet", "random", "neato", "sfdp", "twopi"], size_limit=50)
        # Write out the results to json. Default is to results/{name}-results.json
        experiment.write_results()
        
        data = json.load(open('results/runtimes.json', 'r'))
        mname = m[1]
        if mname not in data:
            data[mname] = []
        data[mname].append(experiment.runtime)
        json.dump(data, open('results/runtimes.json', 'w'), indent=4)

        if mname == "ratios":
            json.dump(experiment.ratios_runtimes, open('results/ratios-runtimes.json', 'w'), indent=4)
