from modules import graph_io
from modules.metrics import Metrics
import json
import time
import os


def time_func(f, arg1, arg2):
    start = time.perf_counter()
    res = f(arg1, arg2)
    end = time.perf_counter()

    return res, end - start


class Experiment():
    def __init__(self, metric_function, metric_name):
        """
        metric_function(G, x) must be a function with two parameters,
        G: a networkx graph
        x: A dict/matrix of positions for G
        """
        self.metric_function = metric_function
        self.metric_name = metric_name
        self.results = dict()

    def conduct_experiment(self, SS=False, algs=["stress", "tsnet", "random", "neato", "sfdp", "twopi"], limit=None, size_limit=1000):
        # Calc length to show progress bar:
        total = len(os.listdir("SS_graphs")) if SS else len(
            os.listdir("Rome_graphs"))

        # Retrieve graph dataset
        corpus = graph_io.get_corpus_SS(
            size_limit) if SS else graph_io.get_corpus_Rome(size_limit)

        # Calculate stresses for every graph-algorithm pair
        from tqdm import tqdm
        for count, G in enumerate(tqdm(corpus, total=total)):

            layouts = [graph_io.load_embedding(G, alg) for alg in algs]

            stresses = dict()
            stresses['runtime'] = {}
            stresses['nnodes'] = G.number_of_nodes()

            for alg, x in zip(algs, layouts):
                try:
                    stress_val, time_val = time_func(
                        self.metric_function, G, x)
                    stresses[f"{alg}"] = stress_val
                    stresses['runtime'][f"{alg}"] = time_val
                except:
                    print(f"\nError: {G.graph['gname']}, {alg}")

            stresses['order'] = sorted([k for k in stresses.keys() if k not in [
                                        'runtime', 'nnodes']], key=stresses.get)

            self.results[G.graph['gname']] = stresses

            if limit and count >= limit:
                break

    def write_results(self, fname=None):
        if not os.path.isdir('results'):
            os.makedirs('results')

        if fname is None:
            fname = f"results/{self.metric_name}-results.json"
        with open(fname, 'w') as fdata:
            json.dump(self.results, fdata, indent=4)


if __name__ == "__main__":
    """
    Template for collecting data about stress measures
    """
    mets = [
        (lambda G, x: Metrics(G, x).compute_stress_raw(), "raw"),
        (lambda G, x: Metrics(G, x).compute_stress_kk(), "kk"),
        (lambda G, x: Metrics(G, x).compute_stress_norm(), "normalized-stress"),
        (lambda G, x: Metrics(G, x).compute_stress_minopt(), "minopt"),
        (lambda G, x: Metrics(G, x).compute_stress_sheppard(), "sheppard"),
        (lambda G, x: Metrics(G, x).compute_stress_sheppardscale(), "sheppardscale"),
        (lambda G, x: Metrics(G, x).compute_stress_kruskal(), "kruskal"),

        # (lambda G, x: Metrics(G, x).compute_stress_ratios(), "ratios")
    ]

    for m in mets:
        experiment = Experiment(*m)

        # Conduct experiments selected in mets (add size_limit=*)
        experiment.conduct_experiment(SS=False)

        # Write out the results to json. Default is to results/{name}-results.json
        experiment.write_results()
