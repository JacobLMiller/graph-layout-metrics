import networkx as nx 
import numpy as np 
from modules import graph_io
from modules.metrics import Metrics
from layouts import Layout


if __name__ == "__main__":
    count  = 0 

    results = dict()
    for G in graph_io.get_corpus_SS():
        if G.number_of_nodes() < 500: 

            print(G.graph['gname'])
            #Will be precomputed ---------
            layouts = [
                Layout(G, "stress").compute(),
                Layout(G, "tsnet").compute(),
                Layout(G, "random").compute()
            ]
            #----------------------------

            stresses = {f"A{i}": Metrics(G,x).compute_stress_norm() for i,x in enumerate(layouts)}
            stresses['order'] = sorted(stresses.keys(), key=stresses.get)

            results[G.graph["gname"]] = stresses

            count += 1

        if count > 5: break

    import json 
    with open("test-results.json", 'w') as fdata:
        json.dump(results,fdata,indent=4)

#TODO send SGD paper to Hamlet, Jonah