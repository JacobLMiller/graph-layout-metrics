import json
import networkx as nx
import numpy as np


def json_to_graph(fname):
    with open(fname, "r") as fdata:
        js_graph = json.load(fdata)

    G = nx.Graph()
    G.add_nodes_from([v["id"] for v in js_graph["nodes"]])
    G.add_edges_from([(e["source"], e["target"]) for e in js_graph["edges"]])
    G = nx.convert_node_labels_to_integers(G)

    return G


def save_graph_with_embedding(G, X, fname):
    """
    Writes graph to graphml with embeding X. Expected 2 dimensional embedding, 
    with X as a n x 2 numpy array. 
    Saves output to "embeddings/{fname}"
    """
    for i, (v, data) in enumerate(G.nodes(data=True)):
        data['x'] = X[i, 0]
        data['y'] = X[i, 1]

    nx.write_graphml(G, f"embeddings/{fname}")


def load_graph_with_embedding(gname, alg):
    G = nx.read_graphml(f"embeddings/{gname}_{alg}.graphml")
    G = nx.convert_node_labels_to_integers(G)

    X = np.zeros((G.number_of_nodes(), 2))
    for i, (v, data) in enumerate(G.nodes(data=True)):
        X[i, 0] = data['x']
        X[i, 1] = data['y']

    return G, X


def load_graphml(fname):
    G = nx.read_graphml(f"graphs/{fname}")
    G = nx.convert_node_labels_to_integers(G)
    name = fname.split(".")[0]
    G.graph.update({"gname": name})
    return G

def get_corpus_file_names():
    import os
    for fname in os.listdir("graphs"):
        yield fname.split(".")[0]

def get_corpus_SS(size_limit=1000):
    import os 
    for fname in os.listdir("SS_graphs"):
        G = load_txt("SS_graphs/" + fname)
        if size_limit and G.number_of_nodes() > size_limit: continue
        yield G
        
def load_txt(fname):
    E = np.loadtxt(fname)
    G = nx.Graph()
    G.add_edges_from(E.tolist())
    G = nx.convert_node_labels_to_integers(G)
    G.graph['gname'] = fname.split(".")[0].replace("SS_graphs/", "")
    return G

def load_embedding(G, alg):
    try:
        return np.load(f"embeddings/{G.graph['gname']}_{alg}.npy")
    except: 
        return None



def get_apsp(G):
    import os
    if not os.path.isdir("apsps"):
        os.makedirs("apsps")

    name = G.graph["gname"]
    try:
        return np.loadtxt(f"apsps/{name}_apsp.txt")
    except:

        d = np.zeros((G.number_of_nodes(), G.number_of_nodes()))

        dists = dict(nx.all_pairs_shortest_path_length(G))
        for u in dists:
            for v in range(u):
                d[u, v] = dists[u][v]
                d[v, u] = dists[u][v]
        np.savetxt(f"apsps/{name}_apsp.txt", d)
        return d
