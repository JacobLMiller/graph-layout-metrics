import numpy as np 
import networkx as nx

class Layout():
    def __init__(self, G: nx.Graph, alg: str):
        assert alg in ["random", "stress", "tsnet"]

        self.G = G
        self.N = G.number_of_nodes()
        self.alg = alg 

        self.X = None
    
    def random_layout(self):
        return np.random.uniform(0, 1, (self.N, 2))
    
    def stress_layout(self):
        import s_gd2
        I = [u for u,v in self.G.edges()]
        J = [v for u,v in self.G.edges()]
        return s_gd2.layout_convergent(I,J)
    
    def tsnet_layout(self):
        from sklearn.manifold import TSNE 
        from modules.graph_io import get_apsp
        return TSNE(init="random", metric="precomputed", perplexity=min(30, self.G.number_of_nodes()-1)).fit_transform(get_apsp(self.G))
    
    def compute(self):
        if self.alg == "random":
            self.X = self.random_layout()
        elif self.alg == "stress": 
            self.X = self.stress_layout()
        elif self.alg == "tsnet":
            self.X = self.tsnet_layout()

        return self.X
    
    def store_layout(self):
        assert self.X is not None
        from modules.graph_io import save_graph_with_embedding
        # save_graph_with_embedding(self.G, self.X, f"{self.G.graph['gname']}_{self.alg}.graphml")
        np.save(f"embeddings/{self.G.graph['gname']}_{self.alg}.npy", self.X)

if __name__ == "__main__":
    import os
    import tqdm
    from modules.graph_io import load_graphml, get_corpus_SS, load_txt, load_embedding

    if not os.path.isdir("embeddings"):
        os.makedirs("embeddings")


    # graph_names = os.listdir("graphs")
    # for gname in tqdm.tqdm(graph_names):
    for gname in tqdm.tqdm(os.listdir("SS_graphs")):
    # for G in tqdm.tqdm(get_corpus_SS()):
        # G = load_graphml(gname)
        G = load_txt("SS_graphs/" + gname)

        print(G.number_of_nodes(), G.number_of_edges())
        if not isinstance(load_embedding(G,"random"), np.ndarray):
            emb = Layout(G, "random")
            emb.compute()
            emb.store_layout()

        if not isinstance(load_embedding(G,"stress"), np.ndarray):
            emb = Layout(G, "stress")
            emb.compute()
            emb.store_layout()

        if not isinstance(load_embedding(G,"tsnet"), np.ndarray):
            emb = Layout(G, "tsnet")
            emb.compute()
            emb.store_layout()            
