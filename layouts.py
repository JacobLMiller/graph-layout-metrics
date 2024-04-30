import numpy as np 
import networkx as nx

class Layout():
    def __init__(self, G: nx.Graph, alg: str):
        assert alg in ["random", "stress", "tsnet", 'sfdp']

        self.G = G
        self.N = G.number_of_nodes()
        self.alg = alg 

        self.X = None
    
    def dict_to_mat(self, pos):
        X = np.zeros((self.G.number_of_nodes(), 2))
        for i, (v, p) in enumerate(pos.items()):
            X[i] = p
        return X     

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
        return TSNE(init="random", learning_rate='auto', square_distances=True, metric="precomputed", perplexity=min(30, self.G.number_of_nodes()-1)).fit_transform(get_apsp(self.G))
    
    def sfdp_layout(self):
        pos = nx.nx_pydot.pydot_layout(self.G, prog='sfdp')
        return self.dict_to_mat(pos)

    def compute(self):
        if self.alg == "random":
            self.X = self.random_layout()
        elif self.alg == "stress": 
            self.X = self.stress_layout()
        elif self.alg == "tsnet":
            self.X = self.tsnet_layout()
        elif self.alg == "sfdp":
            self.X = self.sfdp_layout()

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


    for gname in tqdm.tqdm(os.listdir("SS_graphs")):
        G = load_txt("SS_graphs/" + gname)
        if G.number_of_nodes() > 1000: continue

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

        if not isinstance(load_embedding(G,"sfdp"), np.ndarray):
            emb = Layout(G, "sfdp")
            emb.compute()
            emb.store_layout()                 


        if not isinstance(load_embedding(G,"twopi"), np.ndarray):
            emb = Layout(G, "twopi")
            emb.compute()
            emb.store_layout()                 

        if not isinstance(load_embedding(G,"neato"), np.ndarray):
            emb = Layout(G, "neato")
            emb.compute()
            emb.store_layout()                             