import graph_tool.all as gt 
import networkx as nx
import numpy as np 
import pylab as plt
import tqdm
from sklearn.metrics import pairwise_distances
from modules import graph_io

plt.switch_backend("cairo")

def convert_nx_to_gt(G_NX):
    H = nx.convert_node_labels_to_integers(G_NX)
    V = H.number_of_nodes()
    E = [(u,v) for u,v in G_NX.edges()]

    G = gt.Graph(directed=False)
    G.add_vertex(V)
    G.add_edge_list(E)
    return G

def draw(G,X,ax):
    pos = G.new_vp("vector<float>")
    pos.set_2d_array(X.T)
    gt.graph_draw(G,pos, mplfig=ax)    

def sheppard_diagram(G,X,ax):
    D = graph_io.get_apsp(G)

    output_dists = pairwise_distances(X)

    xij = output_dists[ np.triu_indices( output_dists.shape[0], k=1 ) ]
    dij  = D[ np.triu_indices( D.shape[0], k=1 ) ]

    sorted_indices = np.argsort(dij)
    dij = dij[sorted_indices]
    xij = xij[sorted_indices]    
    
    ax.scatter(xij, dij)

from modules.metrics import Metrics
from modules.normalization import Normalize

def stress_curve(G,X,ax):

    M = Metrics(G,X)
    N = Normalize(G, X)

    normal_factors = {
        "Identity": N.identity(),
        "Unit Norm": N.unit_norm(),
        "Optimize": None
    }

    #A range of alphas between 1e-12 and 20, evenly spaced with 500 samples
    alpha_spectrum = np.linspace(1e-12, 10, 100)

    stress = np.zeros_like(alpha_spectrum)
    for i, alpha in enumerate(alpha_spectrum):
        M.setX(alpha * X)
        stress[i] = M.compute_stress_norm()

    normal_factors["Optimize"] = alpha_spectrum[ np.argmin(stress) ]

    ax.plot(alpha_spectrum, stress, label="stress")

    y_min = np.min(stress)
    y_max = np.max(stress)
    interp_space = 20
    for name, val in normal_factors.items():
        ax.plot([val] * interp_space, np.linspace(y_min,y_max,interp_space), label=name)

    ax.legend()
    ax.set_xlabel("alpha scale factor")
    ax.set_ylabel("stress")
    ax.set_title(f"stress curve")

    return normal_factors

def bar_chart(G,X,ax, normal_factors):
    MH = MetricsH(G, X)
    M = Metrics(G,X)

    fields = ["raw", "unit_norm", "optimized", "kruskal"]
    vals = list()
    for name, val in normal_factors.items():
        M.setX(val * X)
        vals.append(M.compute_stress_norm())
    vals.append(MH.compute_stress_kruskal())


    ax.bar(fields, vals, label=fields)

    ax.set_yscale("log")

    ax.set_ylabel('stress value')
    ax.set_title('Stress comparison')
    # ax.legend()




if __name__ == "__main__":

    for Gnx in graph_io.get_corpus_SS():
        if Gnx.graph['gname'] == "dwt_419":
            G = convert_nx_to_gt(Gnx)
            for alg in ["random", 'stress', 'tsnet', "sfdp", "neato", "twopi"]:
                X = graph_io.load_embedding(Gnx, alg)
                pos = G.new_vp("vector<float>")
                pos.set_2d_array(X.T)
                gt.graph_draw(G,pos, output=f"drawings/dwt_419_{alg}.png")    


