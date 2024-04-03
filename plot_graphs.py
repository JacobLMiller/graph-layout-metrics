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

from modules.metrics_h import MetricsH
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

    graph_corpus = list(graph_io.get_corpus_file_names())

    # for gname in tqdm.tqdm(graph_corpus):
    for gname in tqdm.tqdm(graph_corpus):
        for emb in ["random", "stress", "tsnet"]:
            Gnx, X = graph_io.load_graph_with_embedding(gname, emb)    
            if Gnx.number_of_nodes() <= 1500: continue
            G = convert_nx_to_gt(Gnx)
            print(G.num_vertices())

            fig, axes = plt.subplots(2,2)
            ax1, ax2 = axes[0]
            ax3, ax4 = axes[1]
            fig.set_size_inches(10,10)
            draw(G, X, ax1)
            sheppard_diagram(Gnx, X, ax2)
            normal_vals = stress_curve(Gnx, X, ax3)
            bar_chart(Gnx,X,ax4, normal_vals)

            fig.savefig(f"drawings/{gname}_{emb}.png")
            plt.close(fig)
            plt.close()

