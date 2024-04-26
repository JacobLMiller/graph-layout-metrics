
import networkx as nx
import pylab as plt
import os
from sklearn.metrics import pairwise_distances
import random


K = 1


def main():
    if not os.path.isdir("outputs"):
        os.makedirs("outputs")

    stress_comp_test_raw(5)


def stress_comp_test_raw(ntests):
    for nt in range(ntests):
        nnodes = random.randint(10, 100)
        prob = 0

        G = nx.erdos_renyi_graph(nnodes, prob)
        while random.randint(0, 9) != 0 or not nx.is_connected(G):
            prob += 0.01
            G = nx.erdos_renyi_graph(nnodes, prob)
        prob = round(prob * 100) / 100

        plot_er_stress_raw(G, nnodes, prob)


def plot_er_stress_raw(G, er_num, er_prob):
    D = dict(nx.all_pairs_shortest_path_length(G))
    D = [[D[i][j] for j in range(len(D[i]))] for i in range(len(D))]

    pos = {"spring": nx.spring_layout(G),
           "random": nx.random_layout(G),
           "spiral": nx.spiral_layout(G)}

    X = {k: pairwise_distances([pos[k][i]
                               for i in range(len(pos[k]))]) for k in pos}
    mins, inters = calc_alphas_raw(X, D)

    alphas = list()
    stresses = dict()
    max_alpha = max(max(mins.values()), max(inters.values())) + 0.5

    alpha = 0
    while alpha < max_alpha:
        for k in X:
            if k not in stresses:
                stresses[k] = list()
            stresses[k].append(stress_raw(X[k], D, alpha))
        alphas.append(alpha)
        alpha += max_alpha / 200

    colors = ['tab:red', 'tab:green', 'tab:blue', 'tab:orange', 'tab:pink']

    for i, k in enumerate(stresses.keys()):
        plt.plot(alphas, stresses[k], label=k,
                 c=colors[i % len(colors)], zorder=0)
        plt.scatter(mins[k], stress_raw(X[k], D, mins[k]),
                    c=colors[i % len(colors)], zorder=1)

    plt.vlines([a for a in inters.values() if a > 0], min(mins.values()), max([stress_raw(X[k], D, max_alpha)
               for k in X]), colors='tab:purple', zorder=-1, label='intersections')

    if not os.path.isdir("outputs/stress_comparison_raw"):
        os.makedirs("outputs/stress_comparison_raw")

    plt.xlabel("alpha scale factor")
    plt.ylabel("raw stress")
    plt.legend()
    i = 0

    while os.path.isfile(f"outputs/stress_comparison_raw/er_{er_num}_{str(er_prob).replace('.', '')}_{i}_plot.png"):
        i += 1
    plt.suptitle(
        f"raw stress for erdos reyni {er_num} nodes p={er_prob} test {i}")
    plt.savefig(
        f"outputs/stress_comparison_raw/er_{er_num}_{str(er_prob).replace('.', '')}_{i}_plot.png")
    plt.clf()

    for k in pos:
        nx.draw(G, pos[k])
        plt.savefig(
            f"outputs/stress_comparison_raw/er_{er_num}_{str(er_prob).replace('.','')}_{i}_{k}")
        plt.clf()


def calc_alphas_raw(X, D):
    mins = dict()
    inters = dict()
    for k1 in X:
        mins[k1] = min_alpha_raw(X[k1], D)
        for k2 in X:
            if k1 != k2 and (k2, k1) not in inters and (k1, k2) not in inters:
                inters[(k1, k2)] = intersect_alpha_raw(X[k1], X[k2], D)
    return mins, inters


def min_alpha_raw(X, D):
    num = 0
    den = 0
    for i in range(len(D)):
        for j in range(i+1, len(D)):
            num += X[i][j] * D[i][j]
            den += X[i][j] ** 2
    return num / den


def intersect_alpha_raw(X1, X2, D):
    num = 0
    den = 0
    for i in range(len(D)):
        for j in range(i+1, len(D)):
            num += (X1[i][j] - X2[i][j]) * D[i][j]
            den += X1[i][j] ** 2 - X2[i][j] ** 2
    num *= 2
    return num / den


def stress_raw(X, D, alpha):
    S = 0
    for i in range(len(D)):
        for j in range(i+1, len(D)):
            S += ((alpha * X[i][j] - D[i][j]) ** 2)
    return S


if __name__ == '__main__':
    main()
