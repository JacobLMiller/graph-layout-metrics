
import networkx as nx
import pylab as plt
import os
from sklearn.metrics import pairwise_distances
import random


def main():
    cont = 'y'
    while (cont.lower() == 'y'):
        print()
        interesting_range_test(1)
        cont = input("Another test? (y/n) ")


def interesting_range_test(ntests):
    for nt in range(ntests):
        nnodes = random.randint(10, 100)
        prob = 0

        G = nx.erdos_renyi_graph(nnodes, prob)
        while random.randint(0, 9) != 0 or not nx.is_connected(G):
            prob += 0.01
            G = nx.erdos_renyi_graph(nnodes, prob)
        prob = round(prob * 100) / 100

        find_interesting_range(G, nnodes, prob)


def find_interesting_range(G, er_num, er_prob):
    D = dict(nx.all_pairs_shortest_path_length(G))
    D = [[D[i][j] for j in range(len(D[i]))] for i in range(len(D))]

    pos_spr = nx.spring_layout(G)
    pos_r = nx.random_layout(G)
    pos_spi = nx.spiral_layout(G)

    X_spr = pairwise_distances([pos_spr[i] for i in range(len(pos_spr))])
    X_r = pairwise_distances([pos_r[i] for i in range(len(pos_r))])
    X_spi = pairwise_distances([pos_spi[i] for i in range(len(pos_spi))])

    min_alpha_spr = min_alpha(X_spr, D)
    min_alpha_r = min_alpha(X_r, D)
    min_alpha_spi = min_alpha(X_spi, D)

    spr_r_alpha = intersect_alpha(X_spr, X_r, D)
    spr_spi_alpha = intersect_alpha(X_spr, X_spi, D)
    spi_r_alpha = intersect_alpha(X_spi, X_r, D)

    max_alpha = max(spr_r_alpha, spr_spi_alpha, spi_r_alpha,
                    min_alpha_spr, min_alpha_r, min_alpha_spi)
    min_alpha_ = min(spr_r_alpha, spr_spi_alpha, spi_r_alpha,
                     min_alpha_spr, min_alpha_r, min_alpha_spi, 0)

    print(
        f"(alpha, stress) for erdos renyi graph with {er_num} nodes and p={er_prob}:")
    print("  Intersections:")
    print(
        f"    Spring & Random: ({round(spr_r_alpha, 3)}, {round(stress(X_spr, D, spr_r_alpha), 3)})")
    print(
        f"    Spring & Spiral: ({round(spr_spi_alpha, 3)}, {round(stress(X_spr, D, spr_spi_alpha), 3)})")
    print(
        f"    Spiral & Random: ({round(spi_r_alpha, 3)}, {round(stress(X_spi, D, spi_r_alpha), 3)})")
    print("  Minima:")
    print(
        f"    Spring: ({round(min_alpha_spr, 3)}, {round(stress(X_spr, D, min_alpha_spr), 3)})")
    print(
        f"    Random: ({round(min_alpha_r, 3)}, {round(stress(X_r, D, min_alpha_r), 3)})")
    print(
        f"    Spiral: ({round(min_alpha_spi, 3)}, {round(stress(X_spi, D, min_alpha_spi), 3)})")
    print(
        f"Interesting Range: [{round(min_alpha_, 3)}, {round(max_alpha, 3)}]")
    print()


def min_alpha(X, D):
    num = 0
    den = 0
    for i in range(len(D)):
        for j in range(i+1, len(D)):
            num += X[i][j] / D[i][j]
            den += (X[i][j] ** 2) / (D[i][j] ** 2)
    return num / den


def intersect_alpha(X1, X2, D):
    num = 0
    den = 0
    for i in range(len(D)):
        for j in range(i+1, len(D)):
            num += (X1[i][j] - X2[i][j]) / D[i][j]
            den += (X1[i][j] ** 2 - X2[i][j] ** 2) / (D[i][j] ** 2)
    num *= 2
    return num / den


def stress(X, D, alpha):
    S = 0
    for i in range(len(D)):
        for j in range(i+1, len(D)):
            S += ((alpha * X[i][j] - D[i][j]) ** 2) / (D[i][j] ** 2)
    return S


if __name__ == '__main__':
    main()
