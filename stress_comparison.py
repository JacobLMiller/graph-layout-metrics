
import networkx as nx
import pylab as plt
import os
from sklearn.metrics import pairwise_distances


def main():
    if not os.path.isdir("outputs"):
        os.makedirs("outputs")
    if not os.path.isdir("outputs/stress_comparison"):
        os.makedirs("outputs/stress_comparison")
    if not os.path.isdir("outputs/orders"):
        os.makedirs("outputs/orders")

    ntests = 10
    for gorilla in range(ntests):
        nnodes = [10, 20, 50]
        probs = [0.1, 0.2, 0.5]

        orders = []
        count = 0

        t = 0
        while os.path.isdir(f"outputs/orders/test{t}"):
            t += 1
        os.makedirs(f"outputs/orders/test{t}")

        for i in range(5):
            while len(orders) < 6:
                for n in nnodes:
                    for p in probs:
                        plot_er_stress_orders(n, p, orders, f"test{t}")
                        print(count, orders)
                        count += 1
                # for i in range(10):
                #     plot_er_stress(n, p)


def plot_er_stress(er_num, er_prob):
    G = nx.erdos_renyi_graph(2, 0)
    while not nx.is_connected(G):
        G = nx.erdos_renyi_graph(er_num, er_prob)

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

    alphas = list()
    S_spr = list()
    S_r = list()
    S_spi = list()
    max_alpha = max(spr_r_alpha, spr_spi_alpha, spi_r_alpha,
                    min_alpha_spr, min_alpha_r, min_alpha_spi) + 0.5
    alpha = 0
    while alpha < max_alpha:
        S_spr.append(stress(X_spr, D, alpha))
        S_r.append(stress(X_r, D, alpha))
        S_spi.append(stress(X_spi, D, alpha))
        alphas.append(alpha)
        alpha += max_alpha / 200

    plt.plot(alphas, S_spr, label='spring', c='tab:red', zorder=0)
    plt.plot(alphas, S_r, label='random', c='tab:green', zorder=0)
    plt.plot(alphas, S_spi, label='spiral', c='tab:blue', zorder=0)

    plt.scatter(min_alpha_spr, stress(
        X_spr, D, min_alpha_spr), c='tab:red', zorder=1)
    plt.scatter(min_alpha_r, stress(X_r, D, min_alpha_r),
                c='tab:green', zorder=1)
    plt.scatter(min_alpha_spi, stress(
        X_spi, D, min_alpha_spi), c='tab:blue', zorder=1)

    plt.vlines([a for a in [spr_r_alpha, spr_spi_alpha, spi_r_alpha] if a > 0],
               min(min_alpha_spr, min_alpha_r, min_alpha_spi),
               max(stress(X_spr, D, max_alpha), stress(
                   X_r, D, max_alpha), stress(X_spi, D, max_alpha)),
               colors='tab:purple', zorder=-1, label='intersections')
    plt.xlabel("alpha scale factor")
    plt.ylabel("normalized stress")
    plt.legend()
    i = 0

    while os.path.isfile(f"outputs/stress_comparison/er_{er_num}_{str(er_prob).replace('.', '')}_{i}.png"):
        i += 1
    plt.suptitle(f"stress for erdos reyni {er_num} nodes p={er_prob} test {i}")
    plt.savefig(
        f"outputs/stress_comparison/er_{er_num}_{str(er_prob).replace('.', '')}_{i}.png")
    plt.clf()


def plot_er_stress_orders(er_num, er_prob, orders, test_str):
    G = nx.erdos_renyi_graph(2, 0)
    while not nx.is_connected(G):
        G = nx.erdos_renyi_graph(er_num, er_prob)

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

    mid_alphas = [0, spr_r_alpha, spr_spi_alpha, spi_r_alpha]
    mid_alphas.sort()
    orders_changed = False
    for a in [sum(mid_alphas[i:i+2]) / 2 for i in range(3)]:
        if a > 0:
            order = {"spr": stress(X_spr, D, a), "r": stress(
                X_r, D, a), "spi": stress(X_spi, D, a)}
            order = sorted(order.items(), key=lambda x: x[1])
            order = [o[0] for o in order]
            if order not in orders:
                orders.append(order)
                orders_changed = True
    if not orders_changed:
        return

    alphas = list()
    S_spr = list()
    S_r = list()
    S_spi = list()
    max_alpha = max(spr_r_alpha, spr_spi_alpha, spi_r_alpha,
                    min_alpha_spr, min_alpha_r, min_alpha_spi) + 0.5
    alpha = 0
    while alpha < max_alpha:
        S_spr.append(stress(X_spr, D, alpha))
        S_r.append(stress(X_r, D, alpha))
        S_spi.append(stress(X_spi, D, alpha))
        alphas.append(alpha)
        alpha += max_alpha / 200

    plt.plot(alphas, S_spr, label='spring', c='tab:red', zorder=0)
    plt.plot(alphas, S_r, label='random', c='tab:green', zorder=0)
    plt.plot(alphas, S_spi, label='spiral', c='tab:blue', zorder=0)

    plt.scatter(min_alpha_spr, stress(
        X_spr, D, min_alpha_spr), c='tab:red', zorder=1)
    plt.scatter(min_alpha_r, stress(X_r, D, min_alpha_r),
                c='tab:green', zorder=1)
    plt.scatter(min_alpha_spi, stress(
        X_spi, D, min_alpha_spi), c='tab:blue', zorder=1)

    plt.vlines([a for a in [spr_r_alpha, spr_spi_alpha, spi_r_alpha] if a > 0],
               min(min_alpha_spr, min_alpha_r, min_alpha_spi),
               max(stress(X_spr, D, max_alpha), stress(
                   X_r, D, max_alpha), stress(X_spi, D, max_alpha)),
               colors='tab:purple', zorder=-1, label='intersections')
    plt.xlabel("alpha scale factor")
    plt.ylabel("normalized stress")
    plt.legend()
    i = 0

    while os.path.isfile(f"outputs/orders/{test_str}/er_{er_num}_{str(er_prob).replace('.', '')}_{i}_plot.png"):
        i += 1
    plt.suptitle(f"stress for erdos reyni {er_num} nodes p={er_prob} test {i}")
    plt.savefig(
        f"outputs/orders/{test_str}/er_{er_num}_{str(er_prob).replace('.', '')}_{i}_plot.png")
    plt.clf()

    nx.draw(G, pos_spr)
    plt.savefig(
        f"outputs/orders/{test_str}/er_{er_num}_{str(er_prob).replace('.','')}_{i}_spr")
    plt.clf()

    nx.draw(G, pos_r)
    plt.savefig(
        f"outputs/orders/{test_str}/er_{er_num}_{str(er_prob).replace('.','')}_{i}_r")
    plt.clf()

    nx.draw(G, pos_spi)
    plt.savefig(
        f"outputs/orders/{test_str}/er_{er_num}_{str(er_prob).replace('.','')}_{i}_spi")
    plt.clf()


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
