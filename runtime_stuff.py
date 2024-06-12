import json
import os
import pylab as plt
import numpy as np

METNAMES = {
    "minopt": "Scale-normalized Stress",
    "sheppardscale": "Shepard Constant Stress",
    "sheppard": "Sheppard Goodness Score",
    "stress": "Normalized Stress",
    "raw": "Raw Stress",
    "kruskal": "Non-metric Stress",
    "kk": "KK Stress",
    "ratios": "Ratio Stress"
}

for name in os.listdir("results"):
    if 'results.json' in name:
        metric = '-'.join(name.split('-')[:-1])
        data = json.load(open(f"results/{name}", 'r'))

        nnodes = []
        times = []
        for gname in data:
            nnodes.append(data[gname]['nnodes'])
            times.append(sum(data[gname]['runtime'].values()))
            # print(data[gname]['nnodes'], sum(data[gname]['runtime'].values()))
        plt.scatter(nnodes, times, c='gray', s=20)

        reg2 = np.poly1d(np.polyfit(nnodes, times, 2))
        reg4 = np.poly1d(np.polyfit(nnodes, times, 4))
        x = np.linspace(min(nnodes), max(nnodes), 100)
        plt.plot(x, reg2(x), c='red', label='Quadratic Regression')
        if metric == 'ratios':
            plt.plot(x, reg4(x), c='blue', label='Quartic Regression')

        plt.xlabel('Number of Nodes')
        plt.ylabel('Runtime (s)')
        plt.legend()
        # plt.title(f"Rome Graph Runtimes for {METNAMES[metric]}")
        plt.title(f"SS Graph Runtimes for {METNAMES[metric]}")
        if metric != 'ratios':
            # plt.ylim(top=0.045)
            plt.ylim(top=4)
        plt.savefig(f"results/runtimes-{metric}.pdf")
        plt.clf()

        print(reg4)
