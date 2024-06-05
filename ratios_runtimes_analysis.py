import json
import numpy as np
import pylab as plt

data = json.load(open("results/ratios-runtimes.json", 'r'))
nnodes = []
times = []

for gname in data:
    nnodes.append(data[gname]['nnodes'])
    times.append(data[gname]['time'])

plt.scatter(nnodes, times)

x = np.linspace(10, 50, 100)
y = [2.8e-6 * n ** 4 for n in x]
plt.plot(x, y, c='red')

plt.xlabel("Number of Nodes")
plt.ylabel("Runtime (s)")
plt.savefig("results/ratios-runtimes.png")
plt.clf()