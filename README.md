# graph-layout-metrics
Repository to accompany submission to GD 2024. 

## Installation: 
An example to install the required packages with conda: 
```
conda create --name myenv 
conda activate myenv
conda install pip
pip install -r requirements.txt
```
Or install requirements.txt with your favorite python virtual environment. 

## Stress metrics: modules/metrics.py
The stress metric computation code can be found in modules/metrics.py. 
Each metric has the same interface, through the Metrics class. This class takes a graph and a graph layout. The graph must be in networkx format, and the layout can either be a dictionary with the vertex keys mapping to positions, or a numpy |V| x 2 matrix of positions. 
```
import networkx as nx
from modules.metrics import Metric

G = nx.Graph()

G.add_edge(0,1)
G.add_edge(1,2)
G.add_edge(2,3)
G.add_edge(3,0)

pos = {
    0: [0,0],
    1: [0,1],
    2: [1,1],
    3: [1,0]
}

M = Metric(G, pos)
stress_val = M.compute_stress_norm()
```

## modules/graph_io.py
Contains various helper functions to read and write graphs from file, as well as computing and storing all-pairs-shortest-paths.

## analysis.ipynb
A Jupyter notebook which formats the results from each metric into a single csv.

## experiment.py 
Code used to conduct experiment detailed in accompanying paper.

## graph_statistics.py
Script which computes various relevant graph statistics for graph collections. 

## layouts.py 
Generates several different layouts for graph collection and writes them to file. 

## orders_analysis.ipynb
Notebook which computes and stores the order of layout algorithms according to various stress measures. 

## queryRome.py 
Downloads and stores the RomeLib graph collection to the local directory.

## querySS.py 
Downloads and stores all graphs of less than 1000 vertices from the SuiteSparse matrix collection. 

## runtime_eval.py
Formats runtime results and stores in csv computed during the experiment. 