#Here we will create the graphs
import os
import pandas as pd
import networkx as nx
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
GRAPH_DIR = os.path.join(BASE_DIR, "data", "graphs")

os.makedirs(GRAPH_DIR, exist_ok=True)

input_file = os.path.join(PROCESSED_DIR, "processed_tcga.csv")

print("Loading processed data...")

df = pd.read_csv(input_file)

# Correlation matrix
corr = df.corr()

# Create graph
G = nx.Graph()

threshold = 0.7

for i in range(len(corr.columns)):
    for j in range(i + 1, len(corr.columns)):
        value = corr.iloc[i, j]

        if abs(value) > threshold:
            G.add_edge(corr.columns[i], corr.columns[j], weight=float(value))

output_graph = os.path.join(GRAPH_DIR, "gene_graph.graphml")

nx.write_graphml(G, output_graph)

print("Graph saved!")
print("Nodes:", G.number_of_nodes())
print("Edges:", G.number_of_edges())