#render the graphs
import torch
from torch_geometric.data import Data
import networkx as nx
import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GRAPH_DIR = os.path.join(BASE_DIR, "data", "graphs")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")


def load_dataset():
    graph_file = os.path.join(GRAPH_DIR, "gene_graph.graphml")
    data_file = os.path.join(PROCESSED_DIR, "processed_tcga.csv")

    G = nx.read_graphml(graph_file)
    df = pd.read_csv(data_file)

    edge_index = []

    node_map = {node: idx for idx, node in enumerate(G.nodes())}

    for edge in G.edges():
        edge_index.append([
            node_map[edge[0]],
            node_map[edge[1]]
        ])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    x = torch.tensor(df.values, dtype=torch.float)

    data = Data(x=x, edge_index=edge_index)

    return data