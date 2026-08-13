"""Build a gene-gene co-expression graph from processed TCGA expression."""
from __future__ import annotations

import argparse
from pathlib import Path

import networkx as nx
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def build_correlation_graph(expression_path: Path, graph_path: Path, threshold: float = 0.7) -> Path:
    expression = pd.read_csv(expression_path, index_col="sample_id")
    if expression.shape[1] < 2:
        raise ValueError("At least two genes are required to build a graph.")
    corr = expression.corr()
    graph = nx.Graph()
    graph.add_nodes_from(expression.columns.astype(str))  # retain isolated genes
    for gene in corr.columns:
        for other, value in corr.loc[gene].items():
            if gene < other and abs(value) >= threshold:
                graph.add_edge(gene, other, weight=float(abs(value)), correlation=float(value))
    if graph.number_of_edges() == 0:
        raise ValueError("No edges passed the threshold; lower --threshold.")
    graph_path.parent.mkdir(parents=True, exist_ok=True)
    nx.write_graphml(graph, graph_path)
    return graph_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a TCGA co-expression graph.")
    parser.add_argument("--expression", type=Path, default=ROOT / "data/processed/tcga_expression.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "data/graphs/gene_graph.graphml")
    parser.add_argument("--threshold", type=float, default=0.7)
    args = parser.parse_args()
    path = build_correlation_graph(args.expression, args.output, args.threshold)
    print(f"Wrote graph to {path}")


if __name__ == "__main__":
    main()
