"""TCGA patient graph dataset and durable stratified split utilities."""
from __future__ import annotations

import json
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data

ROOT = Path(__file__).resolve().parents[2]


def _read_labels(path: Path) -> pd.DataFrame:
    labels = pd.read_csv(path)
    required = {"sample_id", "label"}
    if not required.issubset(labels.columns):
        raise ValueError(f"{path} must have columns {sorted(required)}; no labels are inferred from expression.")
    labels = labels.loc[:, ["sample_id", "label"]].dropna().drop_duplicates("sample_id")
    if labels.empty:
        raise ValueError("The label table contains no usable sample_id/label rows.")
    return labels


def load_tcga_graph_dataset(expression_path: Path, graph_path: Path, labels_path: Path) -> tuple[list[Data], list[str], dict[str, int], list[str]]:
    """Return one PyG graph per labeled TCGA sample.

    ``labels_path`` is an explicit clinical/subtype-derived table. Its labels
    are encoded deterministically and never generated from features.
    """
    expression = pd.read_csv(expression_path, index_col="sample_id")
    expression.index = expression.index.astype(str)
    labels = _read_labels(labels_path)
    labels["sample_id"] = labels["sample_id"].astype(str)
    joined = expression.join(labels.set_index("sample_id"), how="inner")
    if joined.empty:
        raise ValueError("No label sample_id matches expression sample IDs. Check TCGA barcode conventions.")
    genes = expression.columns.astype(str).tolist()
    graph = nx.read_graphml(graph_path)
    missing = set(genes).difference(graph.nodes)
    if missing:
        raise ValueError("Graph nodes do not match expression genes; rebuild the graph from this expression file.")
    node_names = genes  # feature order is canonical and remains checkpoint metadata
    node_map = {gene: i for i, gene in enumerate(node_names)}
    edges: list[list[int]] = []
    for source, target in graph.edges():
        edges.extend(([node_map[source], node_map[target]], [node_map[target], node_map[source]]))
    if not edges:
        raise ValueError("Graph has no edges; lower the graph threshold.")
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    class_names = sorted(joined["label"].astype(str).unique().tolist())
    label_map = {name: idx for idx, name in enumerate(class_names)}
    dataset: list[Data] = []
    sample_ids: list[str] = []
    for sample_id, row in joined.iterrows():
        x = torch.tensor(row[genes].to_numpy(dtype=np.float32), dtype=torch.float32).view(-1, 1)
        y = torch.tensor([label_map[str(row["label"])]], dtype=torch.long)
        dataset.append(Data(x=x, edge_index=edge_index, y=y, sample_id=str(sample_id)))
        sample_ids.append(str(sample_id))
    return dataset, sample_ids, label_map, node_names


def make_or_load_split(sample_ids: list[str], labels: list[int], split_path: Path, test_size: float, seed: int) -> dict[str, list[int]]:
    """Persist sample-ID split membership so GNN and XGBoost use identical splits."""
    if split_path.exists():
        stored = json.loads(split_path.read_text())
        positions = {sample: idx for idx, sample in enumerate(sample_ids)}
        try:
            return {key: [positions[s] for s in stored[key]] for key in ("train", "test")}
        except KeyError as exc:
            raise ValueError("Saved split does not match current samples; delete it to recreate.") from exc
    train, test = train_test_split(sample_ids, test_size=test_size, random_state=seed, stratify=labels)
    split_path.parent.mkdir(parents=True, exist_ok=True)
    split_path.write_text(json.dumps({"train": train, "test": test}, indent=2))
    positions = {sample: idx for idx, sample in enumerate(sample_ids)}
    return {"train": [positions[s] for s in train], "test": [positions[s] for s in test]}


def load_dataset():
    """Compatibility loader; supervised use requires ``load_tcga_graph_dataset``."""
    raise RuntimeError("Use load_tcga_graph_dataset(expression, graph, labels); expression alone has no target.")
