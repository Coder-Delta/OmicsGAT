"""Train a GAT or GCN on explicitly supplied TCGA clinical/subtype labels."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch_geometric.loader import DataLoader

from src.data.dataset import load_tcga_graph_dataset, make_or_load_split
from src.models.gat_model import GATModel
from src.models.gcn_model import GCNModel

ROOT = Path(__file__).resolve().parents[1]


def _seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def train(args: argparse.Namespace) -> Path:
    config = yaml.safe_load(args.config.read_text()) or {}
    training, evaluation = config.get("training", {}), config.get("evaluation", {})
    _seed(args.seed if args.seed is not None else training.get("seed", 42))
    dataset, sample_ids, label_map, node_names = load_tcga_graph_dataset(args.expression, args.graph, args.labels)
    split = make_or_load_split(sample_ids, [int(d.y.item()) for d in dataset], args.split,
                               args.test_size or evaluation.get("test_size", .2),
                               args.seed if args.seed is not None else training.get("seed", 42))
    train_loader = DataLoader([dataset[i] for i in split["train"]], batch_size=args.batch_size or training.get("batch_size", 16), shuffle=True)
    test_loader = DataLoader([dataset[i] for i in split["test"]], batch_size=args.batch_size or training.get("batch_size", 16))
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    cls = GATModel if args.model == "gat" else GCNModel
    model = cls(1, args.hidden_dim or training.get("hidden_dim", 64), len(label_map), args.heads or config.get("model", {}).get("heads", 4), args.dropout if args.dropout is not None else training.get("dropout", .2)).to(device)
    optimizer = Adam(model.parameters(), lr=args.learning_rate or training.get("learning_rate", 1e-3), weight_decay=training.get("weight_decay", 1e-4))
    loss_fn = CrossEntropyLoss(); best_loss = float("inf")
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, (args.epochs or training.get("epochs", 100)) + 1):
        model.train(); total = 0.0
        for batch in train_loader:
            batch = batch.to(device); optimizer.zero_grad()
            loss = loss_fn(model(batch.x, batch.edge_index, batch.batch), batch.y)
            loss.backward(); optimizer.step(); total += loss.item() * batch.num_graphs
        model.eval(); validation = 0.0
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                validation += loss_fn(model(batch.x, batch.edge_index, batch.batch), batch.y).item() * batch.num_graphs
        validation /= len(split["test"])
        if validation < best_loss:
            best_loss = validation
            torch.save({"model_state_dict": model.state_dict(), "model_type": args.model, "input_dim": 1,
                        "hidden_dim": model.classifier.in_features, "output_dim": len(label_map), "heads": args.heads or config.get("model", {}).get("heads", 4), "dropout": model.dropout,
                        "class_names": [name for name, _ in sorted(label_map.items(), key=lambda x: x[1])], "node_names": node_names,
                        "expression_path": str(args.expression), "graph_path": str(args.graph), "labels_path": str(args.labels), "split_path": str(args.split)}, args.checkpoint)
        print(f"epoch={epoch:03d} train_loss={total / len(split['train']):.4f} test_loss={validation:.4f}")
    return args.checkpoint


def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--config", type=Path, default=ROOT / "configs/config.yaml")
    p.add_argument("--expression", type=Path, default=ROOT / "data/processed/tcga_expression.csv"); p.add_argument("--graph", type=Path, default=ROOT / "data/graphs/gene_graph.graphml")
    p.add_argument("--labels", type=Path, default=ROOT / "data/processed/tcga_labels.csv", help="CSV with sample_id,label sourced from TCGA clinical/subtype data")
    p.add_argument("--split", type=Path, default=ROOT / "data/processed/train_test_split.json"); p.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints/best_gat.pt")
    p.add_argument("--model", choices=("gat", "gcn"), default="gat"); p.add_argument("--epochs", type=int); p.add_argument("--batch-size", type=int); p.add_argument("--hidden-dim", type=int); p.add_argument("--heads", type=int); p.add_argument("--dropout", type=float); p.add_argument("--learning-rate", type=float); p.add_argument("--test-size", type=float); p.add_argument("--seed", type=int); p.add_argument("--cpu", action="store_true")
    print(f"Saved checkpoint: {train(p.parse_args())}")


if __name__ == "__main__": main()
