"""Evaluate a saved GAT/GCN checkpoint on its persisted TCGA test split."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, precision_recall_fscore_support, roc_auc_score
from torch_geometric.loader import DataLoader
from src.data.dataset import load_tcga_graph_dataset, make_or_load_split
from src.models.gat_model import GATModel
from src.models.gcn_model import GCNModel

ROOT = Path(__file__).resolve().parents[1]

def calculate_metrics(y, predicted, probabilities, class_names):
    precision, recall, f1, _ = precision_recall_fscore_support(y, predicted, average="weighted", zero_division=0)
    result = {"accuracy": float(accuracy_score(y, predicted)), "precision": float(precision), "recall": float(recall), "f1": float(f1)}
    try:
        result["roc_auc"] = float(roc_auc_score(y, probabilities[:, 1] if len(class_names) == 2 else probabilities, multi_class="ovr", average="weighted"))
    except ValueError:
        result["roc_auc"] = None  # e.g. a test set missing one class
    return result

def load_model(checkpoint_path: Path, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_cls = GATModel if ckpt["model_type"] == "gat" else GCNModel
    model = model_cls(ckpt["input_dim"], ckpt["hidden_dim"], ckpt["output_dim"], ckpt.get("heads", 1), ckpt.get("dropout", .2)).to(device)
    model.load_state_dict(ckpt["model_state_dict"]); model.eval()
    return model, ckpt

def evaluate(checkpoint_path: Path, results_path: Path, figure_path: Path, *, expression: Path | None = None, graph: Path | None = None, labels: Path | None = None, split: Path | None = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); model, ckpt = load_model(checkpoint_path, device)
    expression, graph, labels, split = expression or Path(ckpt["expression_path"]), graph or Path(ckpt["graph_path"]), labels or Path(ckpt["labels_path"]), split or Path(ckpt["split_path"])
    dataset, sample_ids, label_map, _ = load_tcga_graph_dataset(expression, graph, labels)
    class_names = ckpt["class_names"]
    if class_names != [k for k, _ in sorted(label_map.items(), key=lambda item:item[1])]: raise ValueError("Checkpoint labels differ from current labels file.")
    indices = make_or_load_split(sample_ids, [int(d.y.item()) for d in dataset], split, .2, 42)["test"]
    loader = DataLoader([dataset[i] for i in indices], batch_size=32); all_y=[]; all_p=[]; all_prob=[]
    with torch.no_grad():
        for batch in loader:
            batch=batch.to(device); logits=model(batch.x,batch.edge_index,batch.batch); all_y.extend(batch.y.cpu().tolist()); all_p.extend(logits.argmax(1).cpu().tolist()); all_prob.extend(torch.softmax(logits,1).cpu().tolist())
    values = calculate_metrics(np.array(all_y), np.array(all_p), np.array(all_prob), class_names) | {"class_names":class_names, "n_test":len(all_y), "checkpoint":str(checkpoint_path)}
    results_path.parent.mkdir(parents=True, exist_ok=True); results_path.write_text(json.dumps(values, indent=2))
    figure_path.parent.mkdir(parents=True, exist_ok=True); fig, ax=plt.subplots(figsize=(6,5)); ConfusionMatrixDisplay.from_predictions(all_y, all_p, display_labels=class_names, cmap="Blues", colorbar=False, ax=ax); ax.set_title("TCGA test-set confusion matrix"); fig.tight_layout(); fig.savefig(figure_path, dpi=180); plt.close(fig)
    return values

def main():
    p=argparse.ArgumentParser(); p.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints/best_gat.pt"); p.add_argument("--results", type=Path, default=ROOT / "outputs/results/gat_metrics.json"); p.add_argument("--figure", type=Path, default=ROOT / "outputs/figures/gat_confusion_matrix.png"); p.add_argument("--expression", type=Path); p.add_argument("--graph", type=Path); p.add_argument("--labels", type=Path); p.add_argument("--split", type=Path); args=p.parse_args(); print(json.dumps(evaluate(args.checkpoint,args.results,args.figure,expression=args.expression,graph=args.graph,labels=args.labels,split=args.split),indent=2))
if __name__ == "__main__": main()
