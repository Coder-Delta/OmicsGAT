"""Reusable data and inference helpers for the Streamlit application."""
from __future__ import annotations
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import torch
from src.data.dataset import load_tcga_graph_dataset
from src.evaluate import load_model

def load_checkpoint(path: str | Path):
    """Load a trained GAT/GCN model and its self-describing checkpoint metadata."""
    return load_model(Path(path), torch.device("cuda" if torch.cuda.is_available() else "cpu"))

def load_processed_data(checkpoint: dict) -> pd.DataFrame:
    return pd.read_csv(checkpoint["expression_path"], index_col="sample_id")

def load_labeled_graph_data(checkpoint: dict):
    return load_tcga_graph_dataset(Path(checkpoint["expression_path"]), Path(checkpoint["graph_path"]), Path(checkpoint["labels_path"]))

def run_inference(model, data) -> np.ndarray:
    device=next(model.parameters()).device; model.eval()
    with torch.no_grad(): return torch.softmax(model(data.x.to(device),data.edge_index.to(device)),dim=1).squeeze(0).cpu().numpy()

def format_prediction(probabilities: np.ndarray, class_names: list[str]) -> pd.DataFrame:
    return pd.DataFrame({"class":class_names,"probability":probabilities}).sort_values("probability",ascending=False)

def find_important_genes(model, data, node_names: list[str], top_n: int=15) -> pd.DataFrame:
    """Rank incoming first-layer GAT attention; unavailable for GCN models."""
    if not hasattr(model,"conv1") or model.__class__.__name__ != "GATModel": return pd.DataFrame(columns=["gene","attention"])
    device=next(model.parameters()).device; model.eval()
    with torch.no_grad(): _,(edges,alpha)=model.conv1(data.x.to(device),data.edge_index.to(device),return_attention_weights=True)
    score=np.zeros(len(node_names)); np.add.at(score,edges[1].cpu().numpy(),alpha.mean(1).cpu().numpy()); chosen=np.argsort(score)[::-1][:top_n]
    return pd.DataFrame({"gene":[node_names[i] for i in chosen],"attention":score[chosen]})

def load_result_json(path: str | Path) -> dict:
    path=Path(path)
    if not path.exists(): return {}
    try: return json.loads(path.read_text())
    except json.JSONDecodeError as exc: raise ValueError(f"Invalid JSON results file: {path}") from exc
