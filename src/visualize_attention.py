"""Extract and visualize GAT first-layer attention for one real TCGA sample."""
from __future__ import annotations
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
from src.data.dataset import load_tcga_graph_dataset
from src.evaluate import load_model

ROOT = Path(__file__).resolve().parents[1]

def attention_for_sample(checkpoint_path: Path, sample_id: str | None, output: Path, top_n: int = 20):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"); model, ckpt=load_model(checkpoint_path,device)
    if ckpt["model_type"] != "gat": raise ValueError("Attention visualization requires a GAT checkpoint.")
    data, ids, _, genes=load_tcga_graph_dataset(Path(ckpt["expression_path"]),Path(ckpt["graph_path"]),Path(ckpt["labels_path"]))
    index=ids.index(sample_id) if sample_id else 0
    if sample_id and sample_id not in ids: raise ValueError(f"Sample {sample_id!r} is not in labeled processed data.")
    graph=data[index].to(device); model.eval()
    with torch.no_grad():
        _, (edge_index, alpha)=model.conv1(graph.x, graph.edge_index, return_attention_weights=True)
    weights=alpha.mean(dim=1).cpu().numpy(); edges=edge_index.cpu().numpy().T
    # Directed messages: score a gene by attention carried into it.
    incoming=np.zeros(len(genes)); np.add.at(incoming, edges[:,1], weights)
    ranked=np.argsort(incoming)[::-1][:top_n]
    sub=nx.Graph();
    for pos, weight in zip(edges,weights):
        if pos[0] in ranked and pos[1] in ranked: sub.add_edge(genes[pos[0]],genes[pos[1]],weight=float(weight))
    output.parent.mkdir(parents=True,exist_ok=True); fig,(ax1,ax2)=plt.subplots(1,2,figsize=(15,6)); names=[genes[i] for i in ranked][::-1]; ax1.barh(names,incoming[ranked][::-1],color="#3b82f6"); ax1.set_title(f"Top attended genes: {ids[index]}"); ax1.set_xlabel("mean incoming attention")
    pos=nx.spring_layout(sub,seed=42); widths=[3*sub[u][v]["weight"] for u,v in sub.edges()]; nx.draw_networkx(sub,pos,ax=ax2,node_color="#bfdbfe",edge_color="#64748b",width=widths,font_size=8); ax2.set_title("Top-gene attention subgraph"); ax2.axis("off"); fig.tight_layout(); fig.savefig(output,dpi=180); plt.close(fig)
    return [(genes[i],float(incoming[i])) for i in ranked]

def main():
    p=argparse.ArgumentParser(); p.add_argument("--checkpoint",type=Path,default=ROOT/"checkpoints/best_gat.pt"); p.add_argument("--sample-id"); p.add_argument("--output",type=Path,default=ROOT/"outputs/figures/gat_attention.png"); p.add_argument("--top-n",type=int,default=20); args=p.parse_args(); print(attention_for_sample(args.checkpoint,args.sample_id,args.output,args.top_n))
if __name__ == "__main__": main()
