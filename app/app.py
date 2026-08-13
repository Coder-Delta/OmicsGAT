from __future__ import annotations
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
import streamlit as st
import torch
from app.utils import find_important_genes, format_prediction, load_checkpoint, load_labeled_graph_data, load_result_json, run_inference

st.set_page_config(page_title="OmicsGAT", layout="wide")
st.title("OmicsGAT: TCGA graph classifier")
st.caption("Predictions are research outputs, not clinical advice. Labels must come from an explicit TCGA clinical/subtype source.")
checkpoint_dir=ROOT/"checkpoints"; choices=sorted(checkpoint_dir.glob("*.pt"))
if not choices:
    st.error("No .pt checkpoint found in checkpoints/. Train a labeled model first."); st.stop()
chosen=st.sidebar.selectbox("Model checkpoint",choices,format_func=lambda p:p.name)
@st.cache_resource(show_spinner=False)
def cached_model(path: str): return load_checkpoint(path)
try: model,ckpt=cached_model(str(chosen)); dataset,ids,_,genes=load_labeled_graph_data(ckpt)
except Exception as exc: st.error(f"Could not load model data: {exc}"); st.stop()
sample=st.selectbox("TCGA sample",ids); item=dataset[ids.index(sample)]
uploaded=st.file_uploader("Or upload one already-processed sample CSV",type="csv",help="One row with exactly the checkpoint gene columns, using the same log/standardization as preprocessing.")
if uploaded is not None:
    try:
        import pandas as pd
        row=pd.read_csv(uploaded)
        extra=set(row.columns)-set(genes); missing=set(genes)-set(row.columns)
        if len(row) != 1 or extra or missing: raise ValueError("CSV must have exactly one row and exactly the checkpoint's gene columns.")
        item=dataset[0].clone(); item.x=torch.tensor(row.loc[:,genes].to_numpy(dtype="float32").reshape(-1,1)); sample="uploaded sample"
        st.success("Using uploaded processed sample.")
    except Exception as exc: st.error(f"Cannot use uploaded sample: {exc}"); st.stop()
if st.button("Run prediction",type="primary"):
    probabilities=run_inference(model,item); table=format_prediction(probabilities,ckpt["class_names"]); st.subheader("Class probabilities"); st.dataframe(table,use_container_width=True); st.bar_chart(table.set_index("class"))
    important=find_important_genes(model,item,genes)
    st.subheader("Important genes")
    if important.empty: st.info("Attention rankings are available only for GAT checkpoints.")
    else: st.dataframe(important,use_container_width=True)
st.subheader("Saved attention visualization")
attention=ROOT/"outputs/figures/gat_attention.png"
if attention.exists(): st.image(str(attention),use_container_width=True)
else: st.info("Run src/visualize_attention.py to create an attention figure.")
st.subheader("Evaluation metrics")
metrics_files=sorted((ROOT/"outputs/results").glob("*_metrics.json"))
if metrics_files:
    selected=st.selectbox("Metrics file",metrics_files,format_func=lambda p:p.name); st.json(load_result_json(selected))
else: st.info("Run src/evaluate.py after training to create metrics.")
