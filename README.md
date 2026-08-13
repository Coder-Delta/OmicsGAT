# OmicsGAT

OmicsGAT trains graph-level GAT and GCN classifiers on real TCGA expression
profiles. Each patient/sample is a graph: nodes are genes, each node has that
patient's processed expression value, and the shared edges come from a
co-expression graph. An XGBoost model provides a tabular baseline.

## Important data requirement

`tcga_RSEM_gene_tpm.gz` is expression only. It does **not** provide a
classification target, so this project will not train until you provide a
clinical or subtype-derived table at `data/processed/tcga_labels.csv`:

```csv
sample_id,label
TCGA-XX-0001-01A,BRCA_LumA
TCGA-XX-0002-01A,BRCA_Basal
```

Use one documented source and endpoint (for example TCGA clinical/subtype
annotations), retain its provenance outside this repository, and ensure its
sample barcodes match the Xena expression columns. Do not derive labels from
expression values or make up labels. The pipeline keeps these distinct:

- features: `data/processed/tcga_expression.csv` (sample × gene)
- labels: `data/processed/tcga_labels.csv` (`sample_id`, `label`)
- graph: `data/graphs/gene_graph.graphml` (gene × gene)
- sample IDs: the index of the expression matrix and the split JSON

## Setup and run

```bash
python -m pip install -r requirements.txt
python src/data/download_tcga.py
python src/data/preprocess.py --max-genes 2000
python src/data/build_graphs.py --threshold 0.7
# Create data/processed/tcga_labels.csv from your chosen TCGA clinical/subtype source.
python src/train.py --model gat --epochs 100
python src/evaluate.py --checkpoint checkpoints/best_gat.pt
python src/visualize_attention.py --checkpoint checkpoints/best_gat.pt
python src/models/baseline_xgb.py
streamlit run app/app.py
```

Run GAT training first so `data/processed/train_test_split.json` is created;
the XGBoost command then uses exactly that split. The training checkpoint stores
the paths, class order, node/gene order, and model dimensions required by
evaluation, attention extraction, and the app.

Outputs are written under `checkpoints/`, `outputs/results/`, and
`outputs/figures/`. Metrics and attention rankings describe the supplied data
only; they are not clinical validation.
