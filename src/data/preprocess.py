"""Prepare the UCSC Xena TCGA TPM matrix for supervised graph learning.

The Xena matrix contains genes in rows and TCGA aliquots in columns.  This
module writes the transposed, patient/sample-by-gene feature matrix without
inventing a target.  A separate clinical/subtype table is required for labels.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]


def prepare_expression(input_path: Path, output_dir: Path, *, max_genes: int | None = None) -> Path:
    """Transpose a Xena gene-by-sample TPM matrix and save aligned metadata.

    ``max_genes`` optionally retains the most variable genes.  It is useful for
    making correlation graphs tractable, but must be recorded with the model.
    """
    frame = pd.read_csv(input_path, sep="\t", compression="infer", index_col=0)
    if frame.empty or frame.shape[1] < 2:
        raise ValueError(f"{input_path} does not look like a Xena expression matrix.")
    frame.index = frame.index.astype(str).str.split(".").str[0]
    frame = frame.apply(pd.to_numeric, errors="coerce").dropna(axis=0, how="all")
    frame = frame.groupby(level=0).mean()
    expression = np.log2(frame.T.clip(lower=0) + 1.0)
    expression.index.name = "sample_id"
    expression = expression.dropna(axis=1, how="all").fillna(expression.median(axis=0))
    if max_genes is not None and expression.shape[1] > max_genes:
        selected = expression.var(axis=0).nlargest(max_genes).index
        expression = expression.loc[:, selected]
    scaled = StandardScaler().fit_transform(expression)
    processed = pd.DataFrame(scaled, index=expression.index, columns=expression.columns)
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "tcga_expression.csv"
    processed.to_csv(output)
    pd.Series(processed.columns, name="gene").to_csv(output_dir / "gene_names.csv", index=False)
    pd.Series(processed.index, name="sample_id").to_csv(output_dir / "sample_ids.csv", index=False)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess UCSC Xena TCGA TPM expression data.")
    parser.add_argument("--input", type=Path, default=ROOT / "data/raw/tcga_RSEM_gene_tpm.gz")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/processed")
    parser.add_argument("--max-genes", type=int, default=2000)
    args = parser.parse_args()
    path = prepare_expression(args.input, args.output_dir, max_genes=args.max_genes)
    print(f"Wrote patient-by-gene expression matrix to {path}")


if __name__ == "__main__":
    main()
