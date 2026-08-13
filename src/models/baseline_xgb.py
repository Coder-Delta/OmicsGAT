"""XGBoost baseline using the same persisted sample split as the GNNs."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from xgboost import XGBClassifier
from src.data.dataset import _read_labels, make_or_load_split

ROOT = Path(__file__).resolve().parents[2]

def metrics(y_true, y_pred, probabilities, n_classes):
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    out = {"accuracy": float(accuracy_score(y_true, y_pred)), "precision": float(precision), "recall": float(recall), "f1": float(f1)}
    try: out["roc_auc"] = float(roc_auc_score(y_true, probabilities[:, 1] if n_classes == 2 else probabilities, multi_class="ovr", average="weighted"))
    except ValueError: out["roc_auc"] = None
    return out

def main():
    p = argparse.ArgumentParser(); p.add_argument("--expression", type=Path, default=ROOT / "data/processed/tcga_expression.csv"); p.add_argument("--labels", type=Path, default=ROOT / "data/processed/tcga_labels.csv"); p.add_argument("--split", type=Path, default=ROOT / "data/processed/train_test_split.json"); p.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints/xgboost.json"); p.add_argument("--results", type=Path, default=ROOT / "outputs/results/xgboost_metrics.json"); p.add_argument("--test-size", type=float, default=.2); p.add_argument("--seed", type=int, default=42); args = p.parse_args()
    x = pd.read_csv(args.expression, index_col="sample_id"); x.index = x.index.astype(str)
    labels = _read_labels(args.labels); labels["sample_id"] = labels["sample_id"].astype(str); labels = labels.set_index("sample_id"); joined = x.join(labels, how="inner")
    if joined.empty: raise ValueError("No clinical labels match the expression sample IDs.")
    classes = sorted(joined.label.astype(str).unique()); y = joined.label.astype(str).map({v:i for i,v in enumerate(classes)})
    split = make_or_load_split(joined.index.astype(str).tolist(), y.tolist(), args.split, args.test_size, args.seed)
    train_i, test_i = split["train"], split["test"]
    parameters = {"objective": "multi:softprob" if len(classes)>2 else "binary:logistic", "eval_metric": "mlogloss", "random_state": args.seed}
    if len(classes) > 2: parameters["num_class"] = len(classes)
    model = XGBClassifier(**parameters)
    model.fit(joined.iloc[train_i, :-1], y.iloc[train_i]); probabilities = model.predict_proba(joined.iloc[test_i, :-1]); predicted = probabilities.argmax(axis=1)
    result = metrics(y.iloc[test_i], predicted, probabilities, len(classes)) | {"class_names": classes, "n_train": len(train_i), "n_test": len(test_i)}
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True); model.save_model(args.checkpoint); args.results.parent.mkdir(parents=True, exist_ok=True); args.results.write_text(json.dumps(result, indent=2)); print(json.dumps(result, indent=2))
if __name__ == "__main__": main()
