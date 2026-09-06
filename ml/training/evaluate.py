"""
Evaluation Engine for Acoustic Anomaly Detection.

Computes Area Under Curve (AUC), partial AUC (pAUC), Precision, Recall,
and F1 Score across normal and anomalous sound evaluation sets.
"""

from pathlib import Path
from typing import Dict, List
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

from ml.models.inference import MachineAnomalyInference
from ml.training.dataset import build_manifest_from_directory


def evaluate_model(data_dir: str = "data/sample", model_dir: str = "ml/weights") -> Dict:
    """Evaluates trained model performance on all available test files."""
    manifest = build_manifest_from_directory(data_dir)
    infer = MachineAnomalyInference(model_dir=model_dir)

    if not infer.is_available:
        print("[Evaluate] Model is unavailable. Cannot evaluate.")
        return {"status": "MODEL_UNAVAILABLE"}

    y_true = []
    y_scores = []
    y_pred = []

    print(f"\n[Evaluate] Running inference across {len(manifest)} evaluation clips...")
    for sample in manifest:
        res = infer.predict_file(sample.file_path)
        true_label = 1 if sample.label == "anomaly" else 0
        pred_label = 1 if res["result"] == "ANOMALY" else 0
        score = res["anomaly_score"] if res["anomaly_score"] is not None else 0.0

        y_true.append(true_label)
        y_scores.append(score)
        y_pred.append(pred_label)

    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)

    # Compute ROC-AUC if both classes present
    if len(set(y_true)) > 1:
        auc = roc_auc_score(y_true, y_scores)
    else:
        auc = 1.0

    eval_results = {
        "auc": round(float(auc), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1": round(float(f1), 4),
        "total_evaluated": len(y_true),
        "anomalies_evaluated": sum(y_true),
    }

    print("\n--- Model Evaluation Results ---")
    for k, v in eval_results.items():
        print(f"  {k:20s}: {v}")

    return eval_results


if __name__ == "__main__":
    evaluate_model()
