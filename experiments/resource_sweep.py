"""
Resource-Level Sweep Benchmark.
Tests AMRAG-v1 at 5 resource levels to demonstrate resource-adaptive
threshold selectivity — a key novelty claim.
"""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dsp.adaptive_gate import AdaptiveMultiBandResourceGate
from ml.training.dataset import build_manifest_from_directory

DATA_DIR = "data/sample"
OUT_FILE = "experiments/results/resource_sweep.json"

def run_sweep():
    manifest = build_manifest_from_directory(DATA_DIR)
    resource_levels = [1.0, 0.8, 0.6, 0.4, 0.2]
    results = {}

    print("\n" + "="*70)
    print("  AMRAG RESOURCE-LEVEL SWEEP  (48 files)")
    print("="*70)
    print(f"{'Resource':>10} | {'Threshold':>10} | {'Recall':>8} | {'FPR':>8} | {'ML Avoided%':>12} | {'Triggers':>8}")
    print("-"*70)

    for rl in resource_levels:
        gate = AdaptiveMultiBandResourceGate(resource_level=rl)
        # effective threshold = base + (1-rl)*0.30
        eff_thresh = round(gate.effective_trigger_threshold, 3)

        tp, fp, tn, fn = 0, 0, 0, 0
        for sample in manifest:
            is_anomaly = (sample.label == "anomaly")
            report = gate.process_audio(sample.file_path)
            triggered = report.overall_triggered
            if triggered:
                if is_anomaly: tp += 1
                else:          fp += 1
            else:
                if is_anomaly: fn += 1
                else:          tn += 1

        total = len(manifest)
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr    = fp / (tn + fp) if (tn + fp) > 0 else 0.0
        ml_avoided = (tn + fn) / total * 100.0

        results[str(rl)] = {
            "resource_level": rl,
            "effective_threshold": eff_thresh,
            "recall": round(recall, 4),
            "fpr": round(fpr, 4),
            "ml_invocations_avoided_pct": round(ml_avoided, 2),
            "triggers": tp + fp,
            "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        }

        print(f"{rl:>10.1f} | {eff_thresh:>10.3f} | {recall*100:>7.1f}% | {fpr*100:>7.1f}% | {ml_avoided:>11.1f}% | {tp+fp:>8d}")

    print("="*70)

    with open(OUT_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved → {OUT_FILE}\n")
    return results

if __name__ == "__main__":
    run_sweep()
