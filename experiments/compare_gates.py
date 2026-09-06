"""
Experimental Gate Comparison Engine.

Executes all 4 acoustic gating systems across an identical corpus of machine recordings:
1. Baseline A: Fixed Amplitude
2. Baseline B: Fixed Frequency Band
3. Baseline C: Adaptive SNR
4. Proposed Method: Adaptive Multi-Band Resource-Aware Acoustic Gate (AMRAG)

Evaluates:
- Precision, Recall, F1 Score
- False Positive Rate (FPR)
- False Negative Rate (FNR)
- Trigger Rate (%)
- Anomaly Recall (%)
- Audio Rejected (%)
- ML Invocations Avoided (%)
"""

import json
from pathlib import Path
import time
from typing import Any, Dict, List
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dsp.adaptive_gate import AdaptiveMultiBandResourceGate
from dsp.baseline_gates import (
    BaselineAdaptiveSNRGate,
    BaselineFixedAmplitudeGate,
    BaselineFixedBandGate,
)
from ml.training.dataset import build_manifest_from_directory


def evaluate_gate_on_manifest(gate, manifest: list) -> Dict[str, Any]:
    """Runs a single gate over all audio clips and collects benchmark metrics."""
    tp, fp, tn, fn = 0, 0, 0, 0
    total_audio_sec = 0.0
    rejected_audio_sec = 0.0
    gate_times = []
    file_records = []

    for sample in manifest:
        is_true_anomaly = (sample.label == "anomaly")
        total_audio_sec += sample.duration_sec

        t_start = time.perf_counter()
        report = gate.process_audio(sample.file_path)
        gate_times.append((time.perf_counter() - t_start) * 1000.0)

        triggered = report.overall_triggered

        if triggered:
            if is_true_anomaly:
                tp += 1
            else:
                fp += 1
        else:
            rejected_audio_sec += sample.duration_sec
            if is_true_anomaly:
                fn += 1
            else:
                tn += 1

        file_records.append({
            "file_name": sample.file_name,
            "machine_type": sample.machine_type,
            "true_label": sample.label,
            "triggered": triggered,
            "trigger_rate": round(report.trigger_rate, 4),
            "max_score": round(report.max_trigger_score, 4),
            "events_count": len(report.events),
        })

    total_samples = len(manifest)
    total_positives = tp + fn
    total_negatives = tn + fp

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    fpr = fp / total_negatives if total_negatives > 0 else 0.0
    fnr = fn / total_positives if total_positives > 0 else 0.0

    trigger_rate = (tp + fp) / total_samples if total_samples > 0 else 0.0
    audio_rejected_pct = (rejected_audio_sec / total_audio_sec * 100.0) if total_audio_sec > 0 else 0.0
    ml_avoided_pct = (tn + fn) / total_samples * 100.0 if total_samples > 0 else 0.0

    return {
        "gate_type": gate.gate_type,
        "gate_version": gate.gate_version,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "false_positive_rate": round(fpr, 4),
        "false_negative_rate": round(fnr, 4),
        "trigger_rate_pct": round(trigger_rate * 100.0, 2),
        "anomaly_recall_pct": round(recall * 100.0, 2),
        "audio_rejected_pct": round(audio_rejected_pct, 2),
        "ml_invocations_avoided_pct": round(ml_avoided_pct, 2),
        "mean_latency_ms": round(float(np.mean(gate_times)), 2),
        "confusion_matrix": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "file_records": file_records,
    }


def run_comparisons(data_dir: str = "data/sample", results_dir: str = "experiments/results") -> Dict[str, Any]:
    """Runs all 4 gates on identical audio files and plots comparative metrics."""
    out_dir = Path(results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = Path("experiments/plots")
    plot_dir.mkdir(parents=True, exist_ok=True)

    manifest = build_manifest_from_directory(data_dir)
    print(f"\n=========================================================")
    print(f"  RUNNING ACOUSTIC GATE BENCHMARK COMPARISON ({len(manifest)} files)")
    print(f"=========================================================\n")

    gates = [
        BaselineFixedAmplitudeGate(threshold=0.05),
        BaselineFixedBandGate(target_band_name="mid", threshold=0.02),
        BaselineAdaptiveSNRGate(snr_threshold_db=6.0),
        AdaptiveMultiBandResourceGate(resource_level=1.0),
    ]

    all_results = {}
    for gate in gates:
        print(f"Evaluating {gate.gate_version} ({gate.gate_type})...")
        res = evaluate_gate_on_manifest(gate, manifest)
        all_results[gate.gate_version] = res

    # Save JSON results
    json_path = out_dir / "gate_comparison.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved benchmark metrics to {json_path}")

    # Print summary table
    print("\n" + "=" * 92)
    print(f"{'Gate Model':<22} | {'Recall':<8} | {'Precision':<10} | {'F1':<6} | {'FPR':<6} | {'ML Saved %':<11} | {'Latency':<8}")
    print("-" * 92)
    for name, r in all_results.items():
        print(f"{name:<22} | {r['recall']*100:<7.1f}% | {r['precision']*100:<9.1f}% | {r['f1_score']:<6.3f} | {r['false_positive_rate']*100:<5.1f}% | {r['ml_invocations_avoided_pct']:<10.1f}% | {r['mean_latency_ms']:<6.2f}ms")
    print("=" * 92 + "\n")

    # Generate visual comparison plot
    plot_comparison_charts(all_results, plot_dir / "gate_comparison.png")

    return all_results


def plot_comparison_charts(results: Dict[str, Any], output_path: Path) -> None:
    """Renders multi-metric comparison chart."""
    labels = list(results.keys())
    recalls = [r["recall"] * 100.0 for r in results.values()]
    fprs = [r["false_positive_rate"] * 100.0 for r in results.values()]
    ml_saved = [r["ml_invocations_avoided_pct"] for r in results.values()]
    f1s = [r["f1_score"] * 100.0 for r in results.values()]

    x = np.arange(len(labels))
    width = 0.20

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - 1.5 * width, recalls, width, label="Anomaly Recall %", color="#2ecc71")
    ax.bar(x - 0.5 * width, f1s, width, label="F1 Score %", color="#3498db")
    ax.bar(x + 0.5 * width, fprs, width, label="False Positive Rate %", color="#e74c3c")
    ax.bar(x + 1.5 * width, ml_saved, width, label="ML Calls Avoided %", color="#f39c12")

    ax.set_ylabel("Percentage (%)", fontsize=11)
    ax.set_title("Acoustic Gate Benchmark Comparison (Identical Audio Corpus)", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim([0, 115])
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(str(output_path), dpi=140)
    plt.close(fig)
    print(f"Saved comparison plot to {output_path}")


if __name__ == "__main__":
    run_comparisons()
