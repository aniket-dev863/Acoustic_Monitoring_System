"""
Six-Stage Ablation Study for Adaptive Acoustic Gate.

Ablation Stages:
Stage A: Energy only (fixed amplitude threshold)
Stage B: Energy + Adaptive SNR (full-band adaptive background noise model)
Stage C: Energy + SNR + Multi-band (independent sub-band SNR analysis)
Stage D: Stage C + Spectral Features (BER, flux, entropy tonal scoring)
Stage E: Stage D + Temporal Persistence (confirmation frames + hysteresis release)
Stage F: Stage E + Resource Awareness (dynamic threshold/confirmation scaling)
"""

import json
from pathlib import Path
from typing import Any, Dict, List
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dsp.adaptive_gate import AdaptiveMultiBandResourceGate
from dsp.baseline_gates import BaselineFixedAmplitudeGate
from experiments.compare_gates import evaluate_gate_on_manifest
from ml.training.dataset import build_manifest_from_directory


def build_ablation_gates() -> List[Dict[str, Any]]:
    """Builds the 6 incremental gate configurations."""
    return [
        {
            "stage": "Stage A",
            "name": "Energy Only",
            "description": "Baseline fixed amplitude energy threshold",
            "gate": BaselineFixedAmplitudeGate(threshold=0.05),
        },
        {
            "stage": "Stage B",
            "name": "+ Adaptive SNR",
            "description": "Adaptive full-band background noise tracking",
            "gate": AdaptiveMultiBandResourceGate(
                enable_multi_band=False,
                enable_spectral_features=False,
                enable_temporal_persistence=False,
                enable_resource_adaptation=False,
                enable_noise_adaptation=True,
            ),
        },
        {
            "stage": "Stage C",
            "name": "+ Multi-band",
            "description": "Independent sub-band spectral decomposition",
            "gate": AdaptiveMultiBandResourceGate(
                enable_multi_band=True,
                enable_spectral_features=False,
                enable_temporal_persistence=False,
                enable_resource_adaptation=False,
                enable_noise_adaptation=True,
            ),
        },
        {
            "stage": "Stage D",
            "name": "+ Spectral Features",
            "description": "Band energy ratio, flux, and tonal entropy",
            "gate": AdaptiveMultiBandResourceGate(
                enable_multi_band=True,
                enable_spectral_features=True,
                enable_temporal_persistence=False,
                enable_resource_adaptation=False,
                enable_noise_adaptation=True,
            ),
        },
        {
            "stage": "Stage E",
            "name": "+ Temporal Persistence",
            "description": "Hysteresis state machine and consecutive frame confirmation",
            "gate": AdaptiveMultiBandResourceGate(
                enable_multi_band=True,
                enable_spectral_features=True,
                enable_temporal_persistence=True,
                enable_resource_adaptation=False,
                enable_noise_adaptation=True,
            ),
        },
        {
            "stage": "Stage F",
            "name": "+ Resource Awareness",
            "description": "Full AMRAG with dynamic battery/compute budget scaling",
            "gate": AdaptiveMultiBandResourceGate(
                resource_level=0.5,  # evaluate constrained edge scenario
                enable_multi_band=True,
                enable_spectral_features=True,
                enable_temporal_persistence=True,
                enable_resource_adaptation=True,
                enable_noise_adaptation=True,
            ),
        },
    ]


def run_ablation_study(
    data_dir: str = "data/sample",
    results_dir: str = "experiments/results",
) -> Dict[str, Any]:
    out_dir = Path(results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = Path("experiments/plots")
    plot_dir.mkdir(parents=True, exist_ok=True)

    manifest = build_manifest_from_directory(data_dir)
    configs = build_ablation_gates()

    print("\n" + "=" * 75)
    print(f"  RUNNING 6-STAGE ABLATION STUDY ({len(manifest)} audio files)")
    print("=" * 75)

    ablation_results = []
    for item in configs:
        stage = item["stage"]
        name = item["name"]
        print(f"Testing {stage}: {name}...")
        metrics = evaluate_gate_on_manifest(item["gate"], manifest)
        metrics["stage"] = stage
        metrics["stage_name"] = name
        metrics["stage_description"] = item["description"]
        ablation_results.append(metrics)

    # Save results
    out_file = out_dir / "ablation_study.json"
    with open(out_file, "w") as f:
        json.dump(ablation_results, f, indent=2)

    # Summary table
    print("\n" + "=" * 96)
    print(f"{'Stage':<10} | {'Configuration':<26} | {'Recall':<8} | {'FPR':<7} | {'ML Saved %':<11} | {'F1':<6}")
    print("-" * 96)
    for res in ablation_results:
        print(f"{res['stage']:<10} | {res['stage_name']:<26} | {res['recall']*100:<7.1f}% | {res['false_positive_rate']*100:<6.1f}% | {res['ml_invocations_avoided_pct']:<10.1f}% | {res['f1_score']:<6.3f}")
    print("=" * 96 + "\n")

    plot_ablation_trends(ablation_results, plot_dir / "ablation_study.png")
    return {"stages": ablation_results}


def plot_ablation_trends(results: List[Dict[str, Any]], output_path: Path) -> None:
    """Renders progression plot across stages A through F."""
    stages = [f"{r['stage']}\n{r['stage_name']}" for r in results]
    recall = [r["recall"] * 100.0 for r in results]
    fpr = [r["false_positive_rate"] * 100.0 for r in results]
    ml_saved = [r["ml_invocations_avoided_pct"] for r in results]
    f1 = [r["f1_score"] * 100.0 for r in results]

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(stages, recall, marker="o", lw=2, color="#2ecc71", label="Anomaly Recall (%)")
    ax.plot(stages, f1, marker="s", lw=2, color="#3498db", label="F1 Score (%)")
    ax.plot(stages, ml_saved, marker="^", lw=2, color="#f39c12", label="Downstream Compute Saved (%)")
    ax.plot(stages, fpr, marker="d", lw=2, color="#e74c3c", linestyle="--", label="False Positive Rate (%)")

    ax.set_title("Ablation Study: Progressive Addition of DSP & Adaptive Components", fontsize=13, fontweight="bold")
    ax.set_ylabel("Metric Value (%)", fontsize=11)
    ax.set_ylim([-5, 110])
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="best", fontsize=10)

    plt.tight_layout()
    plt.savefig(str(output_path), dpi=140)
    plt.close(fig)
    print(f"Saved ablation plot to {output_path}")


if __name__ == "__main__":
    run_ablation_study()
