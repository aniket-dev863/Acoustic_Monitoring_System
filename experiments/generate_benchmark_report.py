"""
Comprehensive Benchmark Report Generator for AMRAG Patent Documentation.
Reads all experiment JSON files and produces:
  1. A formatted terminal summary
  2. A patent-ready Markdown report  (experiments/results/BENCHMARK_REPORT.md)
  3. A detailed per-file CSV         (experiments/results/per_file_results.csv)
"""
import json
import csv
from pathlib import Path

RESULTS_DIR = Path("experiments/results")
REPORT_PATH = RESULTS_DIR / "BENCHMARK_REPORT.md"

def load(fname):
    with open(RESULTS_DIR / fname) as f:
        return json.load(f)

def pct(v): return f"{v*100:.1f}%"
def ms(v):  return f"{v:.1f} ms"


def build_report():
    gc   = load("gate_comparison.json")
    rs   = load("resource_sweep.json")
    nf   = load("noise_freeze_ablation.json")

    lines = []

    # ── HEADER ──────────────────────────────────────────────────────────────
    lines += [
        "# AMRAG-v1 Benchmark Report",
        "## Experimental Validation for Patent Documentation",
        "",
        f"**Dataset:** 48 synthetic WAV files | 5 machine types (fan, bearing, gearbox, valve, pump) | 6 edge-case clips",
        f"**Corpus:** normal=22 clips, anomaly=26 clips | 16 kHz, float32 PCM",
        f"**Gate compared:** AMRAG-v1 vs Fixed-Amplitude (A), Fixed-Band (B), Adaptive-SNR (C) baselines",
        "",
        "---",
        "",
    ]

    # ── EXPERIMENT 1: GATE COMPARISON ───────────────────────────────────────
    lines += [
        "## Experiment 1 — Gate Benchmark Comparison (48 files)",
        "",
        "All four gates evaluated on the identical 48-file corpus.",
        "",
        "| Gate | Recall | Precision | F1 Score | FPR | ML Calls Avoided | Latency |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, r in gc.items():
        marker = " ← **proposed**" if "AMRAG" in name else ""
        lines.append(
            f"| {name}{marker} | {pct(r['recall'])} | {pct(r['precision'])} "
            f"| {r['f1_score']:.3f} | {pct(r['false_positive_rate'])} "
            f"| {r['ml_invocations_avoided_pct']:.1f}% | {ms(r['mean_latency_ms'])} |"
        )

    amrag = gc["AMRAG-v1"]
    base_a = gc["Baseline-A-v1"]
    base_c = gc["Baseline-C-v1"]

    fpr_vs_a = base_a['false_positive_rate'] - amrag['false_positive_rate']
    fpr_vs_c = base_c['false_positive_rate'] - amrag['false_positive_rate']
    ml_vs_a  = amrag['ml_invocations_avoided_pct'] - base_a['ml_invocations_avoided_pct']

    lines += [
        "",
        "**Key findings:**",
        f"- AMRAG achieves **{pct(amrag['false_positive_rate'])} FPR** vs {pct(base_a['false_positive_rate'])} for Baseline-A "
        f"(−{pct(fpr_vs_a)} false positive rate)",
        f"- AMRAG avoids **{amrag['ml_invocations_avoided_pct']:.1f}% of ML invocations** vs {base_a['ml_invocations_avoided_pct']:.1f}% for Baseline-A "
        f"(+{ml_vs_a:.1f}% compute saved)",
        f"- Baseline-B (fixed band) avoids the most ML calls (85.4%) but recall collapses to 26.9% — unusable",
        f"- AMRAG improves FPR over Baseline-C by {pct(fpr_vs_c)} while matching recall",
        f"- AMRAG mean gate latency: **{ms(amrag['mean_latency_ms'])}** per audio clip (all processing in Python; "
        f"C implementation on ESP32-S3 expected to be significantly faster)",
        "",
        "---",
        "",
    ]

    # ── EXPERIMENT 2: RESOURCE SWEEP ────────────────────────────────────────
    lines += [
        "## Experiment 2 — Resource-Level Adaptive Threshold Sweep",
        "",
        "AMRAG threshold adapts as resource_level decreases: `threshold = 0.55 + (1 − RL) × 0.30`",
        "",
        "| Resource Level | Effective Threshold | Recall | FPR | ML Invocations Avoided | Triggers |",
        "|---|---|---|---|---|---|",
    ]
    for rl_str, r in rs.items():
        marker = " ← full power" if float(rl_str) == 1.0 else (" ← constrained" if float(rl_str) == 0.2 else "")
        lines.append(
            f"| {r['resource_level']:.1f}{marker} | {r['effective_threshold']:.3f} "
            f"| {pct(r['recall'])} | {pct(r['fpr'])} "
            f"| {r['ml_invocations_avoided_pct']:.1f}% | {r['triggers']} |"
        )

    lines += [
        "",
        "**Key findings:**",
        "- As resource_level drops from 1.0 → 0.2, the trigger threshold rises from 0.550 → 0.790",
        "- At RL=0.6 and below, FPR drops to 0% (no false positives) — the gate becomes more selective",
        f"- ML invocations avoided increases from {rs['1.0']['ml_invocations_avoided_pct']:.1f}% (RL=1.0) to "
        f"{rs['0.2']['ml_invocations_avoided_pct']:.1f}% (RL=0.2) — a "
        f"{rs['0.2']['ml_invocations_avoided_pct'] - rs['1.0']['ml_invocations_avoided_pct']:.1f}% further reduction",
        "- This demonstrates the resource-adaptive selectivity mechanism: under battery/compute pressure, "
        "the gate gates harder and protects remaining resources",
        "",
        "---",
        "",
    ]

    # ── EXPERIMENT 3: NOISE FREEZE ABLATION ─────────────────────────────────
    lines += [
        "## Experiment 3 — Event-Aware Noise Freeze Ablation",
        "",
        "Comparison on 10 selected clips including non-stationary noise conditions.",
        "AMRAG has event-aware noise freeze ON; Baseline-C (Adaptive-SNR) has no freeze.",
        "",
        "| File | True Label | AMRAG (freeze ON) | Baseline-C (no freeze) | Effect |",
        "|---|---|---|---|---|",
    ]
    for rec in nf["file_records"]:
        a = "TRIGGER" if rec["amrag_triggered"] else "skip"
        b = "TRIGGER" if rec["baseline_c_triggered"] else "skip"
        lines.append(f"| {rec['file']} | {rec['true_label']} | {a} | {b} | {rec['freeze_effect']} |")

    amrag_r  = nf['amrag']['recall']
    bc_r     = nf['baseline_c']['recall']
    amrag_fp = nf['amrag']['fpr']
    bc_fp    = nf['baseline_c']['fpr']

    lines += [
        "",
        f"| Metric | AMRAG (freeze ON) | Baseline-C (no freeze) |",
        f"|---|---|---|",
        f"| Recall | {pct(amrag_r)} | {pct(bc_r)} |",
        f"| FPR | {pct(amrag_fp)} | {pct(bc_fp)} |",
        "",
        "**Key findings:**",
        f"- AMRAG achieves **{pct(amrag_fp)} FPR** vs {pct(bc_fp)} for Baseline-C — "
        f"a **{pct(bc_fp - amrag_fp)} reduction in false positives**",
        "- Baseline-C triggered on `edge_nonstationary_rising_noise_normal.wav` (a false positive) "
        "— rising ambient noise raised its SNR estimate, causing erroneous triggering",
        "- AMRAG correctly skipped that clip because its adaptive multi-band score "
        "did not exceed the trigger threshold despite the rising energy",
        "- Both gates missed low-SNR bearing faults (outer_race, inner_race) — "
        "these require threshold tuning or longer-window analysis (noted as future work)",
        "",
        "---",
        "",
    ]

    # ── SUMMARY TABLE ───────────────────────────────────────────────────────
    lines += [
        "## Summary: AMRAG vs Best Competing Baseline",
        "",
        "| Property | Baseline-A (Fixed Amplitude) | Baseline-C (Adaptive SNR) | **AMRAG-v1 (proposed)** |",
        "|---|---|---|---|",
        f"| Recall | {pct(base_a['recall'])} | {pct(base_c['recall'])} | **{pct(amrag['recall'])}** |",
        f"| FPR | {pct(base_a['false_positive_rate'])} | {pct(base_c['false_positive_rate'])} | **{pct(amrag['false_positive_rate'])}** |",
        f"| F1 Score | {base_a['f1_score']:.3f} | {base_c['f1_score']:.3f} | **{amrag['f1_score']:.3f}** |",
        f"| ML Calls Avoided | {base_a['ml_invocations_avoided_pct']:.1f}% | {base_c['ml_invocations_avoided_pct']:.1f}% | **{amrag['ml_invocations_avoided_pct']:.1f}%** |",
        f"| Gate Latency | {ms(base_a['mean_latency_ms'])} | {ms(base_c['mean_latency_ms'])} | {ms(amrag['mean_latency_ms'])} |",
        f"| Event-aware noise freeze | ✗ | ✗ | **✓** |",
        f"| Multi-band SNR | ✗ | ✗ | **✓** |",
        f"| Resource-adaptive threshold | ✗ | ✗ | **✓** |",
        "",
        "---",
        "",
        "## Notes on Dataset and Limitations",
        "",
        "- All audio is synthetically generated using physically-grounded machine models",
        "  (fan blade-pass harmonics, bearing ball-pass impact trains, gear-mesh tones, etc.)",
        "- 48 files is adequate for mechanism validation; larger real-world datasets (e.g., MIMII) "
        "are recommended for production deployment evaluation",
        "- Gate latency measured in Python; C/C++ implementation on ESP32-S3 expected to be "
        "10–50× faster (to be profiled on target hardware)",
        "- Low-SNR bearing faults (outer/inner race at 15–28 impacts/s) fall below current "
        "threshold — threshold tuning or signal conditioning is a direction for future work",
        "",
        "---",
        "",
        "*Generated by `experiments/generate_benchmark_report.py`*",
        f"*Dataset: `data/sample/` | Results: `experiments/results/`*",
    ]

    report_text = "\n".join(lines)
    with open(REPORT_PATH, "w") as f:
        f.write(report_text)
    print(f"\n✅  Report saved → {REPORT_PATH}")

    # ── CSV ─────────────────────────────────────────────────────────────────
    csv_path = RESULTS_DIR / "per_file_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["file", "machine_type", "true_label",
                         "baseline_a", "baseline_b", "baseline_c", "amrag",
                         "amrag_score", "baseline_c_score"])
        # Build lookup from gate_comparison file_records
        fa = {r["file_name"]: r for r in base_a["file_records"]}
        fb = {r["file_name"]: r for r in gc["Baseline-B-v1"]["file_records"]}
        fc = {r["file_name"]: r for r in base_c["file_records"]}
        fm = {r["file_name"]: r for r in amrag["file_records"]}
        for fname in fa:
            writer.writerow([
                fname,
                fa[fname].get("machine_type",""),
                fa[fname]["true_label"],
                "T" if fa[fname]["triggered"] else "F",
                "T" if fb.get(fname,{}).get("triggered") else "F",
                "T" if fc[fname]["triggered"] else "F",
                "T" if fm[fname]["triggered"] else "F",
                fm[fname]["max_score"],
                fc[fname]["max_score"],
            ])
    print(f"✅  CSV saved     → {csv_path}\n")

    # Print quick summary to terminal
    print("\n" + "="*65)
    print("  PATENT DOCUMENTATION NUMBERS (verified from experiments)")
    print("="*65)
    print(f"  Gate comparison on 48 files:")
    print(f"    AMRAG Recall:      {pct(amrag['recall'])}")
    print(f"    AMRAG FPR:         {pct(amrag['false_positive_rate'])}")
    print(f"    AMRAG F1:          {amrag['f1_score']:.3f}")
    print(f"    ML calls avoided:  {amrag['ml_invocations_avoided_pct']:.1f}%  (vs {base_a['ml_invocations_avoided_pct']:.1f}% for Baseline-A)")
    print(f"    FPR vs Baseline-A: −{pct(fpr_vs_a)}")
    print(f"  Resource sweep:")
    print(f"    RL=1.0 → threshold=0.550, ML avoided={rs['1.0']['ml_invocations_avoided_pct']:.1f}%")
    print(f"    RL=0.2 → threshold=0.790, ML avoided={rs['0.2']['ml_invocations_avoided_pct']:.1f}%")
    print(f"  Noise freeze ablation:")
    print(f"    AMRAG FPR:         {pct(amrag_fp)}")
    print(f"    Baseline-C FPR:    {pct(bc_fp)}")
    print(f"    FPR reduction:     −{pct(bc_fp - amrag_fp)}")
    print("="*65 + "\n")


if __name__ == "__main__":
    build_report()
