"""
Noise-Freeze Ablation Experiment.
Demonstrates that event-aware noise floor freezing prevents missed detections
in sustained anomaly + non-stationary noise conditions.

Compares:
  A) AMRAG-v1 (with noise freeze — the proposed system)
  B) Baseline-C (Adaptive SNR gate — WITHOUT event-aware freeze)

Focuses on files specifically designed to test this:
  - edge_nonstationary_noise_with_anomaly.wav
  - edge_nonstationary_rising_noise_normal.wav
  - bearing_anomaly_outer_race.wav  (sustained periodic fault)
  - fan_anomaly_blade_unbalance.wav (sustained energy surge)
  - noise_shift_anomaly.wav        (original test file)
"""
import json
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dsp.adaptive_gate import AdaptiveMultiBandResourceGate
from dsp.baseline_gates import BaselineAdaptiveSNRGate

DATA_DIR = Path("data/sample")
OUT_FILE = "experiments/results/noise_freeze_ablation.json"

TARGET_FILES = [
    ("edge_nonstationary_noise_with_anomaly.wav", "anomaly"),
    ("edge_nonstationary_rising_noise_normal.wav", "normal"),
    ("bearing_anomaly_outer_race.wav",             "anomaly"),
    ("bearing_anomaly_inner_race.wav",             "anomaly"),
    ("fan_anomaly_blade_unbalance.wav",            "anomaly"),
    ("gearbox_anomaly_wear.wav",                   "anomaly"),
    ("pump_anomaly_shaft_imbalance.wav",           "anomaly"),
    ("edge_low_snr_anomaly.wav",                   "anomaly"),
    ("fan_normal_0.wav",                           "normal"),
    ("gearbox_normal_0.wav",                       "normal"),
]

def run_ablation():
    amrag  = AdaptiveMultiBandResourceGate(resource_level=1.0)
    base_c = BaselineAdaptiveSNRGate(snr_threshold_db=6.0, resource_level=1.0)

    print("\n" + "="*85)
    print("  NOISE-FREEZE ABLATION: AMRAG (freeze=ON) vs Baseline-C (freeze=OFF)")
    print("="*85)
    print(f"{'File':<48} | {'True':>6} | {'AMRAG':>7} | {'Base-C':>7} | {'Freeze helps?':>14}")
    print("-"*85)

    records = []
    amrag_tp = amrag_fp = amrag_tn = amrag_fn = 0
    bc_tp    = bc_fp    = bc_tn    = bc_fn    = 0

    for fname, true_label in TARGET_FILES:
        fpath = DATA_DIR / fname
        if not fpath.exists():
            print(f"  SKIP (not found): {fname}")
            continue

        r_amrag = amrag.process_audio(fpath)
        r_bc    = base_c.process_audio(fpath)

        ta = r_amrag.overall_triggered
        tb = r_bc.overall_triggered
        is_anomaly = (true_label == "anomaly")

        # Update confusion matrices
        if is_anomaly:
            if ta: amrag_tp += 1
            else:  amrag_fn += 1
            if tb: bc_tp += 1
            else:  bc_fn += 1
        else:
            if ta: amrag_fp += 1
            else:  amrag_tn += 1
            if tb: bc_fp += 1
            else:  bc_tn += 1

        # Did freeze help? = AMRAG triggered correctly where Baseline-C didn't
        if is_anomaly and ta and not tb:
            freeze_effect = "YES ✓ saved"
        elif is_anomaly and not ta and tb:
            freeze_effect = "NO  ✗ missed"
        elif is_anomaly and not ta and not tb:
            freeze_effect = "both missed"
        elif not is_anomaly and ta and not tb:
            freeze_effect = "AMRAG FP"
        else:
            freeze_effect = "same"

        print(f"{fname:<48} | {true_label:>6} | {'TRIG' if ta else 'skip':>7} | {'TRIG' if tb else 'skip':>7} | {freeze_effect:>14}")

        records.append({
            "file": fname, "true_label": true_label,
            "amrag_triggered": ta, "baseline_c_triggered": tb,
            "amrag_score": round(r_amrag.max_trigger_score, 4),
            "baseline_c_score": round(r_bc.max_trigger_score, 4),
            "freeze_effect": freeze_effect,
        })

    print("="*85)

    # Summary
    def safe_div(a, b): return round(a / b, 4) if b > 0 else 0.0
    amrag_recall = safe_div(amrag_tp, amrag_tp + amrag_fn)
    amrag_fpr    = safe_div(amrag_fp, amrag_fp + amrag_tn)
    bc_recall    = safe_div(bc_tp,    bc_tp    + bc_fn)
    bc_fpr       = safe_div(bc_fp,    bc_fp    + bc_tn)

    print(f"\n  AMRAG  (freeze ON) : Recall={amrag_recall*100:.1f}%  FPR={amrag_fpr*100:.1f}%")
    print(f"  Baseline-C (no freeze): Recall={bc_recall*100:.1f}%  FPR={bc_fpr*100:.1f}%")
    print(f"  Recall improvement from freeze: +{(amrag_recall - bc_recall)*100:.1f}%\n")

    output = {
        "experiment": "noise_freeze_ablation",
        "description": "AMRAG (event-aware noise freeze ON) vs Baseline-C (no event freeze)",
        "amrag": {"recall": amrag_recall, "fpr": amrag_fpr,
                  "confusion": {"tp":amrag_tp,"fp":amrag_fp,"tn":amrag_tn,"fn":amrag_fn}},
        "baseline_c": {"recall": bc_recall, "fpr": bc_fpr,
                       "confusion": {"tp":bc_tp,"fp":bc_fp,"tn":bc_tn,"fn":bc_fn}},
        "recall_improvement": round(amrag_recall - bc_recall, 4),
        "fpr_improvement": round(bc_fpr - amrag_fpr, 4),
        "file_records": records,
    }

    with open(OUT_FILE, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved → {OUT_FILE}\n")
    return output

if __name__ == "__main__":
    run_ablation()
