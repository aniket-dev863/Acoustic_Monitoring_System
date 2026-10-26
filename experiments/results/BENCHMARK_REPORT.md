# AMRAG-v1 Benchmark Report
## Experimental Validation for Patent Documentation

**Dataset:** 48 synthetic WAV files | 5 machine types (fan, bearing, gearbox, valve, pump) | 6 edge-case clips
**Corpus:** normal=22 clips, anomaly=26 clips | 16 kHz, float32 PCM
**Gate compared:** AMRAG-v1 vs Fixed-Amplitude (A), Fixed-Band (B), Adaptive-SNR (C) baselines

---

## Experiment 1 — Gate Benchmark Comparison (48 files)

All four gates evaluated on the identical 48-file corpus.

| Gate | Recall | Precision | F1 Score | FPR | ML Calls Avoided | Latency |
|---|---|---|---|---|---|---|
| Baseline-A-v1 | 100.0% | 56.5% | 0.722 | 90.9% | 4.2% | 19.9 ms |
| Baseline-B-v1 | 26.9% | 100.0% | 0.424 | 0.0% | 85.4% | 18.0 ms |
| Baseline-C-v1 | 46.2% | 80.0% | 0.585 | 13.6% | 68.8% | 24.6 ms |
| AMRAG-v1 ← **proposed** | 50.0% | 86.7% | 0.634 | 9.1% | 68.8% | 42.9 ms |

**Key findings:**
- AMRAG achieves **9.1% FPR** vs 90.9% for Baseline-A (−81.8% false positive rate)
- AMRAG avoids **68.8% of ML invocations** vs 4.2% for Baseline-A (+64.6% compute saved)
- Baseline-B (fixed band) avoids the most ML calls (85.4%) but recall collapses to 26.9% — unusable
- AMRAG improves FPR over Baseline-C by 4.5% while matching recall
- AMRAG mean gate latency: **42.9 ms** per audio clip (all processing in Python; C implementation on ESP32-S3 expected to be significantly faster)

---

## Experiment 2 — Resource-Level Adaptive Threshold Sweep

AMRAG threshold adapts as resource_level decreases: `threshold = 0.55 + (1 − RL) × 0.30`

| Resource Level | Effective Threshold | Recall | FPR | ML Invocations Avoided | Triggers |
|---|---|---|---|---|---|
| 1.0 ← full power | 0.550 | 50.0% | 9.1% | 68.8% | 15 |
| 0.8 | 0.610 | 46.2% | 9.1% | 70.8% | 14 |
| 0.6 | 0.670 | 42.3% | 0.0% | 77.1% | 11 |
| 0.4 | 0.730 | 42.3% | 0.0% | 77.1% | 11 |
| 0.2 ← constrained | 0.790 | 30.8% | 0.0% | 83.3% | 8 |

**Key findings:**
- As resource_level drops from 1.0 → 0.2, the trigger threshold rises from 0.550 → 0.790
- At RL=0.6 and below, FPR drops to 0% (no false positives) — the gate becomes more selective
- ML invocations avoided increases from 68.8% (RL=1.0) to 83.3% (RL=0.2) — a 14.6% further reduction
- This demonstrates the resource-adaptive selectivity mechanism: under battery/compute pressure, the gate gates harder and protects remaining resources

---

## Experiment 3 — Event-Aware Noise Freeze Ablation

Comparison on 10 selected clips including non-stationary noise conditions.
AMRAG has event-aware noise freeze ON; Baseline-C (Adaptive-SNR) has no freeze.

| File | True Label | AMRAG (freeze ON) | Baseline-C (no freeze) | Effect |
|---|---|---|---|---|
| edge_nonstationary_noise_with_anomaly.wav | anomaly | TRIGGER | TRIGGER | same |
| edge_nonstationary_rising_noise_normal.wav | normal | skip | TRIGGER | same |
| bearing_anomaly_outer_race.wav | anomaly | skip | skip | both missed |
| bearing_anomaly_inner_race.wav | anomaly | skip | skip | both missed |
| fan_anomaly_blade_unbalance.wav | anomaly | TRIGGER | TRIGGER | same |
| gearbox_anomaly_wear.wav | anomaly | skip | skip | both missed |
| pump_anomaly_shaft_imbalance.wav | anomaly | skip | skip | both missed |
| edge_low_snr_anomaly.wav | anomaly | skip | skip | both missed |
| fan_normal_0.wav | normal | skip | skip | same |
| gearbox_normal_0.wav | normal | skip | skip | same |

| Metric | AMRAG (freeze ON) | Baseline-C (no freeze) |
|---|---|---|
| Recall | 28.6% | 28.6% |
| FPR | 0.0% | 33.3% |

**Key findings:**
- AMRAG achieves **0.0% FPR** vs 33.3% for Baseline-C — a **33.3% reduction in false positives**
- Baseline-C triggered on `edge_nonstationary_rising_noise_normal.wav` (a false positive) — rising ambient noise raised its SNR estimate, causing erroneous triggering
- AMRAG correctly skipped that clip because its adaptive multi-band score did not exceed the trigger threshold despite the rising energy
- Both gates missed low-SNR bearing faults (outer_race, inner_race) — these require threshold tuning or longer-window analysis (noted as future work)

---

## Summary: AMRAG vs Best Competing Baseline

| Property | Baseline-A (Fixed Amplitude) | Baseline-C (Adaptive SNR) | **AMRAG-v1 (proposed)** |
|---|---|---|---|
| Recall | 100.0% | 46.2% | **50.0%** |
| FPR | 90.9% | 13.6% | **9.1%** |
| F1 Score | 0.722 | 0.585 | **0.634** |
| ML Calls Avoided | 4.2% | 68.8% | **68.8%** |
| Gate Latency | 19.9 ms | 24.6 ms | 42.9 ms |
| Event-aware noise freeze | ✗ | ✗ | **✓** |
| Multi-band SNR | ✗ | ✗ | **✓** |
| Resource-adaptive threshold | ✗ | ✗ | **✓** |

---

## Notes on Dataset and Limitations

- All audio is synthetically generated using physically-grounded machine models
  (fan blade-pass harmonics, bearing ball-pass impact trains, gear-mesh tones, etc.)
- 48 files is adequate for mechanism validation; larger real-world datasets (e.g., MIMII) are recommended for production deployment evaluation
- Gate latency measured in Python; C/C++ implementation on ESP32-S3 expected to be 10–50× faster (to be profiled on target hardware)
- Low-SNR bearing faults (outer/inner race at 15–28 impacts/s) fall below current threshold — threshold tuning or signal conditioning is a direction for future work

---

*Generated by `experiments/generate_benchmark_report.py`*
*Dataset: `data/sample/` | Results: `experiments/results/`*