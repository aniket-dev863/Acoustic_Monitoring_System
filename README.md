# BUILD THE COMPLETE SOFTWARE-ONLY MACHINE ACOUSTIC MONITORING SYSTEM

Act as a senior **DSP researcher, acoustic anomaly-detection/ML engineer, Python developer, Node.js backend engineer, database engineer, and React developer**.

Build a complete, runnable **software-only machine acoustic condition-monitoring system** centered around an **Adaptive Multi-Band Resource-Aware Acoustic Gate**.

The system must demonstrate that a lightweight acoustic gate can reject normal/background machine audio and selectively forward acoustically significant events to more expensive anomaly processing.

---

# 1. CURRENT SCOPE

**Hardware is NOT available.**

Do NOT implement:

* ESP32 firmware
* INMP441/I2S drivers
* GPIO
* Wi-Fi firmware
* physical sensors

Instead, use WAV files as a software simulation of future edge-node audio.

Future:

```text
INMP441 → ESP32 I2S → Edge Gate
```

Current:

```text
WAV → Software Edge Gate
```

Keep the interfaces compatible so the future ESP32 implementation can replace the WAV source without redesigning the downstream pipeline.

---

# 2. APPLICATION

Application: **Machine Fault / Anomalous Sound Detection**

Primary dataset: **DCASE 2026 Task 2 — Noise-aware Unsupervised Anomalous Sound Detection for Machine Condition Monitoring**.

The dataset provides normal and anomalous operating sounds, environmental noise, two-channel recordings, and domain shifts. Development data contains seven machine types including fan, gearbox, bearing, slide rail, ToyCar, and valve variants.

The ML system should ultimately answer:

```text
Normal machine sound → NORMAL
Abnormal machine sound → ANOMALY
```

Do not assume every anomaly corresponds to a specific fault type unless the dataset labels support that classification.

---

# 3. PRIMARY RESEARCH CONTRIBUTION

Do NOT claim FFT, frequency-band triggering, or adaptive thresholds alone are novel.

Implement and compare:

### Baseline A — Fixed Amplitude

```text
RMS/Energy → fixed threshold → trigger
```

### Baseline B — Fixed Frequency Band

```text
FFT → selected band → fixed threshold → trigger
```

### Baseline C — Adaptive SNR

```text
FFT → adaptive noise floor → SNR → threshold → trigger
```

### Proposed Method

## Adaptive Multi-Band Resource-Aware Acoustic Gate

```text
Audio
 ↓
FFT
 ↓
Multi-band spectral analysis
 ↓
Adaptive noise estimation
 ↓
SNR + lightweight spectral features
 ↓
Temporal persistence
 ↓
Acoustic relevance score
 ↓
Resource-aware selectivity
 ↓
TRIGGER / IGNORE
```

The research question is:

> Can the proposed lightweight gate reduce unnecessary downstream anomaly processing while preserving high recall for machine anomalies under changing environmental noise and operating conditions?

---

# 4. PROJECT STRUCTURE

Create:

```text
machine-acoustic-monitor/
├── dsp/
│   ├── audio_loader.py
│   ├── framing.py
│   ├── fft_processor.py
│   ├── noise_estimator.py
│   ├── spectral_features.py
│   ├── baseline_gates.py
│   ├── adaptive_gate.py
│   ├── trigger_state.py
│   └── visualization.py
│
├── ml/
│   ├── preprocessing/
│   │   ├── audio.py
│   │   └── spectrogram.py
│   ├── models/
│   │   ├── cnn.py
│   │   ├── efficientnet.py
│   │   └── inference.py
│   ├── training/
│   │   ├── dataset.py
│   │   ├── train.py
│   │   └── evaluate.py
│   └── api/
│       └── main.py
│
├── backend/
│   ├── src/
│   │   ├── server.js
│   │   ├── routes/
│   │   ├── controllers/
│   │   ├── services/
│   │   ├── database/
│   │   └── middleware/
│   └── uploads/
│
├── frontend/
│   └── src/
│       ├── components/
│       ├── pages/
│       ├── services/
│       ├── hooks/
│       └── utils/
│
├── data/
│   ├── raw/
│   ├── sample/
│   └── processed/
│
├── experiments/
│   ├── configs/
│   ├── results/
│   ├── plots/
│   ├── ablations/
│   └── compare_gates.py
│
├── docs/
│   ├── architecture.md
│   ├── adaptive_gate.md
│   ├── novelty.md
│   ├── experiments.md
│   └── ml_pipeline.md
│
├── README.md
├── .env.example
└── docker-compose.yml
```

---

# 5. AUDIO INPUT

Support:

* PCM WAV
* mono/stereo
* automatic mono conversion
* automatic resampling
* default processing sample rate: 16 kHz
* 2-second and 10-second clips
* configurable input sample rate

For DCASE data, support its actual source format without destroying channel information before the appropriate processing stage. DCASE 2026 provides two-channel near/far recordings for the development dataset.

---

# 6. FFT/STFT ENGINE

Implement real FFT/STFT processing with NumPy/SciPy.

Configurable:

```text
sample_rate
fft_size
hop_length
window = Hann
min_frequency
max_frequency
```

For each frame calculate:

* magnitude spectrum
* total energy
* dominant frequency
* configurable frequency-band energies

Do not hardcode application-specific bands.

---

# 7. ADAPTIVE NOISE MODEL

Implement a robust running noise-floor estimator supporting:

* moving median/percentile
* exponential moving average
* configurable adaptation rate
* event-aware noise-model freezing

Calculate:

```text
SNR = signal level relative to estimated background
```

The gate must adapt to:

```text
quiet → noisy
noisy → quiet
```

This is particularly important because the DCASE 2026 task explicitly introduces environmental-noise and operating-condition domain shifts.

---

# 8. SPECTRAL FEATURES

Implement configurable lightweight features:

```text
SNR
Band Energy Ratio
Dominant Frequency
Spectral Centroid
Spectral Flatness
Spectral Entropy
Spectral Flux
```

Do not automatically use every feature.

Make feature selection configurable so experiments can determine which features improve anomaly-event gating.

---

# 9. TEMPORAL DECISION

Implement:

* frame-level scores
* rolling score
* consecutive-frame confirmation
* minimum event duration
* hysteresis
* release condition

Example:

```text
score > trigger threshold for N frames
        ↓
TRIGGER
```

and:

```text
score < release threshold for M frames
        ↓
END EVENT
```

Prevent rapid trigger/no-trigger oscillation.

---

# 10. RESOURCE-AWARE GATING

Implement a configurable resource state:

```text
resource_level = 1.0
resource_level = 0.5
resource_level = 0.2
```

Higher resource availability may make the gate more permissive.

Lower resource availability should increase selectivity.

Keep these two mechanisms separately configurable:

```text
Acoustic Adaptation
Resource Adaptation
```

so their individual effects can be measured.

For the current software prototype, resource state is simulated.

Future interpretation:

```text
resource_level
→ future ESP32 battery/compute/network state
```

---

# 11. GATE OUTPUT

Return machine-readable metadata:

```json
{
  "triggered": true,
  "trigger_score": 0.81,
  "threshold": 0.64,
  "snr_db": 11.7,
  "noise_floor": 0.23,
  "dominant_frequency": 3240,
  "band_energy_ratio": 0.72,
  "spectral_entropy": 0.41,
  "start_time": 3.42,
  "end_time": 5.18,
  "resource_level": 0.8,
  "gate_version": "AMRAG-v1"
}
```

All values must be calculated from the actual signal.

---

# 12. MACHINE ANOMALY ML PIPELINE

Only triggered audio should enter the expensive ML stage.

```text
Triggered Audio
 ↓
Mono / Channel Selection
 ↓
Resample
 ↓
Normalize
 ↓
Mel Spectrogram
 ↓
CNN / EfficientNet
 ↓
NORMAL / ANOMALY
```

Because DCASE 2026 is an **unsupervised anomalous sound detection** task, design the main ML path to support a **normal-reference anomaly-detection workflow**, not just a conventional supervised classifier.

Implement the architecture so both approaches are possible:

### A. Baseline supervised experiment

When normal/anomaly labels are explicitly available for development experiments.

### B. Unsupervised anomaly detection

Train/reference-model using normal machine recordings and compute an anomaly score for unseen recordings.

Prefer the unsupervised path for the main DCASE-style experiment.

---

# 13. MEL SPECTROGRAM

For triggered audio:

```text
Audio
 ↓
STFT
 ↓
Mel filter bank
 ↓
Power
 ↓
dB
 ↓
Normalization
 ↓
Model input
```

Configurable:

```text
sample_rate=16000
n_fft=1024
hop_length=512
n_mels=128
fmin=20
fmax=8000
```

Training and inference preprocessing must be identical.

---

# 14. CNN + EFFICIENTNET

Implement:

```text
MODEL_TYPE=cnn
MODEL_TYPE=efficientnet
```

CNN:

```text
Conv2D
BatchNorm
ReLU
MaxPool
Conv2D
BatchNorm
ReLU
MaxPool
Conv2D
GlobalAveragePooling
Dense
Dropout
Output
```

EfficientNet:

* TensorFlow/Keras
* transfer learning
* configurable fine-tuning

Save:

```text
model.keras
class_names.json
config.json
metrics.json
training_history.json
```

If no trained model is available:

```text
MODEL_UNAVAILABLE
```

Never fabricate predictions or metrics.

---

# 15. MACHINE DATASET PIPELINE

Primary dataset:

**DCASE 2026 Task 2**

Support its development data structure and preserve:

* machine type
* normal/anomaly label where available
* source/target domain
* environmental-noise conditions
* other provided attributes

DCASE 2026 includes seven development machine types and two-channel near/far recordings with 990 source-domain normal training clips and target-domain samples, plus normal/anomalous test data.

Create a preprocessing utility that can:

```text
validate dataset
→ inspect metadata
→ generate manifests
→ extract clips/windows
→ preserve source/recording identity
→ prevent leakage
```

Never randomly split related chunks from the same source recording across train and test.

---

# 16. EXPERIMENTAL COMPARISON

Create:

```bash
python experiments/compare_gates.py
```

Run:

```text
Fixed Amplitude
vs
Fixed Frequency Band
vs
Adaptive SNR
vs
Adaptive Multi-Band Resource-Aware Gate
```

using identical machine recordings.

Measure:

```text
Precision
Recall
F1
False Positive Rate
False Negative Rate
Trigger Rate
Anomaly-event recall
Audio rejected %
ML invocations avoided %
```

If an unsupervised anomaly detector is used, also report appropriate anomaly-detection metrics available from the dataset/evaluation protocol.

Do not force classification metrics where the evaluation protocol does not support them.

---

# 17. ABLATION STUDY

Implement:

```text
A: Energy only
B: Energy + SNR
C: Energy + SNR + Multi-band
D: + Spectral features
E: + Temporal persistence
F: + Resource awareness
```

Compare each stage.

Generate plots and tables showing whether each addition improves the trade-off between:

```text
anomaly recall
vs
false triggers
vs
downstream processing
```

---

# 18. NO-HARDWARE DEMO

Provide a software-only simulator:

```bash
python -m dsp.simulator path/to/audio.wav
```

The simulator must show:

```text
Audio
 ↓
FFT
 ↓
Noise Floor
 ↓
Gate Score
 ↓
TRIGGER / IGNORE
```

For triggered events:

```text
TRIGGER
 ↓
Mel Spectrogram
 ↓
ML anomaly score
 ↓
NORMAL / ANOMALY
```

For ignored events:

```text
IGNORE
 ↓
ML NOT EXECUTED
```

This demonstrates the intended edge-computation-saving behavior without hardware.

---

# 19. REACT DASHBOARD

Create:

## Dashboard

Show:

* total recordings
* triggered events
* ignored recordings
* anomalies detected
* normal events
* ML processing avoided
* trigger rate

## Gate Analysis

Show:

* waveform
* FFT spectrum
* frequency bands
* adaptive noise floor
* threshold
* trigger score
* temporal trigger regions
* resource level
* final decision

Allow comparison of:

```text
Fixed Amplitude
Fixed Band
Adaptive SNR
Proposed Gate
```

## Machine Analysis

Show:

* machine type
* normal/anomaly state
* anomaly score
* confidence/score where applicable
* environmental/domain metadata
* processing time

## Event Details

Show:

* audio player
* waveform
* FFT
* gate metadata
* Mel Spectrogram
* anomaly score
* ML model/version
* processing time

## Experiment Results

Show:

* gate comparison
* ablation results
* processing reduction
* false-trigger reduction
* anomaly recall

All values must come from actual experiment/API results.

---

# 20. NODE.JS BACKEND

Implement:

```text
GET  /api/health

POST /api/audio/upload
GET  /api/audio/:id

POST /api/events
GET  /api/events
GET  /api/events/:id

GET /api/stats
GET /api/experiments
```

Pipeline:

```text
Upload
 ↓
Validate
 ↓
Store
 ↓
Run gate
 ↓
IGNORE → persist gate result, stop expensive ML
TRIGGER → persist event → invoke ML → persist result
```

Node.js orchestrates the workflow.

Python owns DSP and ML.

---

# 21. DATABASE

Use SQLite.

### events

```text
id
machine_type
audio_path
timestamp
duration
sample_rate
gate_type
triggered
trigger_score
gate_metadata
processing_status
spectrogram_path
created_at
```

### predictions

```text
id
event_id
result
anomaly_score
confidence
model_version
processing_time
created_at
```

### experiments

```text
id
experiment_name
gate_type
dataset
config
metrics
created_at
```

Store audio/spectrogram files outside the database.

---

# 22. API / SECURITY

Implement:

* upload validation
* file-size limits
* safe filenames
* path traversal protection
* configurable CORS
* environment variables
* structured logging
* meaningful HTTP errors
* no secrets committed

Provide:

```text
.env.example
```

---

# 23. TESTING

### DSP

Test:

* silence
* stationary noise
* changing noise
* machine-like tones
* transient impulses
* threshold adaptation
* hysteresis
* temporal persistence
* resource adaptation

### ML

Test:

* audio loading
* preprocessing
* spectrogram shape
* model loading
* normal-reference inference
* missing-model handling

### Backend

Test:

* upload
* event creation
* gate invocation
* ML invocation
* failure recovery
* retrieval

### Integration

Verify:

```text
WAV
→ Gate
→ Trigger/Ignore
→ ML only when triggered
→ Database
→ API
→ Dashboard
```

---

# 24. DOCUMENTATION

Create:

```text
docs/architecture.md
docs/adaptive_gate.md
docs/novelty.md
docs/experiments.md
docs/ml_pipeline.md
```

Explain clearly:

### What is established

```text
FFT
Fixed energy/frequency triggering
Adaptive SNR concepts
Acoustic anomaly detection
Edge processing
```

### What we investigate

```text
Adaptive Multi-Band
+
Temporal Persistence
+
Resource-Aware Selectivity
+
Selective Downstream Anomaly Processing
```

Do not claim guaranteed novelty.

Frame this as the **research hypothesis/contribution under investigation**, to be supported by experimental evidence and literature review.

---

# 25. FUTURE ESP32 INTERFACE

The gate must expose a streaming API concept such as:

```python
process_frame(samples)
```

Current:

```text
AudioFileSource → AdaptiveGate
```

Future:

```text
ESP32AudioSource → AdaptiveGate
```

Do not implement ESP32 code now.

Keep the gate computationally lightweight enough that a later C/C++ implementation on ESP32 is realistic.

---

# 26. DOCKER

Provide:

```text
frontend
backend
ml-service
```

through Docker Compose.

Document:

```bash
docker compose up --build
```

---

# 27. FINAL ACCEPTANCE CRITERIA

Do not declare completion until:

* [ ] WAV input works without hardware
* [ ] DCASE-compatible dataset pipeline works
* [ ] FFT works
* [ ] fixed amplitude baseline works
* [ ] fixed frequency baseline works
* [ ] adaptive SNR baseline works
* [ ] proposed adaptive multi-band gate works
* [ ] adaptive noise estimation works
* [ ] temporal persistence works
* [ ] hysteresis works
* [ ] resource-aware selectivity works
* [ ] gate metadata is explainable
* [ ] all four gates run on identical data
* [ ] ablation experiments work
* [ ] triggered-only ML execution works
* [ ] Mel Spectrogram generation works
* [ ] CNN exists
* [ ] EfficientNet exists
* [ ] unsupervised anomaly-detection pathway exists
* [ ] missing-model behavior works
* [ ] events are stored
* [ ] anomaly results are stored
* [ ] dashboard displays real data
* [ ] FFT/gate visualizations work
* [ ] experiment results are visualized
* [ ] processing-reduction metrics are calculated
* [ ] no fabricated metrics or predictions
* [ ] no hardcoded secrets
* [ ] future ESP32 streaming boundary is clearly defined

---

# FINAL INSTRUCTION

Build this as a **research-grade machine acoustic condition-monitoring prototype**.

The central research component is:

> **Adaptive Multi-Band Resource-Aware Acoustic Gating**

and its comparison against:

1. Fixed amplitude threshold
2. Fixed frequency-band threshold
3. Adaptive SNR threshold

The final software must demonstrate:

```text
Machine Audio
   ↓
Adaptive Acoustic Gate
   ↓
┌───────────────┬───────────────┐
│               │
IGNORE        TRIGGER
│               │
No ML          ML
                ↓
        Normal / Anomaly
```

The goal is to experimentally determine whether the proposed gate can **reduce unnecessary downstream anomaly-processing workload while preserving anomaly detection performance under noise and operating-condition changes**.

Do not fabricate novelty, accuracy, energy savings, or experimental results.

Prioritize:

**correct DSP → rigorous baselines → reproducible experiments → measurable gating improvement → machine anomaly detection → reliable backend → useful dashboard → future ESP32 portability.**

