# System Architecture: Machine Acoustic Condition-Monitoring

## 1. Executive Summary

This system implements an end-to-end, software-only machine acoustic condition-monitoring architecture designed around an **Adaptive Multi-Band Resource-Aware Acoustic Gate (AMRAG)**.

The primary objective is to dramatically reduce edge-to-cloud transmission and expensive deep learning anomaly detection workloads by rejecting normal operational sounds and background noise at the edge, selectively forwarding only acoustically relevant events to the ML stage.

---

## 2. High-Level Data Flow

```text
               ┌────────────────────────────────────────────────────────┐
               │              Edge / DSP Gating Layer                   │
               └────────────────────────────────────────────────────────┘
                                           │
                                  PCM WAV Audio Stream
                                (16 kHz, 1024 FFT, Hann)
                                           │
                                           ▼
                                 FFT / STFT Processor
                          (Sub-bands: Low, Mid, High)
                                           │
                                           ▼
                                 Adaptive Noise Model
                        (Asymmetric EMA + Percentile Floor)
                                           │
                                           ▼
                                Spectral Feature Extractor
                      (SNR, BER, Flux, Flatness, Entropy)
                                           │
                                           ▼
                            Acoustic Relevance Scorer &
                            Resource-Aware Selectivity
                                           │
                                           ▼
                           Temporal Persistence Engine
                        (Hysteresis, M-frame confirmation)
                                           │
                                ┌──────────┴──────────┐
                                │                     │
                             IGNORE                TRIGGER
                                │                     │
                                ▼                     ▼
                        Normal Background       Acoustic Event
                          (ML Skipped,           Forwarded to
                         Compute Saved)           ML Pipeline
                                                      │
                                                      ▼
                                            Log-Mel Spectrogram
                                            (128 Mels, Dynamic dB)
                                                      │
                                                      ▼
                                           Deep Anomaly Detection
                                            (Autoencoder / CNN)
                                                      │
                                                      ▼
                                              NORMAL / ANOMALY
```

---

## 3. Subsystem Breakdown

### 3.1 DSP Edge Gating Layer (`dsp/`)
* **AudioLoader (`dsp/audio_loader.py`):** High-fidelity audio ingestion with polyphase resampling (16 kHz target) and multi-channel microphone support (e.g. Near/Far channels from DCASE 2026).
* **FrameBuffer (`dsp/framing.py`):** FIFO circular streaming buffer that exposes a continuous streaming interface (`process_frame`) directly analogous to an embedded microcontroller (e.g. ESP32 I2S DMA circular buffer).
* **FFTProcessor (`dsp/fft_processor.py`):** Windowed real FFT generating linear magnitude spectra, power spectra, total energy, and configurable sub-band powers.
* **AdaptiveNoiseEstimator (`dsp/noise_estimator.py`):** Asymmetric tracking with event-aware model freezing to prevent anomaly spikes from corrupting the background noise baseline.
* **FeatureExtractor (`dsp/spectral_features.py`):** Lightweight feature calculation: full/band SNR, Band Energy Ratio (BER), Spectral Centroid, Flatness, Entropy, and Flux.
* **TriggerStateMachine (`dsp/trigger_state.py`):** Dual-threshold hysteresis, consecutive confirmation frames, and minimum hold duration to eliminate false-trigger chattering.
* **AdaptiveMultiBandResourceGate (`dsp/adaptive_gate.py`):** The core proposed gate (AMRAG-v1). Modulates trigger selectivity dynamically according to edge battery/compute availability.

### 3.2 ML Anomaly Detection Layer (`ml/`)
* **MelSpectrogramExtractor (`ml/preprocessing/spectrogram.py`):** HTK Mel filterbank (128 bins, $20\text{ Hz} - 8000\text{ Hz}$) with peak dynamic range compression.
* **Unsupervised Autoencoder (`ml/models/cnn.py`):** Fully convolutional autoencoder trained solely on normal operational sounds (DCASE 2026 style). Computes reconstruction MSE anomaly scores.
* **Supervised CNN & EfficientNet (`ml/models/cnn.py`, `ml/models/efficientnet.py`):** 2D convolutional networks and transfer-learning backbones for baseline supervised benchmarking.
* **Inference Engine (`ml/models/inference.py`):** Unified scoring interface with graceful `MODEL_UNAVAILABLE` fallback when weights are missing.
* **FastAPI Microservice (`ml/api/main.py`):** Asynchronous REST API serving `/predict`, `/extract-spectrogram`, and `/gate-and-predict`.

### 3.3 Backend Orchestration Layer (`backend/`)
* **Express & Node.js (`backend/src/server.js`):** High-concurrency REST API handling audio file uploads, validation, and pipeline coordination.
* **SQLite Database (`backend/src/database/db.js`):** Relational storage tracking `events`, `predictions`, and `experiments`. Audio and spectrogram image files reside on the filesystem outside the DB.
* **GateService (`backend/src/services/gateService.js`):** Dual invocation engine: queries the Python FastAPI service via HTTP, with automatic fallback to native Python subprocess execution if the microservice is offline.

### 3.4 Frontend Monitoring Dashboard (`frontend/`)
* **React + Vite + Tailwind CSS (`frontend/src/`):**
  * **Dashboard:** Real-time KPI cards (compute saved, trigger rate), interactive live gate simulator, recent ingestion logs.
  * **Gate Analysis:** Sub-band power breakdowns, noise floor tracking, resource level sliders.
  * **Machine Analysis:** Condition breakdowns across machine categories (fan, bearing, gearbox, slide rail, valve).
  * **Event Details:** HTML5 audio player, spectrogram viewer, and explainable DSP metadata inspection.
  * **Experiments:** 4-way gate benchmark tables and 6-stage ablation progression charts.
