# Machine Anomaly ML Pipeline Documentation

## 1. Overview

The machine acoustic anomaly detection pipeline complies with the unsupervised task formulation of **DCASE 2026 Task 2** (Noise-aware Unsupervised Anomalous Sound Detection for Machine Condition Monitoring).

Only audio frames or clips that satisfy the edge gate triggering conditions are forwarded to this pipeline.

---

## 2. Audio & Spectrogram Preprocessing

To ensure 100% mathematical consistency between training and runtime inference:

1. **Mono Normalization:** Audio signals are downmixed to mono or selected by channel index (e.g. Near mic channel in DCASE 2026).
2. **Resampling:** High-precision polyphase resampling to $f_s = 16{,}000\text{ Hz}$.
3. **STFT Analysis:**
   * Window: Hann, $N = 1024$ samples (64 ms).
   * Hop Size: $H = 512$ samples (32 ms).
4. **Mel Filterbank:**
   * $N_{mels} = 128$ triangular filters.
   * Frequency Range: $20\text{ Hz} \le f \le 8{,}000\text{ Hz}$.
   * Formula: HTK Mel conversion scale with Slaney area normalization.
5. **Decibel Dynamic Clamping:**
   * Power to dB: $10 \log_{10}(P + 10^{-10})$.
   * Dynamic Range: Clamped to top 80 dB below peak.
   * Scaled to $[0.0, 1.0]$.

---

## 3. Model Architectures

### 3.1 Unsupervised Spectrogram Autoencoder (Primary DCASE Model)
* **Design Philosophy:** Trained exclusively on normal machine operating audio.
* **Mechanism:** The network learns the low-dimensional manifold of normal operational sounds. When an anomalous defect sound is ingested, the network cannot reconstruct the novel spectral patterns accurately.
* **Anomaly Score:** Reconstruction Mean Squared Error (MSE):
  $$\mathcal{A}(x) = \frac{1}{M \cdot T} \sum_{m=1}^M \sum_{t=1}^T |S[m, t] - \hat{S}[m, t]|^2$$
* **Decision Threshold:** Calibrated at the 95th percentile of normal training errors $\times 1.5$.

### 3.2 2D Convolutional Classifier (Baseline Supervised Model)
* **Architecture:**
  ```text
  Conv2D(32, 3x3) -> BatchNorm -> ReLU -> MaxPool(2x2)
  Conv2D(64, 3x3) -> BatchNorm -> ReLU -> MaxPool(2x2)
  Conv2D(128, 3x3) -> BatchNorm -> ReLU -> GlobalAveragePooling
  Dense(64) -> Dropout(0.3) -> Output Dense(2, Softmax)
  ```

### 3.3 Acoustic EfficientNet (Transfer Learning Option)
* EfficientNet-B0 backbone adapted for 1-channel Mel spectrogram input with fine-tuning.

---

## 4. Operational Safety and Missing-Model Behavior

If the ML service is queried before models are trained, it responds with:
```json
{
  "status": "MODEL_UNAVAILABLE",
  "result": "MODEL_UNAVAILABLE",
  "anomaly_score": null,
  "confidence": null,
  "model_version": "None"
}
```
Predictions and metrics are **never fabricated**.
