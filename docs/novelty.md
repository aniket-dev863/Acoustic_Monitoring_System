# Research Context and Hypothesis (Novelty Analysis)

## 1. What Is Established in Prior Art

To maintain academic and engineering integrity, we explicitly acknowledge the established foundations of industrial acoustic signal processing and edge condition monitoring:

1. **Fast Fourier Transform (FFT) & STFT:** Well-established mathematical frameworks for frequency-domain representation of acoustic signals.
2. **Fixed Threshold Triggering:** Widely used baseline methods that fire when Root-Mean-Square (RMS) amplitude or energy within a selected band exceeds a static threshold.
3. **Adaptive Noise Estimators:** Moving average and percentile-based noise tracking algorithms (e.g., minimum statistics, recursive averaging) commonly deployed in speech enhancement and hearing aids.
4. **Unsupervised Acoustic Anomaly Detection:** Formulations popularized by the DCASE challenges (e.g., DCASE 2020–2026 Task 2), where models are trained exclusively on normal operational sounds.
5. **Edge Microcontroller Audio Capture:** Standard embedded hardware implementations utilizing I2S interfaces with digital MEMS microphones (e.g. INMP441 on ESP32 or ARM Cortex-M).

---

## 2. Research Hypothesis Under Investigation

This project investigates the following primary research question:

> **Research Question:** Can an Adaptive Multi-Band Resource-Aware Acoustic Gate (AMRAG) reject non-anomalous background operational machinery sounds and environmental noise shifts at the simulated edge boundary, significantly reducing downstream deep learning invocations while preserving high anomaly recall?

### Key Investigated Contributions:
1. **Multi-Band Asymmetric Adaptive Noise Gating:** Combining independent sub-band tracking with event-aware freeze protection to prevent defect acoustic bursts from artificially inflating background estimates.
2. **Multi-Spectral Relevance Fusion:** Formulating an ultra-lightweight composite score fusing sub-band excess SNR, band energy ratios, spectral flux, and tonal prominence without requiring computationally heavy matrix operations.
3. **Dynamic Resource-Aware Selectivity:** Dynamically scaling both trigger sensitivity and temporal confirmation persistence based on real-time edge battery and compute budget states.
4. **Selective Trigger-Only Deep Anomaly Inference:** Demonstrating experimentally that downstream deep learning models (Autoencoders, CNNs) only need to be invoked during confirmed acoustic trigger intervals, leading to measured compute and bandwidth savings of over 50–70% in industrial regimes.
