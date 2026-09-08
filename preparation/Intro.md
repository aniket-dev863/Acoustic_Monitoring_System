# Three-Minute Project Introduction

## Why
In industrial condition monitoring, acoustic sensing is one of the most effective ways to detect early mechanical failures in motors, pumps, and bearings before severe damage occurs. But in real-world deployments on battery-powered edge sensor nodes, continuously running deep neural networks or streaming raw high-bandwidth audio over wireless networks drains batteries in days and causes heavy network congestion. 

Over 90% of industrial sound is normal, repetitive background noise. However, naive fixed amplitude or energy thresholds fail because ambient factory noise continuously shifts—forklifts drive by, ventilation cycles turn on, and operational regimes change. I wanted to solve this edge energy bottleneck by building a lightweight, intelligent acoustic gatekeeper that filters out stationary machine noise at the edge and wakes up expensive deep learning models only when genuine acoustic anomalies occur.

## What
I built a software-only machine acoustic condition-monitoring system centered on an **Adaptive Multi-Band Resource-Aware Acoustic Gate**, or **AMRAG**. The system simulates an edge-node pipeline compliant with the DCASE 2026 unsupervised anomalous sound detection challenge. 

When audio is ingested, it first passes through the lightweight DSP gate. If the audio is normal operational hum or ambient factory noise, the gate marks it as ignored, logs the telemetry, and completely skips the downstream ML stage—saving compute and power. If a significant acoustic event is confirmed—such as a bearing race crack or fan unbalance—the gate triggers, forwards the audio, generates a 128-mel log-spectrogram, and executes an unsupervised convolutional autoencoder to compute a precise anomaly score.

## How
The architecture is divided into a pure Python DSP engine, an unsupervised PyTorch ML pipeline, an Express and SQLite orchestration backend, and a React Vite dashboard. 

On the DSP side, rather than relying on a single energy threshold, AMRAG decomposes audio frames into sub-bands and tracks an asymmetric background noise floor with event-aware freeze protection. This ensures that prolonged defect noise doesn't corrupt the baseline and mask itself. We fuse multi-band SNR, spectral flux, and Wiener flatness with a temporal hysteresis state machine to eliminate false-trigger chattering. Furthermore, the gate dynamically scales its sensitivity based on the edge node's remaining battery budget.

To evaluate this rigorously, I implemented three industry baselines: Fixed Amplitude, Fixed Band, and Adaptive SNR. In our benchmark experiments on identical audio, the fixed amplitude baseline suffered a 75% false positive rate, whereas our proposed gate achieved a 0% false positive rate at constrained resource levels, successfully avoiding over 71% of downstream deep learning invocations while preserving anomaly recall.

## What Now
Currently, the system is a validated software prototype with simulated streaming framing that matches microcontroller I2S ring-buffer callbacks. 

The biggest current limitation is that the edge hardware is simulated via software WAV framing rather than deployed on a physical ESP32 or ARM Cortex microcontroller. The most valuable next step is porting the core DSP algorithms into C using the ESP-DSP library and benchmarking them directly on an ESP32-S3 with an INMP441 I2S microphone. 

I'd be glad to dive deeper into the multi-band DSP mathematics, the event-freezing noise model, or how the temporal state machine suppresses false triggers.

---

## Thirty-Second Version

> "I built an edge acoustic condition-monitoring system centered around an Adaptive Multi-Band Resource-Aware Acoustic Gate. In industrial IoT, running deep neural networks 24/7 on battery-powered edge sensors is completely impractical because most audio is normal operational hum. My system implements a lightweight DSP gatekeeper that tracks background noise floors across sub-bands with temporal hysteresis, selectively triggering expensive convolutional autoencoder inference only on acoustically significant events. In empirical benchmark comparisons, our gate avoided over 71% of unnecessary ML invocations while eliminating false alarms under shifting factory noise."

---

## Key Points to Remember

1. **The Core Problem:** Continuous edge deep learning drains batteries; naive fixed amplitude thresholds trigger constantly on shifting factory noise (75% FPR).
2. **The Innovation (AMRAG):** Sub-band spectral analysis + asymmetric noise tracking + event-aware baseline freeze + temporal hysteresis + resource scaling.
3. **The Workflow:** `Audio Stream` $\rightarrow$ `Lightweight Gate` $\rightarrow$ `IGNORE (Skip ML, Save Compute)` or `TRIGGER (Run Mel-Spectrogram + Autoencoder)`.
4. **Empirical Evidence:** Avoids 57% to 71.4% of downstream ML invocations on identical benchmark audio ($F_1 = 0.800$ at Stage F).
5. **DCASE 2026 Compliance:** Unsupervised normal-reference anomaly detection (trained exclusively on normal operational sounds; flags anomalies via reconstruction MSE).
6. **Hardware-Ready Boundary:** Built around a stateful circular FIFO `FrameBuffer` matching embedded microcontroller I2S DMA interrupts.
7. **Honest Limitation:** Currently a software prototype using simulated streaming audio; next step is a physical C99 port for ESP32-S3.
