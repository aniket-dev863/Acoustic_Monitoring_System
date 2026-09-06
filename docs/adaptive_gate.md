# Adaptive Multi-Band Resource-Aware Acoustic Gate (AMRAG)

## 1. Motivation

In industrial acoustic condition monitoring, continuous streaming of raw high-resolution audio to cloud inference endpoints or running deep convolutional neural networks directly on edge microcontrollers introduces severe battery drain, network bandwidth congestion, and high cloud compute costs.

Standard energy or amplitude triggering fails in real-world industrial environments:
* Background ambient noise fluctuates dynamically (factory forklifts, ventilation cycles, shifting operating regimes).
* Stationary machinery noise exhibits high baseline energy that easily overwhelms fixed thresholds.
* Low-energy, high-frequency defects (such as early-stage bearing raceway fatigue or micro-leaks in valves) remain hidden beneath high overall RMS levels.

**AMRAG** solves this problem by decomposing the spectrum into distinct sub-bands, dynamically tracking the background floor with event-aware freeze protection, evaluating multi-spectral acoustic relevance, applying temporal hysteresis, and scaling selectivity according to available edge compute/battery resources.

---

## 2. Mathematical Formulation

### 2.1 Sub-Band Decomposition
Audio frames $x[n]$ of length $N = 1024$ windowed with a Hann window $w[n]$ are transformed into the frequency domain:

$$X[k] = \sum_{n=0}^{N-1} x[n] w[n] e^{-j 2\pi k n / N}, \quad k = 0, \dots, \frac{N}{2}$$

Power spectrum:

$$P[k] = \frac{|X[k]|^2}{N / 2}$$

Energy across configured sub-band $b = (k_{min}, k_{max})$:

$$E_b = \sum_{k=k_{min}}^{k_{max}} P[k]$$

Default industrial bands:
* **Low (20 – 800 Hz):** Structural looseness, sub-harmonics, rotational unbalance.
* **Mid (800 – 3000 Hz):** Gear meshing, blade pass frequencies, fundamental bearing impact resonance.
* **High (3000 – 8000 Hz):** High-frequency friction, surface rubbing, valve fluid hissing.

---

### 2.2 Asymmetric Adaptive Noise Model with Event-Aware Freezing
To track shifting ambient noise while preventing defect energy bursts from corrupting the baseline, the noise floor $\hat{N}_b$ is updated asymmetrically:

$$\hat{N}_b[t] = (1 - \alpha) \hat{N}_b[t-1] + \alpha \cdot \text{Percentile}_{25}(E_b[\tau])$$

$$\alpha = \begin{cases} \alpha_{up} = 0.015 & \text{if } E_b[t] > \hat{N}_b[t-1] \quad \text{(slow upward adaptation)} \\ \alpha_{down} = 0.060 & \text{if } E_b[t] \le \hat{N}_b[t-1] \quad \text{(faster downward recovery)} \end{cases}$$

**Event Freeze Mechanism:**
When the trigger state machine transitions to `TRIGGERED`, updates to $\hat{N}_b[t]$ are held constant:

$$\hat{N}_b[t] = \hat{N}_b[t-1] \quad \forall t \in \text{Event}$$

This ensures that prolonged defect noise cannot inflate the background estimate and prematurely terminate the trigger.

---

### 2.3 Acoustic Relevance Score ($S_{acoustic}$)
The composite gate score is computed as a convex combination of normalized spectral indicators:

$$S_{acoustic} = w_{snr} S_{snr} + w_{ber} S_{ber} + w_{flux} S_{flux} + w_{entropy} S_{tonal}$$

1. **Sub-Band Excess SNR:**
   $$S_{snr} = \text{clip}\left(\frac{\max_b \text{SNR}_{b,\text{dB}}}{20.0}, 0.0, 1.0\right)$$

2. **Band Energy Ratio Imbalance:**
   $$S_{ber} = \text{clip}\left(1.5 \cdot \text{BER}_{high} + 0.8 \cdot \text{BER}_{mid}, 0.0, 1.0\right)$$

3. **Normalized Spectral Flux:**
   $$S_{flux} = \text{clip}\left(2.0 \cdot \|\bar{X}[t] - \bar{X}[t-1]\|_2, 0.0, 1.0\right)$$

4. **Tonal Peak Prominence (Inverted Shannon Entropy):**
   $$S_{tonal} = \text{clip}\left(1.0 - H_{spectral}, 0.0, 1.0\right)$$

---

### 2.4 Resource-Aware Selectivity Modulation
Given the simulated or physical edge node resource level $R \in [0.1, 1.0]$ (where $1.0$ represents mains power and $0.1$ represents critical battery preservation):

1. **Effective Trigger Threshold:**
   $$T_{effective} = T_{base} + (1.0 - R) \cdot \Delta T$$
   *(Default: $T_{base} = 0.55$, $\Delta T = 0.30$. At $R=0.2$, $T_{effective} \approx 0.79$)*

2. **Effective Release Threshold (Hysteresis):**
   $$T_{release} = 0.65 \cdot T_{effective}$$

3. **Required Consecutive Confirmation Frames:**
   $$N_{confirm} = \text{round}(N_{base} + (1.0 - R) \cdot 2.0)$$
