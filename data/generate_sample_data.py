"""
Synthetic Audio Data Generator for Machine Acoustic Condition Monitoring.

Generates physically grounded synthetic machine operating sounds matching
DCASE 2026 industrial acoustics (fans, gearboxes, bearings, valves):
- Normal operational hums with stationary/drifting background noise
- Anomalies: bearing cage/ball impact clicks, valve leakage hissing, tooth breakage
- Non-stationary factory noise shifts (quiet -> noisy -> quiet)
- Isolated transient impulses to test temporal persistence rejection
"""

from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import scipy.signal
import soundfile as sf


def generate_pink_noise(num_samples: int) -> np.ndarray:
    """Generates 1/f pink noise using Voss-McCartney algorithm."""
    uneven = num_samples % 2
    X = np.random.randn(num_samples // 2 + 1 + uneven) + 1j * np.random.randn(num_samples // 2 + 1 + uneven)
    S = np.sqrt(np.arange(len(X)) + 1.0)
    y = (np.fft.irfft(X / S)).real
    if uneven:
        y = y[:-1]
    return y / (np.max(np.abs(y)) + 1e-12)


def generate_synthetic_samples(output_dir: str = "data/sample", sample_rate: int = 16000) -> None:
    """Generates synthetic WAV files for local testing and research validation."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    t_10s = np.linspace(0, 10.0, int(10.0 * sample_rate), endpoint=False)
    t_3s = np.linspace(0, 3.0, int(3.0 * sample_rate), endpoint=False)

    # 1. Pure Silence / Ambient sensor floor (-60 dB)
    silence = np.random.normal(0, 0.0005, size=len(t_3s)).astype(np.float32)
    sf.write(out / "silence.wav", silence, sample_rate)

    # 2. Normal Fan: steady 60 Hz hum + blade passing harmonics (180 Hz, 360 Hz) + pink noise
    fan_normal = (
        0.10 * np.sin(2 * np.pi * 60 * t_10s)
        + 0.08 * np.sin(2 * np.pi * 180 * t_10s)
        + 0.05 * np.sin(2 * np.pi * 360 * t_10s)
        + 0.04 * generate_pink_noise(len(t_10s))
    ).astype(np.float32)
    sf.write(out / "fan_normal.wav", fan_normal, sample_rate)

    # 3. Anomaly Fan: blade unbalance + bearing squeal (3200 Hz tone) and periodic shudder
    fan_anomaly = (
        fan_normal.copy()
        + 0.18 * np.sin(2 * np.pi * 3200 * t_10s) * (0.5 + 0.5 * np.sin(2 * np.pi * 4 * t_10s))
    ).astype(np.float32)
    # Add fault surge in the middle (from 4.0s to 7.0s)
    surge_mask = (t_10s >= 4.0) & (t_10s <= 7.5)
    fan_anomaly[surge_mask] += 0.25 * np.random.normal(0, 1, size=int(np.sum(surge_mask)))
    sf.write(out / "fan_anomaly.wav", fan_anomaly, sample_rate)

    # 4. Normal Bearing / Gearbox: steady shaft rotation and tooth mesh tone (1200 Hz)
    gearbox_normal = (
        0.08 * np.sin(2 * np.pi * 1200 * t_10s)
        + 0.04 * np.sin(2 * np.pi * 2400 * t_10s)
        + 0.03 * np.random.normal(0, 1, len(t_10s))
    ).astype(np.float32)
    sf.write(out / "bearing_normal.wav", gearbox_normal, sample_rate)

    # 5. Anomaly Bearing: sharp repeated impacts (defect on outer raceway ball pass frequency)
    bearing_anomaly = gearbox_normal.copy()
    impact_rate_hz = 15.0  # 15 impacts per sec
    impact_interval = int(sample_rate / impact_rate_hz)
    decay_samples = int(0.015 * sample_rate)  # 15ms impact decay
    decay_curve = np.exp(-np.linspace(0, 5, decay_samples))

    # Add impacts between 3.0s and 8.0s
    start_sample = int(3.0 * sample_rate)
    end_sample = int(8.0 * sample_rate)
    for sample_pos in range(start_sample, end_sample - decay_samples, impact_interval):
        impact_tone = np.sin(2 * np.pi * 4500 * np.linspace(0, 0.015, decay_samples))
        bearing_anomaly[sample_pos : sample_pos + decay_samples] += 0.35 * (decay_curve * impact_tone).astype(np.float32)
    sf.write(out / "bearing_anomaly.wav", bearing_anomaly, sample_rate)

    # 6. Isolated transient clicks (brief clicks < 2 frames) to test temporal persistence false-trigger immunity
    clicks = fan_normal.copy()
    click_pos = int(2.5 * sample_rate)
    clicks[click_pos : click_pos + 128] += 0.6  # Very brief 8ms spike
    click_pos2 = int(6.0 * sample_rate)
    clicks[click_pos2 : click_pos2 + 128] += 0.7
    sf.write(out / "isolated_transient_clicks.wav", clicks, sample_rate)

    # 7. Non-stationary changing background noise (quiet -> factory drone -> quiet) with defect in the middle
    t_changing = np.linspace(0, 12.0, int(12.0 * sample_rate), endpoint=False)
    noise_envelope = 0.02 + 0.15 * (1.0 / (1.0 + np.exp(-2.0 * (t_changing - 3.0)))) * (1.0 / (1.0 + np.exp(2.0 * (t_changing - 9.0))))
    changing_noise = (generate_pink_noise(len(t_changing)) * noise_envelope).astype(np.float32)
    # Add defect between 6.0s and 7.5s (inside high-noise region)
    defect_mask = (t_changing >= 6.0) & (t_changing <= 7.5)
    changing_noise[defect_mask] += 0.22 * np.sin(2 * np.pi * 2800 * t_changing[defect_mask]).astype(np.float32)
    sf.write(out / "noise_shift_anomaly.wav", changing_noise, sample_rate)

    print(f"Successfully generated 6 synthetic audio benchmark samples in {output_dir}/")


if __name__ == "__main__":
    generate_synthetic_samples()
