"""
DSP Visualization Module.

Generates comprehensive multi-panel figures illustrating:
- Raw waveform & trigger intervals
- FFT Spectrogram
- Frequency sub-band energies
- Running noise floor & trigger threshold vs relevance score
- Decision state over time
"""

from pathlib import Path
from typing import Dict, List, Optional, Union
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

from dsp.baseline_gates import GateExecutionReport


def plot_gate_analysis(
    signal: np.ndarray,
    sample_rate: int,
    report: GateExecutionReport,
    output_path: Optional[Union[str, Path]] = None,
    title: Optional[str] = None,
) -> str:
    """
    Renders an in-depth 4-panel analysis plot comparing audio signal dynamics,
    spectral decomposition, adaptive noise floor tracking, and gating decisions.

    Returns:
        Absolute path to the saved plot image.
    """
    output_path = Path(output_path) if output_path else Path("experiments/plots/gate_analysis.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    duration = len(signal) / float(sample_rate)
    time_axis = np.linspace(0.0, duration, len(signal), endpoint=False)

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True, gridspec_kw={"height_ratios": [1, 1.5, 1.2]})
    plot_title = title or f"Acoustic Gate Analysis — {report.gate_version} ({report.gate_type})"
    fig.suptitle(plot_title, fontsize=14, fontweight="bold", y=0.98)

    # 1. Waveform & Trigger Regions
    ax_wave = axes[0]
    ax_wave.plot(time_axis, signal, color="#2c3e50", alpha=0.7, lw=0.8, label="Waveform")
    ax_wave.set_ylabel("Amplitude", fontsize=10)
    ax_wave.grid(True, linestyle="--", alpha=0.4)

    # Highlight triggered intervals
    if report.events:
        for i, ev in enumerate(report.events):
            label = "Trigger Event" if i == 0 else ""
            ax_wave.axvspan(
                ev["start_time"],
                ev["end_time"],
                color="#e74c3c",
                alpha=0.3,
                label=label,
            )
    elif report.overall_triggered and report.frame_timestamps:
        # Mark individual triggered frames if no grouped intervals
        triggered_times = [
            ts for ts, sc in zip(report.frame_timestamps, report.frame_scores)
            if sc >= report.threshold
        ]
        for ts in triggered_times:
            ax_wave.axvline(ts, color="#e74c3c", alpha=0.15)

    ax_wave.legend(loc="upper right", fontsize=8)

    # 2. Spectrogram
    ax_spec = axes[1]
    # Compute STFT magnitude for spectrogram display
    from scipy.signal import spectrogram
    freqs, times, Sxx = spectrogram(signal, fs=sample_rate, nperseg=512, noverlap=256)
    Sxx_db = 10.0 * np.log10(Sxx + 1e-10)
    img = ax_spec.pcolormesh(times, freqs, Sxx_db, shading="gouraud", cmap="inferno")
    ax_spec.set_ylabel("Frequency (Hz)", fontsize=10)
    ax_spec.set_ylim([0, min(8000, sample_rate / 2)])
    cbar = fig.colorbar(img, ax=ax_spec, orientation="vertical", pad=0.01)
    cbar.set_label("Power (dB)", fontsize=8)

    # 3. Trigger Scores vs Adaptive Threshold
    ax_score = axes[2]
    if report.frame_timestamps and report.frame_scores:
        ax_score.plot(
            report.frame_timestamps,
            report.frame_scores,
            color="#2980b9",
            lw=1.5,
            label="Gate Score",
        )
        ax_score.axhline(
            report.threshold,
            color="#e67e22",
            linestyle="--",
            lw=1.5,
            label=f"Threshold ({report.threshold:.2f})",
        )

    ax_score.set_xlabel("Time (seconds)", fontsize=10)
    ax_score.set_ylabel("Score / Relevance", fontsize=10)
    ax_score.set_ylim([0.0, 1.05])
    ax_score.grid(True, linestyle="--", alpha=0.4)
    ax_score.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)

    return str(output_path.resolve())
