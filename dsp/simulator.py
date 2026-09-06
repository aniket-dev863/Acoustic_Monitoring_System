"""
Software-Only Acoustic Edge Gate & ML Simulator CLI.

Demonstrates the full edge-computation saving mechanism without physical hardware:
WAV -> Edge Gate -> TRIGGER / IGNORE -> (Only if TRIGGER) -> ML Anomaly Inference.

Usage:
    python -m dsp.simulator path/to/audio.wav
    python -m dsp.simulator path/to/audio.wav --gate amrag --resource 0.5
    python -m dsp.simulator path/to/audio.wav --gate baseline_amplitude
"""

import argparse
import json
import sys
import time
from pathlib import Path
import numpy as np

from dsp.adaptive_gate import AdaptiveMultiBandResourceGate
from dsp.audio_loader import AudioLoader
from dsp.baseline_gates import (
    BaselineAdaptiveSNRGate,
    BaselineFixedAmplitudeGate,
    BaselineFixedBandGate,
)


def run_simulator(
    audio_path: str,
    gate_name: str = "amrag",
    resource_level: float = 1.0,
    plot_output: bool = False,
) -> int:
    """Runs simulation pipeline for a single audio file."""
    path = Path(audio_path)
    if not path.is_file():
        print(f"Error: Audio file not found: {audio_path}", file=sys.stderr)
        return 1

    loader = AudioLoader(target_sample_rate=16000)
    try:
        meta = loader.get_metadata(path)
    except Exception as e:
        print(f"Error reading audio metadata: {e}", file=sys.stderr)
        return 1

    print("\n" + "=" * 65)
    print("  ACOUSTIC MONITORING SYSTEM — SOFTWARE EDGE GATE SIMULATOR")
    print("=" * 65)
    print(f"Audio File     : {path.name}")
    print(f"Original SR    : {meta.sample_rate} Hz | Target SR: 16000 Hz")
    print(f"Channels       : {meta.channels} | Duration: {meta.duration_sec:.2f} s")
    print(f"Gate Engine    : {gate_name.upper()} (Resource Level: {resource_level})")
    print("-" * 65)

    # Initialize requested gate
    gate_name_clean = gate_name.lower().strip()
    if gate_name_clean in ("amrag", "adaptive", "proposed"):
        gate = AdaptiveMultiBandResourceGate(resource_level=resource_level)
    elif gate_name_clean in ("baseline_a", "amplitude", "fixed_amplitude"):
        gate = BaselineFixedAmplitudeGate(threshold=0.05, resource_level=resource_level)
    elif gate_name_clean in ("baseline_b", "band", "fixed_band"):
        gate = BaselineFixedBandGate(target_band_name="mid", threshold=0.02, resource_level=resource_level)
    elif gate_name_clean in ("baseline_c", "snr", "adaptive_snr"):
        gate = BaselineAdaptiveSNRGate(snr_threshold_db=6.0, resource_level=resource_level)
    else:
        print(f"Unknown gate type: {gate_name}. Choose from: amrag, amplitude, band, snr", file=sys.stderr)
        return 1

    start_gate_time = time.perf_counter()
    report = gate.process_audio(path)
    gate_duration_ms = (time.perf_counter() - start_gate_time) * 1000.0

    print(f"\n[DSP Stage] Processed {report.total_frames} frames in {gate_duration_ms:.2f} ms")
    print(f"Noise Floor    : {report.metadata.get('noise_floor', 'N/A')}")
    print(f"Peak SNR       : {report.metadata.get('snr_db', 'N/A')} dB")
    print(f"Max Gate Score : {report.max_trigger_score:.4f} (Threshold: {report.threshold:.4f})")
    print(f"Trigger Rate   : {report.trigger_rate * 100:.1f}% ({report.triggered_frames}/{report.total_frames} frames)")

    if report.overall_triggered:
        print("\n>>> GATE DECISION : [ TRIGGER ] <<<")
        print(f"    Events Detected: {len(report.events)}")
        for i, ev in enumerate(report.events):
            print(f"      Event #{i+1}: {ev['start_time']}s -> {ev['end_time']}s "
                  f"(dur: {ev['duration_sec']}s, peak_score: {ev['peak_score']})")

        print("\n[ML Stage] Gating condition satisfied -> FORWARDING TO ML PIPELINE")
        try:
            from ml.models.inference import MachineAnomalyInference
            infer_engine = MachineAnomalyInference()
            ml_start = time.perf_counter()
            pred = infer_engine.predict_file(str(path))
            ml_duration_ms = (time.perf_counter() - ml_start) * 1000.0

            print(f"    ML Latency     : {ml_duration_ms:.2f} ms")
            print(f"    Model Status   : {pred['status']}")
            print(f"    ML Diagnosis   : {pred.get('result', 'MODEL_UNAVAILABLE')}")
            print(f"    Anomaly Score  : {pred.get('anomaly_score', 'N/A')}")
            print(f"    Confidence     : {pred.get('confidence', 'N/A')}")
        except Exception as e:
            print(f"    ML Pipeline status: MODEL_UNAVAILABLE ({e})")
    else:
        print("\n>>> GATE DECISION : [ IGNORE ] <<<")
        print("    Signal classified as normal operational background.")
        print("    >>> ML NOT EXECUTED <<< (Expensive compute and network transmission saved)")

    print("\n" + "=" * 65 + "\n")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Acoustic Gate Simulator CLI")
    parser.add_argument("audio_path", type=str, help="Path to input WAV file")
    parser.add_argument("--gate", type=str, default="amrag", choices=["amrag", "amplitude", "band", "snr"],
                        help="Gate type to evaluate")
    parser.add_argument("--resource", type=float, default=1.0, help="Simulated edge resource level (0.1 to 1.0)")
    parser.add_argument("--plot", action="store_true", help="Generate analysis plot")
    args = parser.parse_args()

    sys.exit(run_simulator(
        audio_path=args.audio_path,
        gate_name=args.gate,
        resource_level=args.resource,
        plot_output=args.plot,
    ))
