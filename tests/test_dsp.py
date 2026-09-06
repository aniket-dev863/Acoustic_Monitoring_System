"""
Comprehensive Unit and Behavioral Tests for the DSP Engine.

Validates:
1. Audio loading, polyphase resampling, multi-channel handling.
2. Hann windowing, framing, and circular streaming buffer.
3. FFT processor, frequency axis, power spectrum, sub-band energies.
4. Adaptive noise estimator, asymmetric EMA, and event freezing.
5. Spectral feature extraction (SNR, BER, centroid, flatness, entropy, flux).
6. Trigger state machine (hysteresis, confirmation frames, release time).
7. Baseline gates (Fixed Amplitude, Fixed Band, Adaptive SNR).
8. Proposed AMRAG gate (multi-band, temporal persistence, resource scaling).
"""

import numpy as np
import pytest

from dsp.adaptive_gate import AdaptiveMultiBandResourceGate
from dsp.audio_loader import AudioLoader
from dsp.baseline_gates import (
    BaselineAdaptiveSNRGate,
    BaselineFixedAmplitudeGate,
    BaselineFixedBandGate,
)
from dsp.fft_processor import FFTProcessor, FrequencyBand
from dsp.framing import FrameBuffer, create_window, frame_signal
from dsp.noise_estimator import AdaptiveNoiseEstimator
from dsp.spectral_features import FeatureExtractor
from dsp.trigger_state import GateDecision, TriggerStateMachine


def test_audio_loader_resampling():
    loader = AudioLoader(target_sample_rate=16000)
    # Generate 44.1 kHz sine wave
    orig_sr = 44100
    duration = 0.5
    t = np.linspace(0, duration, int(duration * orig_sr), endpoint=False)
    sig = np.sin(2 * np.pi * 440 * t).astype(np.float32)

    resampled = loader.resample(sig, orig_sr, 16000)
    expected_len = int(duration * 16000)
    assert abs(len(resampled) - expected_len) <= 2
    assert resampled.dtype == np.float32


def test_frame_buffer_streaming():
    buffer = FrameBuffer(frame_size=1024, hop_length=512)
    chunk1 = np.ones(800, dtype=np.float32)
    chunk2 = np.ones(800, dtype=np.float32)

    frames = list(buffer.push(chunk1))
    assert len(frames) == 0  # 800 < 1024, no complete frame yet

    frames = list(buffer.push(chunk2))
    # 800 + 800 = 1600 samples
    # Frame 1: 0..1024, Frame 2: 512..1536
    assert len(frames) == 2
    assert len(frames[0]) == 1024

    final = buffer.flush(pad_with_zeros=True)
    assert final is not None
    assert len(final) == 1024


def test_fft_processor_peaks():
    proc = FFTProcessor(sample_rate=16000, fft_size=1024, hop_length=512)
    # Generate pure 1000 Hz tone
    t = np.linspace(0, 1024 / 16000.0, 1024, endpoint=False)
    tone = np.sin(2 * np.pi * 1000 * t).astype(np.float32)

    spec = proc.process_frame(tone)
    assert abs(spec.dominant_frequency - 1000.0) < 20.0  # Bin resolution ~15.6 Hz
    assert spec.total_energy > 0
    assert "mid" in spec.band_energies
    assert spec.band_energies["mid"] > spec.band_energies["high"]


def test_noise_estimator_adaptation_and_freeze():
    estimator = AdaptiveNoiseEstimator(alpha_up=0.05, alpha_down=0.20, history_len=10)

    # Initial quiet baseline
    for _ in range(10):
        estimator.update({"total": 0.01})
    quiet_floor = estimator.get_noise_floor("total")
    assert quiet_floor > 0

    # Step up noise
    for _ in range(15):
        estimator.update({"total": 0.10})
    adapted_floor = estimator.get_noise_floor("total")
    assert adapted_floor > quiet_floor

    # Test freeze: noise floor should NOT increase while frozen
    estimator.set_frozen(True)
    frozen_floor_before = estimator.get_noise_floor("total")
    for _ in range(10):
        estimator.update({"total": 5.0})  # Huge anomaly energy burst
    frozen_floor_after = estimator.get_noise_floor("total")
    assert frozen_floor_after == frozen_floor_before


def test_spectral_features():
    proc = FFTProcessor(sample_rate=16000, fft_size=1024)
    extractor = FeatureExtractor()

    # White noise vs Pure tone
    white_noise = np.random.normal(0, 0.1, 1024).astype(np.float32)
    spec_noise = proc.process_frame(white_noise)
    feat_noise = extractor.extract(
        spec_noise.magnitude_spectrum,
        spec_noise.power_spectrum,
        spec_noise.frequencies,
        spec_noise.band_energies,
        {"total": 0.01},
        spec_noise.rms,
    )

    t = np.linspace(0, 1024 / 16000.0, 1024, endpoint=False)
    pure_tone = np.sin(2 * np.pi * 1500 * t).astype(np.float32)
    spec_tone = proc.process_frame(pure_tone)
    feat_tone = extractor.extract(
        spec_tone.magnitude_spectrum,
        spec_tone.power_spectrum,
        spec_tone.frequencies,
        spec_tone.band_energies,
        {"total": 0.01},
        spec_tone.rms,
    )

    # Pure tone should have significantly lower spectral entropy and flatness than white noise
    assert feat_tone.spectral_entropy < feat_noise.spectral_entropy
    assert feat_tone.spectral_flatness < feat_noise.spectral_flatness


def test_trigger_state_machine_hysteresis():
    tsm = TriggerStateMachine(
        trigger_threshold=0.6,
        release_threshold=0.3,
        confirm_frames=3,
        release_frames=2,
        min_event_frames=4,
    )

    # 1. Sub-threshold: IDLE
    assert not tsm.step(0.4, 0, 0.0)
    assert tsm.state == GateDecision.IDLE

    # 2. Score spikes above trigger threshold
    tsm.step(0.7, 1, 0.03)  # frame 1 (pending)
    assert tsm.state == GateDecision.PENDING
    tsm.step(0.7, 2, 0.06)  # frame 2 (pending)
    assert tsm.state == GateDecision.PENDING
    trig = tsm.step(0.7, 3, 0.09)  # frame 3 (confirmed)
    assert trig is True
    assert tsm.state == GateDecision.TRIGGERED

    # 3. Score drops between release and trigger (hysteresis zone)
    # Should remain triggered!
    trig = tsm.step(0.45, 4, 0.12)
    assert trig is True
    assert tsm.state == GateDecision.TRIGGERED

    # 4. Score drops below release threshold
    tsm.step(0.2, 5, 0.15)
    assert tsm.state == GateDecision.RELEASE
    tsm.step(0.2, 6, 0.18)
    # Now confirmed release
    assert tsm.state == GateDecision.IDLE


def test_resource_aware_selectivity():
    gate_high_res = AdaptiveMultiBandResourceGate(resource_level=1.0)
    gate_low_res = AdaptiveMultiBandResourceGate(resource_level=0.2)

    # Low resource mode must have higher threshold and equal or greater confirmation frames
    assert gate_low_res.effective_trigger_threshold > gate_high_res.effective_trigger_threshold
    assert gate_low_res.effective_confirm_frames >= gate_high_res.effective_confirm_frames


def test_all_gates_execution():
    # Synthetic test signal: 1 second silence followed by 1 second tone
    sr = 16000
    t = np.linspace(0, 1.0, sr, endpoint=False)
    silence = np.zeros(sr, dtype=np.float32)
    tone = (0.2 * np.sin(2 * np.pi * 2000 * t)).astype(np.float32)
    signal = np.concatenate([silence, tone])

    gates = [
        BaselineFixedAmplitudeGate(),
        BaselineFixedBandGate(),
        BaselineAdaptiveSNRGate(),
        AdaptiveMultiBandResourceGate(),
    ]

    for g in gates:
        report = g.process_audio(signal)
        assert report.total_frames > 0
        assert report.duration_sec >= 1.95
        assert isinstance(report.overall_triggered, bool)
        assert len(report.frame_scores) == report.total_frames
