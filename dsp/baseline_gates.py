"""
Baseline Acoustic Gating Algorithms.

Implements three reference baseline gates for rigorous empirical comparison:
1. Baseline A — Fixed Amplitude (RMS / Energy threshold)
2. Baseline B — Fixed Frequency Band (Band Energy threshold)
3. Baseline C — Adaptive SNR (Running noise floor + SNR threshold)
"""

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np

from dsp.audio_loader import AudioLoader
from dsp.fft_processor import FFTProcessor, FrequencyBand
from dsp.noise_estimator import AdaptiveNoiseEstimator
from dsp.spectral_features import FeatureExtractor
from dsp.trigger_state import EventInterval, TriggerStateMachine


@dataclass
class GateResult:
    """Result of evaluating a gate on a single frame."""
    triggered: bool
    trigger_score: float
    threshold: float
    snr_db: float
    noise_floor: float
    dominant_frequency: float
    band_energy_ratio: float
    spectral_entropy: float
    start_time: float
    end_time: float
    resource_level: float
    gate_type: str
    gate_version: str
    frame_index: int = 0
    raw_energy: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class GateExecutionReport:
    """Aggregated report of evaluating a gate over an entire audio recording."""
    audio_path: str
    duration_sec: float
    total_frames: int
    triggered_frames: int
    trigger_rate: float
    overall_triggered: bool
    max_trigger_score: float
    mean_trigger_score: float
    threshold: float
    events: List[Dict[str, Any]]
    gate_type: str
    gate_version: str
    resource_level: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    frame_scores: List[float] = field(default_factory=list)
    frame_timestamps: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class BaseAcousticGate(ABC):
    """Abstract base class for all acoustic gates."""

    def __init__(
        self,
        gate_type: str,
        gate_version: str,
        sample_rate: int = 16000,
        fft_size: int = 1024,
        hop_length: int = 512,
        resource_level: float = 1.0,
    ):
        self.gate_type = gate_type
        self.gate_version = gate_version
        self.sample_rate = int(sample_rate)
        self.fft_size = int(fft_size)
        self.hop_length = int(hop_length)
        self.resource_level = float(np.clip(resource_level, 0.1, 1.0))

        self.fft_processor = FFTProcessor(
            sample_rate=self.sample_rate,
            fft_size=self.fft_size,
            hop_length=self.hop_length,
        )
        self.feature_extractor = FeatureExtractor()
        self.loader = AudioLoader(target_sample_rate=self.sample_rate)

    @abstractmethod
    def reset(self) -> None:
        """Resets gate internal state for a new audio stream."""
        pass

    @abstractmethod
    def process_frame(
        self,
        frame_samples: np.ndarray,
        frame_index: int = 0,
        timestamp: float = 0.0,
    ) -> GateResult:
        """Processes a single audio frame (edge streaming boundary)."""
        pass

    def process_audio(
        self,
        audio_or_path: Union[str, Path, np.ndarray],
        channel_index: Optional[int] = None,
    ) -> GateExecutionReport:
        """
        Processes an entire audio clip and returns summary metrics and event intervals.
        """
        self.reset()

        if isinstance(audio_or_path, (str, Path)):
            path_str = str(audio_or_path)
            samples, sr = self.loader.load(path_str, mono=True, channel_index=channel_index)
        else:
            path_str = "in-memory"
            samples = np.asarray(audio_or_path, dtype=np.float32)
            sr = self.sample_rate

        duration_sec = len(samples) / float(sr)
        frame_results: List[GateResult] = []
        frame_scores: List[float] = []
        frame_timestamps: List[float] = []

        # Stream frames
        idx = 0
        frame_idx = 0
        num_samples = len(samples)

        while idx + self.fft_size <= num_samples:
            frame = samples[idx : idx + self.fft_size]
            ts = idx / float(sr)
            result = self.process_frame(frame, frame_index=frame_idx, timestamp=ts)
            frame_results.append(result)
            frame_scores.append(result.trigger_score)
            frame_timestamps.append(ts)
            idx += self.hop_length
            frame_idx += 1

        total_frames = len(frame_results)
        triggered_frames = sum(1 for r in frame_results if r.triggered)
        overall_triggered = triggered_frames > 0
        trigger_rate = triggered_frames / total_frames if total_frames > 0 else 0.0
        max_score = float(max(frame_scores)) if frame_scores else 0.0
        mean_score = float(np.mean(frame_scores)) if frame_scores else 0.0

        # Collect event intervals
        events = self._get_events(total_frames, duration_sec)

        # Base representative metadata from the peak or final frame
        rep_result = max(frame_results, key=lambda r: r.trigger_score) if frame_results else None
        rep_meta = rep_result.to_dict() if rep_result else {}

        return GateExecutionReport(
            audio_path=path_str,
            duration_sec=duration_sec,
            total_frames=total_frames,
            triggered_frames=triggered_frames,
            trigger_rate=trigger_rate,
            overall_triggered=overall_triggered,
            max_trigger_score=max_score,
            mean_trigger_score=mean_score,
            threshold=float(rep_result.threshold if rep_result else 0.0),
            events=events,
            gate_type=self.gate_type,
            gate_version=self.gate_version,
            resource_level=self.resource_level,
            metadata=rep_meta,
            frame_scores=frame_scores,
            frame_timestamps=frame_timestamps,
        )

    def _get_events(self, total_frames: int, duration_sec: float) -> List[Dict[str, Any]]:
        """Hook for gates with stateful temporal hysteresis to return structured intervals."""
        return []


# =========================================================================
# Baseline A: Fixed Amplitude Gate
# =========================================================================

class BaselineFixedAmplitudeGate(BaseAcousticGate):
    """
    Baseline A: Fixed RMS/Energy threshold.
    Triggers whenever the time-domain RMS or total energy exceeds a static threshold.
    """

    def __init__(
        self,
        threshold: float = 0.05,  # Fixed RMS threshold
        sample_rate: int = 16000,
        fft_size: int = 1024,
        hop_length: int = 512,
        resource_level: float = 1.0,
    ):
        super().__init__(
            gate_type="baseline_fixed_amplitude",
            gate_version="Baseline-A-v1",
            sample_rate=sample_rate,
            fft_size=fft_size,
            hop_length=hop_length,
            resource_level=resource_level,
        )
        self.threshold = float(threshold)

    def reset(self) -> None:
        pass

    def process_frame(
        self,
        frame_samples: np.ndarray,
        frame_index: int = 0,
        timestamp: float = 0.0,
    ) -> GateResult:
        rms = float(np.sqrt(np.mean(frame_samples**2) + 1e-12))
        triggered = bool(rms >= self.threshold)

        # Compute basic spectrum for standardized metadata reporting
        spectrum = self.fft_processor.process_frame(frame_samples, frame_index, timestamp)
        features = self.feature_extractor.extract(
            magnitude_spectrum=spectrum.magnitude_spectrum,
            power_spectrum=spectrum.power_spectrum,
            frequencies=spectrum.frequencies,
            band_energies=spectrum.band_energies,
            noise_floors={"total": self.threshold**2},
            rms=rms,
            frame_index=frame_index,
            timestamp=timestamp,
        )

        return GateResult(
            triggered=triggered,
            trigger_score=rms,
            threshold=self.threshold,
            snr_db=features.snr_db,
            noise_floor=self.threshold,
            dominant_frequency=features.dominant_frequency,
            band_energy_ratio=features.band_energy_ratios.get("mid", 0.0),
            spectral_entropy=features.spectral_entropy,
            start_time=timestamp,
            end_time=timestamp + (self.fft_size / self.sample_rate),
            resource_level=self.resource_level,
            gate_type=self.gate_type,
            gate_version=self.gate_version,
            frame_index=frame_index,
            raw_energy=spectrum.total_energy,
        )


# =========================================================================
# Baseline B: Fixed Frequency Band Gate
# =========================================================================

class BaselineFixedBandGate(BaseAcousticGate):
    """
    Baseline B: Fixed frequency band energy threshold.
    Performs FFT, computes power in a preselected band (e.g. 800-3000 Hz),
    and triggers if that band energy exceeds a static threshold.
    """

    def __init__(
        self,
        target_band_name: str = "mid",
        threshold: float = 0.01,
        sample_rate: int = 16000,
        fft_size: int = 1024,
        hop_length: int = 512,
        resource_level: float = 1.0,
    ):
        super().__init__(
            gate_type="baseline_fixed_band",
            gate_version="Baseline-B-v1",
            sample_rate=sample_rate,
            fft_size=fft_size,
            hop_length=hop_length,
            resource_level=resource_level,
        )
        self.target_band_name = target_band_name
        self.threshold = float(threshold)

    def reset(self) -> None:
        pass

    def process_frame(
        self,
        frame_samples: np.ndarray,
        frame_index: int = 0,
        timestamp: float = 0.0,
    ) -> GateResult:
        spectrum = self.fft_processor.process_frame(frame_samples, frame_index, timestamp)
        band_energy = spectrum.band_energies.get(self.target_band_name, 0.0)
        triggered = bool(band_energy >= self.threshold)

        features = self.feature_extractor.extract(
            magnitude_spectrum=spectrum.magnitude_spectrum,
            power_spectrum=spectrum.power_spectrum,
            frequencies=spectrum.frequencies,
            band_energies=spectrum.band_energies,
            noise_floors={"total": self.threshold},
            rms=spectrum.rms,
            frame_index=frame_index,
            timestamp=timestamp,
        )

        return GateResult(
            triggered=triggered,
            trigger_score=band_energy,
            threshold=self.threshold,
            snr_db=features.snr_db,
            noise_floor=self.threshold,
            dominant_frequency=features.dominant_frequency,
            band_energy_ratio=features.band_energy_ratios.get(self.target_band_name, 0.0),
            spectral_entropy=features.spectral_entropy,
            start_time=timestamp,
            end_time=timestamp + (self.fft_size / self.sample_rate),
            resource_level=self.resource_level,
            gate_type=self.gate_type,
            gate_version=self.gate_version,
            frame_index=frame_index,
            raw_energy=spectrum.total_energy,
        )


# =========================================================================
# Baseline C: Adaptive SNR Gate
# =========================================================================

class BaselineAdaptiveSNRGate(BaseAcousticGate):
    """
    Baseline C: Adaptive SNR threshold.
    Tracks a running background noise floor across full-band energy
    and triggers whenever full-band SNR (dB) exceeds a fixed SNR threshold.
    """

    def __init__(
        self,
        snr_threshold_db: float = 6.0,
        sample_rate: int = 16000,
        fft_size: int = 1024,
        hop_length: int = 512,
        resource_level: float = 1.0,
    ):
        super().__init__(
            gate_type="baseline_adaptive_snr",
            gate_version="Baseline-C-v1",
            sample_rate=sample_rate,
            fft_size=fft_size,
            hop_length=hop_length,
            resource_level=resource_level,
        )
        self.snr_threshold_db = float(snr_threshold_db)
        self.noise_estimator = AdaptiveNoiseEstimator(alpha_up=0.02, alpha_down=0.08)

    def reset(self) -> None:
        self.noise_estimator.reset()

    def process_frame(
        self,
        frame_samples: np.ndarray,
        frame_index: int = 0,
        timestamp: float = 0.0,
    ) -> GateResult:
        spectrum = self.fft_processor.process_frame(frame_samples, frame_index, timestamp)
        energies = {"total": spectrum.total_energy}

        # Update running noise floor
        floors = self.noise_estimator.update(energies)
        current_noise = floors.get("total", 1e-6)

        _, snr_db = self.noise_estimator.calculate_snr(spectrum.total_energy, "total")
        triggered = bool(snr_db >= self.snr_threshold_db)

        features = self.feature_extractor.extract(
            magnitude_spectrum=spectrum.magnitude_spectrum,
            power_spectrum=spectrum.power_spectrum,
            frequencies=spectrum.frequencies,
            band_energies=spectrum.band_energies,
            noise_floors=floors,
            rms=spectrum.rms,
            frame_index=frame_index,
            timestamp=timestamp,
        )

        return GateResult(
            triggered=triggered,
            trigger_score=snr_db,
            threshold=self.snr_threshold_db,
            snr_db=snr_db,
            noise_floor=current_noise,
            dominant_frequency=features.dominant_frequency,
            band_energy_ratio=features.band_energy_ratios.get("mid", 0.0),
            spectral_entropy=features.spectral_entropy,
            start_time=timestamp,
            end_time=timestamp + (self.fft_size / self.sample_rate),
            resource_level=self.resource_level,
            gate_type=self.gate_type,
            gate_version=self.gate_version,
            frame_index=frame_index,
            raw_energy=spectrum.total_energy,
        )
