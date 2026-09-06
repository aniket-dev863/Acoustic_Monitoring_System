"""
Proposed Method: Adaptive Multi-Band Resource-Aware Acoustic Gate (AMRAG).

Combines:
1. Multi-band spectral decomposition (low, mid, high sub-bands)
2. Adaptive noise floor estimation per band with freeze mechanism
3. Lightweight acoustic relevance scoring (multi-band SNR, BER, flux, entropy)
4. Temporal persistence and hysteresis state machine
5. Resource-aware selectivity modulation (battery/compute budget scaling)
"""

from typing import Any, Dict, List, Optional, Union
import numpy as np

from dsp.baseline_gates import BaseAcousticGate, GateExecutionReport, GateResult
from dsp.fft_processor import FrequencyBand
from dsp.noise_estimator import AdaptiveNoiseEstimator
from dsp.spectral_features import FeatureExtractor
from dsp.trigger_state import EventInterval, TriggerStateMachine


class AdaptiveMultiBandResourceGate(BaseAcousticGate):
    """
    Adaptive Multi-Band Resource-Aware Acoustic Gate (AMRAG-v1).

    Hypothesis:
    By evaluating SNR across distinct acoustic sub-bands and fusing spectral
    flux/entropy with adaptive noise tracking and resource-modulated selectivity,
    we can filter out normal stationary operating noise and ambient fluctuations,
    selectively forwarding only genuinely anomalous acoustic events to ML.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        fft_size: int = 1024,
        hop_length: int = 512,
        base_trigger_threshold: float = 0.55,
        base_release_threshold: float = 0.35,
        base_confirm_frames: int = 2,
        base_release_frames: int = 4,
        min_event_frames: int = 5,
        resource_level: float = 1.0,
        # Ablation / modular feature flags
        enable_multi_band: bool = True,
        enable_spectral_features: bool = True,
        enable_temporal_persistence: bool = True,
        enable_resource_adaptation: bool = True,
        enable_noise_adaptation: bool = True,
        # Configurable weights for acoustic relevance score
        weight_snr: float = 0.50,
        weight_ber: float = 0.20,
        weight_flux: float = 0.15,
        weight_entropy: float = 0.15,
        frequency_bands: Optional[List[FrequencyBand]] = None,
    ):
        super().__init__(
            gate_type="adaptive_multiband_resource_gate",
            gate_version="AMRAG-v1",
            sample_rate=sample_rate,
            fft_size=fft_size,
            hop_length=hop_length,
            resource_level=resource_level,
        )

        self.base_trigger_threshold = float(base_trigger_threshold)
        self.base_release_threshold = float(base_release_threshold)
        self.base_confirm_frames = int(base_confirm_frames)
        self.base_release_frames = int(base_release_frames)
        self.min_event_frames = int(min_event_frames)

        # Feature flags for ablation study
        self.enable_multi_band = enable_multi_band
        self.enable_spectral_features = enable_spectral_features
        self.enable_temporal_persistence = enable_temporal_persistence
        self.enable_resource_adaptation = enable_resource_adaptation
        self.enable_noise_adaptation = enable_noise_adaptation

        # Score weights
        self.weight_snr = weight_snr
        self.weight_ber = weight_ber
        self.weight_flux = weight_flux
        self.weight_entropy = weight_entropy

        # Custom or default sub-bands
        if frequency_bands:
            self.fft_processor.bands = frequency_bands
            self.fft_processor._precompute_band_indices()

        # Noise tracking
        self.noise_estimator = AdaptiveNoiseEstimator(
            alpha_up=0.015,
            alpha_down=0.060,
            history_len=35,
            percentile=25.0,
        )

        # Temporal state machine
        self.state_machine = TriggerStateMachine(
            trigger_threshold=self.effective_trigger_threshold,
            release_threshold=self.effective_release_threshold,
            confirm_frames=self.effective_confirm_frames,
            release_frames=self.base_release_frames,
            min_event_frames=self.min_event_frames,
        )

        self.active_event_start_time: Optional[float] = None
        self.active_event_end_time: Optional[float] = None

    @property
    def effective_trigger_threshold(self) -> float:
        """
        Dynamically adjusts sensitivity based on available edge compute/battery resources.
        Lower resources -> higher selectivity (raised threshold).
        """
        if not self.enable_resource_adaptation:
            return self.base_trigger_threshold
        # Resource penalty: at resource_level = 0.2, threshold increases by ~0.24
        resource_penalty = (1.0 - self.resource_level) * 0.30
        return float(np.clip(self.base_trigger_threshold + resource_penalty, 0.1, 0.95))

    @property
    def effective_release_threshold(self) -> float:
        """Dynamically scaled release threshold maintaining hysteresis gap."""
        return float(self.effective_trigger_threshold * 0.65)

    @property
    def effective_confirm_frames(self) -> int:
        """
        Under tight resource constraints, require more confirmation frames
        to avoid frivolous wakes.
        """
        if not self.enable_temporal_persistence:
            return 1
        if not self.enable_resource_adaptation:
            return self.base_confirm_frames
        extra = int(round((1.0 - self.resource_level) * 2.0))
        return int(max(1, self.base_confirm_frames + extra))

    def set_resource_level(self, level: float) -> None:
        """Dynamically updates edge node resource availability."""
        self.resource_level = float(np.clip(level, 0.1, 1.0))
        self.state_machine.update_thresholds(
            trigger_threshold=self.effective_trigger_threshold,
            release_threshold=self.effective_release_threshold,
        )
        self.state_machine.confirm_frames = self.effective_confirm_frames

    def reset(self) -> None:
        """Resets gate state for a fresh audio stream."""
        self.noise_estimator.reset()
        self.feature_extractor.reset()
        self.state_machine.reset()
        self.active_event_start_time = None
        self.active_event_end_time = None

    def compute_acoustic_relevance_score(
        self,
        features,
        band_energies: Dict[str, float],
        noise_floors: Dict[str, float],
    ) -> float:
        """
        Calculates a composite acoustic relevance score [0.0, 1.0].
        Combines multi-band SNR, spectral flux, band energy imbalance, and entropy.
        """
        # 1. Multi-band vs single-band SNR score
        if self.enable_multi_band and features.band_snrs_db:
            # Maximum SNR across any individual sub-band
            max_band_snr = max(features.band_snrs_db.values())
            # Map 0 dB -> 0.0, 20 dB -> 1.0
            snr_score = float(np.clip(max_band_snr / 20.0, 0.0, 1.0))
        else:
            snr_score = float(np.clip(features.snr_db / 20.0, 0.0, 1.0))

        if not self.enable_spectral_features:
            return snr_score

        # 2. Band Energy Ratio deviation (energy concentration in high/mid bands)
        # Bearing faults & friction manifest in elevated mid/high ratios
        high_ber = features.band_energy_ratios.get("high", 0.0)
        mid_ber = features.band_energy_ratios.get("mid", 0.0)
        ber_score = float(np.clip((high_ber * 1.5 + mid_ber * 0.8), 0.0, 1.0))

        # 3. Spectral Flux score (abrupt spectral shifts)
        flux_score = float(np.clip(features.spectral_flux * 2.0, 0.0, 1.0))

        # 4. Spectral Tonal/Entropy score (low entropy = discrete tonal fault, e.g. squeal/meshing)
        # Invert entropy so prominent peaks produce high score
        tonal_score = float(np.clip(1.0 - features.spectral_entropy, 0.0, 1.0))

        # Weighted combination
        composite_score = (
            self.weight_snr * snr_score
            + self.weight_ber * ber_score
            + self.weight_flux * flux_score
            + self.weight_entropy * tonal_score
        )

        return float(np.clip(composite_score, 0.0, 1.0))

    def process_frame(
        self,
        frame_samples: np.ndarray,
        frame_index: int = 0,
        timestamp: float = 0.0,
    ) -> GateResult:
        """
        Processes a single audio frame with multi-band analysis, adaptive noise tracking,
        acoustic relevance scoring, temporal persistence, and resource modulation.
        """
        # 1. FFT Analysis & Sub-band energy
        spectrum = self.fft_processor.process_frame(frame_samples, frame_index, timestamp)

        # 2. Noise floor tracking (multi-band + total)
        energies = {"total": spectrum.total_energy}
        energies.update(spectrum.band_energies)

        if self.enable_noise_adaptation:
            floors = self.noise_estimator.update(energies)
        else:
            # Fixed baseline noise floor
            floors = {k: 1e-4 for k in energies.keys()}

        # 3. Lightweight spectral features
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

        # 4. Composite acoustic relevance score
        relevance_score = self.compute_acoustic_relevance_score(
            features=features,
            band_energies=spectrum.band_energies,
            noise_floors=floors,
        )

        # 5. Temporal persistence and hysteresis
        thresh = self.effective_trigger_threshold
        if self.enable_temporal_persistence:
            triggered = self.state_machine.step(relevance_score, frame_index, timestamp)
        else:
            # Immediate frame-level thresholding without hysteresis or confirmation
            triggered = bool(relevance_score >= thresh)

        # 6. Event-aware noise model freezing
        # Pause background adaptation while an acoustic event is confirmed active
        if self.enable_noise_adaptation:
            self.noise_estimator.set_frozen(triggered)

        # Track event time bounds
        if triggered:
            if self.active_event_start_time is None:
                self.active_event_start_time = timestamp
            self.active_event_end_time = timestamp + (self.fft_size / self.sample_rate)
            start_t = self.active_event_start_time
            end_t = self.active_event_end_time
        else:
            start_t = timestamp
            end_t = timestamp + (self.fft_size / self.sample_rate)
            self.active_event_start_time = None
            self.active_event_end_time = None

        return GateResult(
            triggered=triggered,
            trigger_score=round(relevance_score, 4),
            threshold=round(thresh, 4),
            snr_db=round(features.snr_db, 2),
            noise_floor=round(floors.get("total", 1e-6), 6),
            dominant_frequency=round(features.dominant_frequency, 1),
            band_energy_ratio=round(features.band_energy_ratios.get("mid", 0.0), 4),
            spectral_entropy=round(features.spectral_entropy, 4),
            start_time=round(start_t, 3),
            end_time=round(end_t, 3),
            resource_level=round(self.resource_level, 2),
            gate_type=self.gate_type,
            gate_version=self.gate_version,
            frame_index=frame_index,
            raw_energy=spectrum.total_energy,
        )

    def _get_events(self, total_frames: int, duration_sec: float) -> List[Dict[str, Any]]:
        """Extracts finalized event intervals from the temporal state machine."""
        if self.enable_temporal_persistence:
            intervals = self.state_machine.finalize(total_frames, duration_sec)
            return [
                {
                    "start_frame": iv.start_frame,
                    "end_frame": iv.end_frame,
                    "start_time": round(iv.start_time, 3),
                    "end_time": round(iv.end_time, 3),
                    "duration_sec": round(iv.end_time - iv.start_time, 3),
                    "peak_score": round(iv.peak_score, 4),
                    "mean_score": round(iv.mean_score, 4),
                }
                for iv in intervals
            ]
        return []
