"""
Spectral Features Extraction Module.

Calculates lightweight, explainable spectral features for acoustic anomaly gating:
- Full-band and multi-band SNR
- Band Energy Ratio (BER)
- Dominant Frequency
- Spectral Centroid
- Spectral Flatness (Wiener entropy)
- Spectral Entropy (Shannon entropy)
- Spectral Flux (consecutive frame variation)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np


@dataclass
class SpectralFeatures:
    """Extracted lightweight spectral features for a single frame."""
    frame_index: int
    timestamp: float
    total_energy: float
    rms: float
    snr_linear: float
    snr_db: float
    band_snrs_db: Dict[str, float] = field(default_factory=dict)
    band_energy_ratios: Dict[str, float] = field(default_factory=dict)
    dominant_frequency: float = 0.0
    spectral_centroid: float = 0.0
    spectral_flatness: float = 0.0
    spectral_entropy: float = 0.0
    spectral_flux: float = 0.0


class FeatureExtractor:
    """
    Configurable lightweight spectral feature extractor.
    Enables selective activation of features for ablation experiments.
    """

    def __init__(
        self,
        enable_snr: bool = True,
        enable_ber: bool = True,
        enable_dominant_freq: bool = True,
        enable_centroid: bool = True,
        enable_flatness: bool = True,
        enable_entropy: bool = True,
        enable_flux: bool = True,
        eps: float = 1e-12,
    ):
        self.enable_snr = enable_snr
        self.enable_ber = enable_ber
        self.enable_dominant_freq = enable_dominant_freq
        self.enable_centroid = enable_centroid
        self.enable_flatness = enable_flatness
        self.enable_entropy = enable_entropy
        self.enable_flux = enable_flux
        self.eps = eps

        self.prev_magnitude_spectrum: Optional[np.ndarray] = None

    def reset(self) -> None:
        """Resets state (e.g. previous frame memory for flux)."""
        self.prev_magnitude_spectrum = None

    def extract(
        self,
        magnitude_spectrum: np.ndarray,
        power_spectrum: np.ndarray,
        frequencies: np.ndarray,
        band_energies: Dict[str, float],
        noise_floors: Dict[str, float],
        rms: float,
        frame_index: int = 0,
        timestamp: float = 0.0,
    ) -> SpectralFeatures:
        """
        Extracts configured spectral features for the current frame.
        """
        total_energy = float(np.sum(power_spectrum))

        # 1. Full-band SNR and per-band SNRs
        snr_linear = 1.0
        snr_db = 0.0
        band_snrs_db: Dict[str, float] = {}

        if self.enable_snr:
            noise_total = noise_floors.get("total", self.eps)
            snr_linear = max(total_energy / max(noise_total, self.eps), self.eps)
            snr_db = float(10.0 * np.log10(snr_linear))

            for band_name, band_e in band_energies.items():
                band_noise = noise_floors.get(band_name, self.eps)
                ratio = max(band_e / max(band_noise, self.eps), self.eps)
                band_snrs_db[band_name] = float(10.0 * np.log10(ratio))

        # 2. Band Energy Ratio (BER)
        band_energy_ratios: Dict[str, float] = {}
        if self.enable_ber and total_energy > self.eps:
            for band_name, band_e in band_energies.items():
                band_energy_ratios[band_name] = float(band_e / total_energy)

        # 3. Dominant Frequency
        dominant_freq = 0.0
        if self.enable_dominant_freq:
            peak_bin = int(np.argmax(magnitude_spectrum))
            dominant_freq = float(frequencies[peak_bin])

        # 4. Spectral Centroid
        spectral_centroid = 0.0
        if self.enable_centroid and total_energy > self.eps:
            # Centroid = sum(f * |X(f)|) / sum(|X(f)|)
            mag_sum = float(np.sum(magnitude_spectrum))
            if mag_sum > self.eps:
                spectral_centroid = float(np.sum(frequencies * magnitude_spectrum) / mag_sum)

        # 5. Spectral Flatness (Wiener entropy = geometric mean / arithmetic mean)
        spectral_flatness = 0.0
        if self.enable_flatness and total_energy > self.eps:
            # Add eps to avoid log(0)
            log_p = np.log(power_spectrum + self.eps)
            geom_mean = np.exp(np.mean(log_p))
            arith_mean = np.mean(power_spectrum) + self.eps
            spectral_flatness = float(np.clip(geom_mean / arith_mean, 0.0, 1.0))

        # 6. Spectral Entropy (Shannon entropy of normalized power distribution)
        spectral_entropy = 0.0
        if self.enable_entropy and total_energy > self.eps:
            # Probability distribution p_i = P_i / sum(P)
            prob_dist = (power_spectrum + self.eps) / (total_energy + self.eps * len(power_spectrum))
            raw_entropy = -np.sum(prob_dist * np.log2(prob_dist + self.eps))
            max_entropy = np.log2(len(prob_dist))  # Uniform distribution normalization
            spectral_entropy = float(np.clip(raw_entropy / max_entropy, 0.0, 1.0))

        # 7. Spectral Flux (L2 norm of difference between normalized consecutive magnitude spectra)
        spectral_flux = 0.0
        if self.enable_flux:
            mag_norm = magnitude_spectrum / (np.linalg.norm(magnitude_spectrum) + self.eps)
            if self.prev_magnitude_spectrum is not None:
                diff = mag_norm - self.prev_magnitude_spectrum
                spectral_flux = float(np.sqrt(np.sum(diff**2)))
            self.prev_magnitude_spectrum = mag_norm.copy()

        return SpectralFeatures(
            frame_index=frame_index,
            timestamp=timestamp,
            total_energy=total_energy,
            rms=rms,
            snr_linear=snr_linear,
            snr_db=snr_db,
            band_snrs_db=band_snrs_db,
            band_energy_ratios=band_energy_ratios,
            dominant_frequency=dominant_freq,
            spectral_centroid=spectral_centroid,
            spectral_flatness=spectral_flatness,
            spectral_entropy=spectral_entropy,
            spectral_flux=spectral_flux,
        )
