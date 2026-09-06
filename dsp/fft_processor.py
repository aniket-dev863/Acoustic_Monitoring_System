"""
Real FFT and STFT Processing Engine.

Performs efficient single-frame and batch spectral analysis using NumPy/SciPy.
Calculates magnitude spectra, power spectra, total energy, dominant frequency,
and configurable frequency-subband energies.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from dsp.framing import create_window, frame_signal


@dataclass
class FrequencyBand:
    """Definition of a frequency sub-band."""
    name: str
    f_min: float
    f_max: float


@dataclass
class FrameSpectrum:
    """Spectral analysis result for a single frame."""
    frame_index: int
    timestamp: float
    frequencies: np.ndarray          # Frequency bin centers in Hz
    magnitude_spectrum: np.ndarray   # Linear magnitude
    power_spectrum: np.ndarray       # Power (|X|^2 / N)
    total_energy: float              # Sum of power spectrum
    rms: float                       # Time-domain RMS
    dominant_frequency: float        # Frequency with peak magnitude in Hz
    band_energies: Dict[str, float]  # Energy in each configured band


# Default standard acoustic sub-bands (industrial machine monitoring)
DEFAULT_FREQUENCY_BANDS = [
    FrequencyBand("low", 20.0, 800.0),       # Unbalance, structural looseness, sub-harmonics
    FrequencyBand("mid", 800.0, 3000.0),     # Blade pass, gear meshing, bearing defect fundamentals
    FrequencyBand("high", 3000.0, 8000.0),   # Friction, valve hissing, bearing impact resonance
]


class FFTProcessor:
    """
    Configurable FFT/STFT Engine.
    Compatible with both single-frame streaming execution and batch processing.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        fft_size: int = 1024,
        hop_length: int = 512,
        window: str = "hann",
        min_frequency: float = 20.0,
        max_frequency: Optional[float] = None,
        frequency_bands: Optional[List[FrequencyBand]] = None,
    ):
        self.sample_rate = int(sample_rate)
        self.fft_size = int(fft_size)
        self.hop_length = int(hop_length)
        self.window_type = window
        self.min_frequency = float(min_frequency)
        self.max_frequency = float(max_frequency) if max_frequency is not None else float(sample_rate / 2)

        # Precompute window
        self.window = create_window(self.window_type, self.fft_size)

        # Precompute frequency axis for rfft
        self.frequencies = np.fft.rfftfreq(self.fft_size, d=1.0 / self.sample_rate)
        self.num_bins = len(self.frequencies)

        # Configurable frequency sub-bands
        self.bands = frequency_bands if frequency_bands is not None else list(DEFAULT_FREQUENCY_BANDS)
        self.band_bin_indices: Dict[str, Tuple[int, int]] = {}
        self._precompute_band_indices()

    def _precompute_band_indices(self) -> None:
        """Finds start and end FFT bin indices for each defined band."""
        for band in self.bands:
            f_min = max(band.f_min, 0.0)
            f_max = min(band.f_max, self.sample_rate / 2.0)
            idx_start = np.searchsorted(self.frequencies, f_min, side="left")
            idx_end = np.searchsorted(self.frequencies, f_max, side="right")
            idx_end = max(idx_start + 1, idx_end)
            self.band_bin_indices[band.name] = (idx_start, idx_end)

    def process_frame(
        self,
        frame_samples: np.ndarray,
        frame_index: int = 0,
        timestamp: Optional[float] = None,
    ) -> FrameSpectrum:
        """
        Processes a single audio frame of length `fft_size`.
        Streaming edge boundary interface.
        """
        if len(frame_samples) != self.fft_size:
            # Pad or truncate to fft_size
            if len(frame_samples) < self.fft_size:
                frame_samples = np.pad(frame_samples, (0, self.fft_size - len(frame_samples)), mode="constant")
            else:
                frame_samples = frame_samples[: self.fft_size]

        frame_samples = np.asarray(frame_samples, dtype=np.float32)

        # Time-domain RMS
        rms = float(np.sqrt(np.mean(frame_samples**2) + 1e-12))

        # Windowing
        windowed = frame_samples * self.window

        # Real FFT
        complex_spectrum = np.fft.rfft(windowed, n=self.fft_size)
        magnitude_spectrum = np.abs(complex_spectrum) / (self.fft_size / 2.0)
        power_spectrum = (magnitude_spectrum**2)

        total_energy = float(np.sum(power_spectrum))

        # Dominant frequency (search within [min_frequency, max_frequency])
        valid_mask = (self.frequencies >= self.min_frequency) & (self.frequencies <= self.max_frequency)
        if np.any(valid_mask):
            sub_mag = magnitude_spectrum[valid_mask]
            sub_freqs = self.frequencies[valid_mask]
            peak_idx = int(np.argmax(sub_mag))
            dominant_frequency = float(sub_freqs[peak_idx])
        else:
            dominant_frequency = 0.0

        # Sub-band energies
        band_energies: Dict[str, float] = {}
        for band_name, (start_idx, end_idx) in self.band_bin_indices.items():
            band_energy = float(np.sum(power_spectrum[start_idx:end_idx]))
            band_energies[band_name] = band_energy

        ts = timestamp if timestamp is not None else float(frame_index * self.hop_length / self.sample_rate)

        return FrameSpectrum(
            frame_index=frame_index,
            timestamp=ts,
            frequencies=self.frequencies,
            magnitude_spectrum=magnitude_spectrum,
            power_spectrum=power_spectrum,
            total_energy=total_energy,
            rms=rms,
            dominant_frequency=dominant_frequency,
            band_energies=band_energies,
        )

    def process_signal(self, signal: np.ndarray) -> List[FrameSpectrum]:
        """
        Batch-processes an entire audio array, returning a list of FrameSpectrum.
        """
        raw_frames, _ = frame_signal(
            signal,
            frame_size=self.fft_size,
            hop_length=self.hop_length,
            window_type=None,  # windowing handled inside process_frame
            pad=True,
        )

        results: List[FrameSpectrum] = []
        for i, frame in enumerate(raw_frames):
            ts = float(i * self.hop_length / self.sample_rate)
            spec = self.process_frame(frame, frame_index=i, timestamp=ts)
            results.append(spec)

        return results
