"""
Mel Spectrogram Preprocessing Module.

Computes Mel Filter Bank and Log-Mel Spectrogram representations matching DCASE conventions.
Ensures 100% deterministic, identical preprocessing between training and inference.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal


@dataclass
class SpectrogramConfig:
    sample_rate: int = 16000
    n_fft: int = 1024
    hop_length: int = 512
    n_mels: int = 128
    f_min: float = 20.0
    f_max: float = 8000.0
    top_db: float = 80.0


def hz_to_mel(frequencies: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Converts frequency in Hz to Mel scale (HTK formula)."""
    return 2595.0 * np.log10(1.0 + np.asarray(frequencies) / 700.0)


def mel_to_hz(mels: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Converts frequency in Mel scale to Hz (HTK formula)."""
    return 700.0 * (10.0 ** (np.asarray(mels) / 2595.0) - 1.0)


def create_mel_filterbank(
    sample_rate: int = 16000,
    n_fft: int = 1024,
    n_mels: int = 128,
    f_min: float = 20.0,
    f_max: float = 8000.0,
) -> np.ndarray:
    """
    Constructs triangular Mel filterbank matrix of shape (n_mels, n_fft // 2 + 1).
    """
    f_max = min(f_max, sample_rate / 2.0)
    mel_min = hz_to_mel(f_min)
    mel_max = hz_to_mel(f_max)

    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = mel_to_hz(mel_points)

    fft_freqs = np.fft.rfftfreq(n_fft, d=1.0 / sample_rate)
    filterbank = np.zeros((n_mels, len(fft_freqs)), dtype=np.float32)

    for m in range(n_mels):
        f_left = hz_points[m]
        f_center = hz_points[m + 1]
        f_right = hz_points[m + 2]

        # Left slope
        left_mask = (fft_freqs >= f_left) & (fft_freqs <= f_center)
        if f_center > f_left:
            filterbank[m, left_mask] = (fft_freqs[left_mask] - f_left) / (f_center - f_left)

        # Right slope
        right_mask = (fft_freqs >= f_center) & (fft_freqs <= f_right)
        if f_right > f_center:
            filterbank[m, right_mask] = (f_right - fft_freqs[right_mask]) / (f_right - f_center)

        # Slaney area normalization
        enorm = 2.0 / (f_right - f_left) if f_right > f_left else 1.0
        filterbank[m] *= enorm

    return filterbank


class MelSpectrogramExtractor:
    """
    Deterministic Mel Spectrogram extractor.
    """

    def __init__(self, config: Optional[SpectrogramConfig] = None):
        self.config = config or SpectrogramConfig()
        self.mel_basis = create_mel_filterbank(
            sample_rate=self.config.sample_rate,
            n_fft=self.config.n_fft,
            n_mels=self.config.n_mels,
            f_min=self.config.f_min,
            f_max=self.config.f_max,
        )
        self.window = np.hanning(self.config.n_fft).astype(np.float32)

    def compute(self, audio: np.ndarray) -> np.ndarray:
        """
        Computes normalized log-mel spectrogram for 1D audio array.

        Returns:
            log_mel_spec: 2D array of shape (n_mels, n_frames).
        """
        audio = np.asarray(audio, dtype=np.float32).flatten()
        if len(audio) < self.config.n_fft:
            audio = np.pad(audio, (0, self.config.n_fft - len(audio)), mode="constant")

        # STFT computation using scipy
        _, _, Zxx = scipy.signal.stft(
            audio,
            fs=self.config.sample_rate,
            window=self.window,
            nperseg=self.config.n_fft,
            noverlap=self.config.n_fft - self.config.hop_length,
            boundary=None,
            padded=True,
        )

        power_spec = (np.abs(Zxx) ** 2).astype(np.float32)
        # Apply Mel filterbank: (n_mels, n_fft//2 + 1) @ (n_fft//2 + 1, n_frames) -> (n_mels, n_frames)
        mel_power = np.dot(self.mel_basis, power_spec)
        mel_power = np.maximum(mel_power, 1e-10)

        # Convert to dB scale
        mel_db = 10.0 * np.log10(mel_power)
        # Dynamic range clamping relative to peak
        max_db = np.max(mel_db)
        mel_db = np.maximum(mel_db, max_db - self.config.top_db)

        # Normalize to [0.0, 1.0] range
        normalized = (mel_db - (max_db - self.config.top_db)) / self.config.top_db
        return normalized.astype(np.float32)

    def save_image(self, mel_spec: np.ndarray, output_path: Union[str, Path]) -> str:
        """
        Saves a 2D Mel spectrogram visualization image to disk.
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(6, 4))
        img = ax.imshow(
            mel_spec,
            origin="lower",
            aspect="auto",
            cmap="magma",
            extent=[0, mel_spec.shape[1], 0, self.config.n_mels],
        )
        ax.set_title("Log-Mel Spectrogram (Triggered Audio)", fontsize=11)
        ax.set_xlabel("Time Frames", fontsize=9)
        ax.set_ylabel("Mel Frequency Bin", fontsize=9)
        fig.colorbar(img, ax=ax, orientation="vertical", label="Normalized dB")
        plt.tight_layout()
        plt.savefig(str(path), dpi=120)
        plt.close(fig)
        return str(path.resolve())
