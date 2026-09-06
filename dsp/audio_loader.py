"""
Audio Loading and Format Normalization Module.

Supports PCM WAV (mono and multi-channel), automatic resampling to a target
sample rate (default 16 kHz), channel selection (e.g., near/far microphones
for DCASE 2026 Task 2), and chunked streaming simulation.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Optional, Tuple, Union
import numpy as np
import scipy.signal
import soundfile as sf


@dataclass
class AudioMetadata:
    """Metadata describing an audio file."""
    path: str
    sample_rate: int
    channels: int
    duration_sec: float
    total_samples: int
    subtype: str
    target_sample_rate: int


class AudioLoader:
    """
    Robust audio loader with support for resampling, mono downmixing,
    channel preservation (e.g., near/far channels for DCASE), and streaming simulation.
    """

    def __init__(self, target_sample_rate: int = 16000):
        self.target_sample_rate = int(target_sample_rate)

    def get_metadata(self, file_path: Union[str, Path]) -> AudioMetadata:
        """Inspects an audio file without loading entire data into memory."""
        file_path = str(file_path)
        info = sf.info(file_path)
        return AudioMetadata(
            path=file_path,
            sample_rate=info.samplerate,
            channels=info.channels,
            duration_sec=info.duration,
            total_samples=info.frames,
            subtype=info.subtype,
            target_sample_rate=self.target_sample_rate,
        )

    def load(
        self,
        file_path: Union[str, Path],
        target_sr: Optional[int] = None,
        mono: bool = True,
        channel_index: Optional[int] = None,
        normalize: bool = False,
    ) -> Tuple[np.ndarray, int]:
        """
        Loads an audio file and returns (samples, sample_rate).

        Args:
            file_path: Path to the WAV file.
            target_sr: Target sample rate. If None, uses self.target_sample_rate.
            mono: If True, downmixes multi-channel audio to mono (unless channel_index is specified).
            channel_index: Optional zero-based channel index (e.g. 0 for Near, 1 for Far in DCASE).
            normalize: If True, peak-normalizes the signal to [-1.0, 1.0].

        Returns:
            Tuple of (audio_array, sample_rate).
            Shape is (N,) for mono or (C, N) for multi-channel.
        """
        target_sr = int(target_sr or self.target_sample_rate)
        file_path = str(file_path)

        data, orig_sr = sf.read(file_path, dtype="float32")

        # soundfile returns (N,) for mono, (N, C) for multi-channel
        if data.ndim > 1:
            if channel_index is not None:
                if channel_index < 0 or channel_index >= data.shape[1]:
                    raise ValueError(
                        f"Requested channel_index {channel_index}, but file has {data.shape[1]} channels."
                    )
                data = data[:, channel_index]
            elif mono:
                data = np.mean(data, axis=1)
            else:
                data = data.T  # Shape: (C, N)

        # Resample if sample rates differ
        if orig_sr != target_sr:
            data = self.resample(data, orig_sr, target_sr)

        if normalize:
            max_abs = np.max(np.abs(data))
            if max_abs > 1e-8:
                data = data / max_abs

        return np.ascontiguousarray(data, dtype=np.float32), target_sr

    @staticmethod
    def resample(data: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """
        High-quality polyphase resampling.
        Supports both 1D (samples,) and 2D (channels, samples).
        """
        if orig_sr == target_sr:
            return data

        gcd = np.gcd(orig_sr, target_sr)
        up = target_sr // gcd
        down = orig_sr // gcd

        if data.ndim == 1:
            resampled = scipy.signal.resample_poly(data, up, down)
        else:
            resampled = scipy.signal.resample_poly(data, up, down, axis=1)

        return np.ascontiguousarray(resampled, dtype=np.float32)

    def stream_frames(
        self,
        file_path: Union[str, Path],
        frame_size: int = 1024,
        hop_length: int = 512,
        target_sr: Optional[int] = None,
        mono: bool = True,
        channel_index: Optional[int] = None,
    ) -> Generator[Tuple[np.ndarray, float], None, None]:
        """
        Simulates streaming edge input by loading the audio and yielding
        sequential frames alongside their timestamp in seconds.
        Future ESP32 edge code replaces this generator with I2S ring-buffer reads.

        Yields:
            (frame_samples, timestamp_sec)
        """
        samples, sr = self.load(
            file_path,
            target_sr=target_sr,
            mono=mono,
            channel_index=channel_index,
        )

        if samples.ndim > 1:
            samples = samples[0]  # default to primary channel for streaming gate

        num_samples = len(samples)
        idx = 0
        while idx + frame_size <= num_samples:
            frame = samples[idx : idx + frame_size]
            timestamp = idx / float(sr)
            yield frame, timestamp
            idx += hop_length

        # Final partial frame if leftover samples exist
        if idx < num_samples:
            tail = samples[idx:]
            padded = np.pad(tail, (0, frame_size - len(tail)), mode="constant")
            yield padded, idx / float(sr)
