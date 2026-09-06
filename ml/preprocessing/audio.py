"""
Audio Preprocessing Utilities for Machine Anomaly ML Pipeline.
"""

from typing import Optional, Tuple
import numpy as np
from dsp.audio_loader import AudioLoader


class AudioPreprocessor:
    """Prepares raw audio arrays for downstream ML model ingestion."""

    def __init__(self, target_sample_rate: int = 16000, clip_duration_sec: float = 10.0):
        self.sample_rate = target_sample_rate
        self.clip_duration_sec = clip_duration_sec
        self.clip_samples = int(clip_duration_sec * target_sample_rate)
        self.loader = AudioLoader(target_sample_rate=self.sample_rate)

    def process(
        self,
        audio_path_or_array,
        channel_index: Optional[int] = None,
        normalize: bool = True,
    ) -> np.ndarray:
        """Loads, downmixes/selects channel, resamples, and pads/crops to fixed length."""
        if isinstance(audio_path_or_array, str):
            samples, _ = self.loader.load(
                audio_path_or_array,
                target_sr=self.sample_rate,
                mono=True,
                channel_index=channel_index,
                normalize=normalize,
            )
        else:
            samples = np.asarray(audio_path_or_array, dtype=np.float32)
            if samples.ndim > 1:
                if channel_index is not None:
                    samples = samples[channel_index]
                else:
                    samples = np.mean(samples, axis=0)

        # Ensure consistent length (pad with zeros or center crop)
        if len(samples) < self.clip_samples:
            pad_len = self.clip_samples - len(samples)
            samples = np.pad(samples, (0, pad_len), mode="constant")
        elif len(samples) > self.clip_samples:
            samples = samples[: self.clip_samples]

        return samples.astype(np.float32)
