"""
Audio Framing and Windowing Module.

Provides both batch framing for offline analysis and stateful streaming framing
for edge simulation (mimicking circular/ring-buffer DSP ingestion).
"""

from typing import Generator, List, Optional, Tuple, Union
import numpy as np


class FrameBuffer:
    """
    Stateful circular FIFO frame buffer for streaming samples.
    Accumulates incoming samples and produces complete overlapping frames
    of length `frame_size` stepped by `hop_length`.
    Compatible with future ESP32 I2S DMA buffer callbacks.
    """

    def __init__(self, frame_size: int = 1024, hop_length: int = 512):
        if hop_length > frame_size:
            raise ValueError(f"hop_length ({hop_length}) cannot be larger than frame_size ({frame_size})")
        self.frame_size = int(frame_size)
        self.hop_length = int(hop_length)
        self.buffer = np.zeros(0, dtype=np.float32)
        self.total_samples_ingested = 0

    def push(self, new_samples: np.ndarray) -> Generator[np.ndarray, None, None]:
        """
        Pushes a new batch of audio samples into the buffer.
        Yields all completed frames of size `frame_size`.
        """
        if new_samples.size == 0:
            return

        new_samples = np.asarray(new_samples, dtype=np.float32).flatten()
        self.buffer = np.concatenate((self.buffer, new_samples))
        self.total_samples_ingested += len(new_samples)

        while len(self.buffer) >= self.frame_size:
            frame = self.buffer[: self.frame_size]
            yield frame
            # Slide window forward by hop_length
            self.buffer = self.buffer[self.hop_length :]

    def flush(self, pad_with_zeros: bool = True) -> Optional[np.ndarray]:
        """
        Flushes remaining buffer samples upon stream end.
        """
        if len(self.buffer) == 0:
            return None
        if pad_with_zeros:
            padded = np.pad(self.buffer, (0, self.frame_size - len(self.buffer)), mode="constant")
            self.buffer = np.zeros(0, dtype=np.float32)
            return padded
        remaining = self.buffer
        self.buffer = np.zeros(0, dtype=np.float32)
        return remaining

    def reset(self) -> None:
        """Resets the internal buffer state."""
        self.buffer = np.zeros(0, dtype=np.float32)
        self.total_samples_ingested = 0


def create_window(window_type: str, frame_size: int) -> np.ndarray:
    """Creates a normalized analysis window."""
    window_type = window_type.lower()
    if window_type == "hann" or window_type == "hanning":
        return np.hanning(frame_size).astype(np.float32)
    elif window_type == "hamming":
        return np.hamming(frame_size).astype(np.float32)
    elif window_type == "blackman":
        return np.blackman(frame_size).astype(np.float32)
    elif window_type == "rectangular" or window_type == "rect":
        return np.ones(frame_size, dtype=np.float32)
    else:
        raise ValueError(f"Unsupported window type: {window_type}")


def frame_signal(
    signal: np.ndarray,
    frame_size: int = 1024,
    hop_length: int = 512,
    window_type: Optional[str] = "hann",
    pad: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Frames a 1D audio signal into overlapping frames.

    Args:
        signal: 1D array of audio samples.
        frame_size: Length of each frame in samples.
        hop_length: Number of samples between successive frames.
        window_type: Windowing function ("hann", "hamming", etc.), or None.
        pad: If True, pads the signal so the full audio duration is covered.

    Returns:
        Tuple of (frames, windowed_frames).
        Both have shape (num_frames, frame_size).
    """
    signal = np.asarray(signal, dtype=np.float32).flatten()
    sig_len = len(signal)

    if pad:
        if sig_len < frame_size:
            pad_amount = frame_size - sig_len
            signal = np.pad(signal, (0, pad_amount), mode="constant")
            sig_len = len(signal)
        else:
            remainder = (sig_len - frame_size) % hop_length
            if remainder != 0:
                pad_amount = hop_length - remainder
                signal = np.pad(signal, (0, pad_amount), mode="constant")
                sig_len = len(signal)

    num_frames = 1 + max(0, (sig_len - frame_size) // hop_length)
    shape = (num_frames, frame_size)
    strides = (signal.strides[0] * hop_length, signal.strides[0])
    raw_frames = np.lib.stride_tricks.as_strided(signal, shape=shape, strides=strides).copy()

    if window_type:
        win = create_window(window_type, frame_size)
        windowed_frames = raw_frames * win
    else:
        windowed_frames = raw_frames.copy()

    return raw_frames, windowed_frames
