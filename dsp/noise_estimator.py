"""
Adaptive Noise-Floor Estimator Module.

Maintains running estimates of background noise levels using Exponential Moving
Average (EMA) or moving percentile/median with asymmetric adaptation and
event-aware model freezing.
"""

from collections import deque
from typing import Dict, List, Optional, Tuple, Union
import numpy as np


class AdaptiveNoiseEstimator:
    """
    Adaptive noise floor tracking for audio streams.

    Features:
    - Moving median / percentile buffer to reject brief transient spikes.
    - Asymmetric Exponential Moving Average (EMA):
        * slower adaptation upward (prevents transient bursts from polluting noise model)
        * faster adaptation downward (recovers when noise drops)
    - Event-aware freezing: updates are paused during confirmed trigger events.
    - Multi-band support: tracks noise floor independently for each configured sub-band
      as well as overall full-band energy.
    """

    def __init__(
        self,
        alpha_up: float = 0.01,
        alpha_down: float = 0.05,
        history_len: int = 30,
        percentile: float = 20.0,
        min_noise_floor: float = 1e-7,
    ):
        """
        Args:
            alpha_up: EMA smoothing rate when incoming energy > current noise floor (slow).
            alpha_down: EMA smoothing rate when incoming energy < current noise floor (faster).
            history_len: Number of recent frames stored for percentile computation.
            percentile: Percentile (0-100) to use for baseline noise estimation (e.g. 20th percentile).
            min_noise_floor: Minimum floor to prevent division by zero in SNR calculation.
        """
        self.alpha_up = float(alpha_up)
        self.alpha_down = float(alpha_down)
        self.history_len = int(history_len)
        self.percentile = float(percentile)
        self.min_noise_floor = float(min_noise_floor)

        # State storage: keys are "total" or band names
        self.noise_floors: Dict[str, float] = {}
        self.history_buffers: Dict[str, deque] = {}
        self.is_frozen: bool = False
        self.frame_count: int = 0

    def reset(self) -> None:
        """Resets all noise floor tracking states."""
        self.noise_floors.clear()
        self.history_buffers.clear()
        self.is_frozen = False
        self.frame_count = 0

    def set_frozen(self, frozen: bool) -> None:
        """
        Freezes or unfreezes noise floor adaptation.
        When frozen (during acoustic events), the background noise model is held static
        so the defect signal does not raise the baseline and mask itself.
        """
        self.is_frozen = frozen

    def update(
        self,
        energies: Dict[str, float],
        freeze_override: Optional[bool] = None,
    ) -> Dict[str, float]:
        """
        Updates the running noise floor for given band energies.

        Args:
            energies: Dictionary of {channel/band_name: current_energy_value}.
                      Can include "total" for full-band energy.
            freeze_override: If specified, overrides self.is_frozen for this step.

        Returns:
            Dictionary of updated noise floor values for each key.
        """
        should_freeze = self.is_frozen if freeze_override is None else freeze_override

        for key, energy in energies.items():
            energy = max(float(energy), self.min_noise_floor)

            if key not in self.history_buffers:
                self.history_buffers[key] = deque(maxlen=self.history_len)
                self.noise_floors[key] = energy

            buffer = self.history_buffers[key]

            # In the initial warm-up phase (first 5 frames), unconditionally update
            if self.frame_count < 5:
                buffer.append(energy)
                self.noise_floors[key] = energy
                continue

            # If frozen, do not append to history or adapt EMA
            if should_freeze:
                continue

            buffer.append(energy)

            # 1. Percentile estimate from history window
            if len(buffer) >= 3:
                perc_est = float(np.percentile(list(buffer), self.percentile))
            else:
                perc_est = energy

            # 2. Asymmetric EMA toward percentile estimate
            current_floor = self.noise_floors[key]
            if perc_est > current_floor:
                alpha = self.alpha_up
            else:
                alpha = self.alpha_down

            updated_floor = (1.0 - alpha) * current_floor + alpha * perc_est
            self.noise_floors[key] = max(updated_floor, self.min_noise_floor)

        self.frame_count += 1
        return self.noise_floors.copy()

    def get_noise_floor(self, key: str = "total") -> float:
        """Returns the current noise floor for the specified key."""
        return self.noise_floors.get(key, self.min_noise_floor)

    def calculate_snr(self, energy: float, key: str = "total") -> Tuple[float, float]:
        """
        Calculates linear SNR and SNR in dB for a given energy reading against the tracked floor.

        Returns:
            Tuple of (snr_linear, snr_db).
        """
        noise = self.get_noise_floor(key)
        snr_linear = max(energy / noise, 1e-12)
        snr_db = 10.0 * np.log10(snr_linear)
        return float(snr_linear), float(snr_db)
