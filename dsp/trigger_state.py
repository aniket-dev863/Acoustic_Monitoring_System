"""
Trigger State and Temporal Hysteresis Module.

Manages temporal decision logic, confirmation windows, hysteresis thresholds,
and minimum hold durations to avoid chattering and false edge triggers.
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional


class GateDecision(str, Enum):
    IDLE = "IDLE"           # Signal below threshold
    PENDING = "PENDING"     # Exceeded threshold, awaiting confirmation frames
    TRIGGERED = "TRIGGERED" # Active confirmed trigger event
    RELEASE = "RELEASE"     # Dropped below release threshold, in release grace period


@dataclass
class EventInterval:
    """Represents a confirmed acoustic event interval."""
    start_frame: int
    end_frame: int
    start_time: float
    end_time: float
    peak_score: float
    mean_score: float


class TriggerStateMachine:
    """
    Temporal confirmation and hysteresis state machine.

    Parameters:
        trigger_threshold: Score required to initiate / enter pending trigger.
        release_threshold: Score threshold below which release countdown starts (hysteresis).
        confirm_frames: Number of consecutive frames exceeding trigger_threshold needed to trigger.
        release_frames: Number of consecutive frames below release_threshold needed to return to idle.
        min_event_frames: Minimum number of frames a triggered event must stay active (hold time).
    """

    def __init__(
        self,
        trigger_threshold: float = 0.60,
        release_threshold: float = 0.40,
        confirm_frames: int = 3,
        release_frames: int = 4,
        min_event_frames: int = 6,
    ):
        if release_threshold > trigger_threshold:
            raise ValueError(f"release_threshold ({release_threshold}) cannot exceed trigger_threshold ({trigger_threshold})")

        self.trigger_threshold = float(trigger_threshold)
        self.release_threshold = float(release_threshold)
        self.confirm_frames = int(max(1, confirm_frames))
        self.release_frames = int(max(1, release_frames))
        self.min_event_frames = int(max(1, min_event_frames))

        self.state: GateDecision = GateDecision.IDLE
        self.consecutive_high: int = 0
        self.consecutive_low: int = 0
        self.active_event_frames: int = 0

        self.current_event_start_frame: Optional[int] = None
        self.current_event_start_time: Optional[float] = None
        self.current_event_scores: List[float] = []

        self.completed_events: List[EventInterval] = []

    def reset(self) -> None:
        """Resets state machine to initial IDLE state."""
        self.state = GateDecision.IDLE
        self.consecutive_high = 0
        self.consecutive_low = 0
        self.active_event_frames = 0
        self.current_event_start_frame = None
        self.current_event_start_time = None
        self.current_event_scores.clear()
        self.completed_events.clear()

    def update_thresholds(
        self,
        trigger_threshold: Optional[float] = None,
        release_threshold: Optional[float] = None,
    ) -> None:
        """Dynamically adjusts thresholds (e.g. for resource-awareness adaptation)."""
        if trigger_threshold is not None:
            self.trigger_threshold = float(trigger_threshold)
        if release_threshold is not None:
            self.release_threshold = float(release_threshold)
        if self.release_threshold > self.trigger_threshold:
            self.release_threshold = self.trigger_threshold * 0.7

    def step(
        self,
        score: float,
        frame_index: int,
        timestamp: float,
    ) -> bool:
        """
        Steps the state machine with a new frame score.

        Returns:
            bool: True if the gate is currently TRIGGERED, False otherwise.
        """
        score = float(score)

        if self.state in (GateDecision.IDLE, GateDecision.PENDING):
            if score >= self.trigger_threshold:
                self.consecutive_high += 1
                self.consecutive_low = 0

                if self.state == GateDecision.IDLE:
                    self.state = GateDecision.PENDING
                    self.current_event_start_frame = frame_index
                    self.current_event_start_time = timestamp
                    self.current_event_scores = [score]
                else:
                    self.current_event_scores.append(score)

                if self.consecutive_high >= self.confirm_frames:
                    self.state = GateDecision.TRIGGERED
                    self.active_event_frames = self.consecutive_high
            else:
                # Dropped before confirmation
                self.consecutive_high = 0
                self.state = GateDecision.IDLE
                self.current_event_scores.clear()
                self.current_event_start_frame = None
                self.current_event_start_time = None

        elif self.state in (GateDecision.TRIGGERED, GateDecision.RELEASE):
            self.active_event_frames += 1
            self.current_event_scores.append(score)

            if score < self.release_threshold:
                self.consecutive_low += 1
                self.consecutive_high = 0
                self.state = GateDecision.RELEASE

                # Check release conditions: must satisfy min_event_frames AND consecutive_low >= release_frames
                if (
                    self.active_event_frames >= self.min_event_frames
                    and self.consecutive_low >= self.release_frames
                ):
                    self._finalize_event(frame_index, timestamp)
                    self.state = GateDecision.IDLE
                    self.consecutive_low = 0
            else:
                # Score climbed back above release threshold; reset release countdown
                self.consecutive_low = 0
                self.state = GateDecision.TRIGGERED

        return self.state in (GateDecision.TRIGGERED, GateDecision.RELEASE)

    def _finalize_event(self, end_frame: int, end_time: float) -> None:
        """Records a completed event interval."""
        if self.current_event_start_frame is not None and len(self.current_event_scores) > 0:
            peak = float(max(self.current_event_scores))
            mean = float(sum(self.current_event_scores) / len(self.current_event_scores))
            event = EventInterval(
                start_frame=self.current_event_start_frame,
                end_frame=end_frame,
                start_time=self.current_event_start_time if self.current_event_start_time is not None else 0.0,
                end_time=end_time,
                peak_score=peak,
                mean_score=mean,
            )
            self.completed_events.append(event)
        self.current_event_scores = []
        self.current_event_start_frame = None
        self.current_event_start_time = None
        self.active_event_frames = 0

    def finalize(self, total_frames: int, total_duration: float) -> List[EventInterval]:
        """Finalizes any active in-flight event when audio stream finishes."""
        if self.state in (GateDecision.TRIGGERED, GateDecision.RELEASE):
            self._finalize_event(total_frames - 1, total_duration)
            self.state = GateDecision.IDLE
        return list(self.completed_events)
