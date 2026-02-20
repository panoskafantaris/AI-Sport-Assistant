"""
Temporal tracker — Kalman filter for court boundary smoothing.

Tracks the 8 corner coordinates (4 points × 2 dims) across frames.
Benefits:
  - Smooths jitter between frames
  - Interpolates through brief occlusions
  - Provides strong prior for next frame's RANSAC
  - Resets on scene changes
"""
from __future__ import annotations
from typing import Optional
import numpy as np


class TemporalTracker:
    """Kalman filter tracking 4 corner positions over time."""

    def __init__(
        self,
        process_noise: float = 2.0,
        measurement_noise: float = 5.0,
        max_prediction_frames: int = 10,
    ):
        self._process_noise = process_noise
        self._measurement_noise = measurement_noise
        self._max_predict = max_prediction_frames

        # State: 8-dimensional (x1,y1,x2,y2,x3,y3,x4,y4)
        # No velocity model — corners change slowly in broadcast
        self._state: Optional[np.ndarray] = None   # (8,)
        self._cov: Optional[np.ndarray] = None      # (8,8)
        self._frames_without_measurement = 0
        self._initialized = False

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    @property
    def predicted_corners(self) -> Optional[np.ndarray]:
        """Current state as (4,2) corners, or None."""
        if not self._initialized:
            return None
        return self._state.reshape(4, 2).copy()

    def reset(self) -> None:
        """Reset tracker (e.g., on scene change)."""
        self._state = None
        self._cov = None
        self._initialized = False
        self._frames_without_measurement = 0

    def update(
        self,
        corners: Optional[np.ndarray],
        confidence: float = 1.0,
    ) -> np.ndarray:
        """
        Update tracker with a new measurement.

        Args:
            corners: (4,2) detected corners, or None if detection failed
            confidence: detection confidence (scales measurement noise)

        Returns:
            (4,2) smoothed corners
        """
        if corners is not None:
            measurement = corners.flatten().astype(np.float64)
        else:
            measurement = None

        if not self._initialized:
            if measurement is not None:
                self._initialize(measurement)
                return self._state.reshape(4, 2).astype(np.float32)
            return None

        # Predict step (simple: state doesn't change, covariance grows)
        Q = np.eye(8) * self._process_noise ** 2
        self._cov = self._cov + Q

        if measurement is not None:
            # Measurement update
            # Scale noise inversely with confidence
            noise_scale = 1.0 / max(confidence, 0.1)
            R = np.eye(8) * (self._measurement_noise * noise_scale) ** 2

            # Kalman gain
            S = self._cov + R
            K = self._cov @ np.linalg.inv(S)

            # Update
            innovation = measurement - self._state
            self._state = self._state + K @ innovation
            self._cov = (np.eye(8) - K) @ self._cov
            self._frames_without_measurement = 0
        else:
            self._frames_without_measurement += 1

        # If too many frames without measurement, tracker is unreliable
        if self._frames_without_measurement > self._max_predict:
            self.reset()
            return None

        return self._state.reshape(4, 2).astype(np.float32)

    def _initialize(self, measurement: np.ndarray) -> None:
        """Initialize state from first measurement."""
        self._state = measurement.copy()
        self._cov = np.eye(8) * self._measurement_noise ** 2
        self._initialized = True
        self._frames_without_measurement = 0

    def get_prior_corners(self) -> Optional[np.ndarray]:
        """
        Get predicted corners as a prior for the next detection.

        This helps RANSAC by providing an initial guess that
        narrows the search space.
        """
        if not self._initialized:
            return None
        return self._state.reshape(4, 2).astype(np.float32)