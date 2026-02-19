"""
Ball-related data models.
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple
import numpy as np


class BallStatus(Enum):
    IN_PLAY  = "in_play"
    BOUNCING = "bouncing"
    IN       = "in"
    OUT      = "out"
    UNKNOWN  = "unknown"


@dataclass
class Ball:
    """Single-frame ball detection."""
    x: float
    y: float
    radius: float
    confidence: float
    frame_number: int
    timestamp_ms: float

    # World coordinates if homography available
    world_x: Optional[float] = None
    world_y: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "frame": self.frame_number,
            "px": [round(self.x, 1), round(self.y, 1)],
            "radius": round(self.radius, 1),
            "conf": round(self.confidence, 3),
            "world_m": (
                [round(self.world_x, 3), round(self.world_y, 3)]
                if self.world_x is not None else None
            ),
        }


@dataclass
class BallTrajectory:
    """
    Fitted trajectory from recent ball detections.

    We fit a parabola in both x(t) and y(t) using least squares,
    which accounts for gravity naturally in the image plane.
    """
    detections: List[Ball] = field(default_factory=list)

    # Polynomial coefficients [a, b, c] for x=at²+bt+c, y=at²+bt+c
    coeff_x: Optional[np.ndarray] = None
    coeff_y: Optional[np.ndarray] = None

    # Fitted in world coords (metres) if available
    world_coeff_x: Optional[np.ndarray] = None
    world_coeff_y: Optional[np.ndarray] = None

    @property
    def is_fitted(self) -> bool:
        return self.coeff_x is not None and self.coeff_y is not None

    def predict_position(self, t: float) -> Tuple[float, float]:
        """Predict pixel position at relative time t."""
        if not self.is_fitted:
            raise RuntimeError("Trajectory not fitted")
        x = np.polyval(self.coeff_x, t)
        y = np.polyval(self.coeff_y, t)
        return float(x), float(y)

    def predict_world_position(self, t: float) -> Optional[Tuple[float, float]]:
        if self.world_coeff_x is None:
            return None
        x = np.polyval(self.world_coeff_x, t)
        y = np.polyval(self.world_coeff_y, t)
        return float(x), float(y)


@dataclass
class LandingEstimate:
    """
    Estimated ball landing point and in/out call.
    """
    pixel_x: float
    pixel_y: float
    world_x: Optional[float] = None
    world_y: Optional[float] = None
    status: BallStatus = BallStatus.UNKNOWN
    confidence: float = 0.0          # 0-1, how confident the model is
    frames_until_landing: int = 0

    def to_dict(self) -> dict:
        return {
            "landing_px": [round(self.pixel_x, 1), round(self.pixel_y, 1)],
            "landing_world_m": (
                [round(self.world_x, 3), round(self.world_y, 3)]
                if self.world_x is not None else None
            ),
            "status": self.status.value,
            "confidence": round(self.confidence, 3),
            "frames_ahead": self.frames_until_landing,
        }