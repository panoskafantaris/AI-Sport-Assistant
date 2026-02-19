"""
Phase 4 – Ball speed + rally detection.

Ball speed is computed from world-space consecutive positions.
Rally detection uses ball presence + player activity.
"""
from __future__ import annotations
from collections import deque
from typing import List, Optional
import numpy as np

from ..models.ball   import Ball, BallTrajectory
from ..models.frame  import FrameData, RallyStats
from ..models.court  import CourtHomography
import config


class BallSpeedEstimator:
    """Estimates ball speed in metres/second using world coordinates."""

    def __init__(self, fps: float = 30.0):
        self._fps  = fps
        self._prev: Optional[Ball] = None
        self._buf  = deque(maxlen=config.SPEED_SMOOTHING_WINDOW)

    def update(self, ball: Optional[Ball], homography: Optional[CourtHomography]) -> float:
        """Return smoothed ball speed (m/s). 0.0 if not computable."""
        if ball is None or self._prev is None:
            self._prev = ball
            return 0.0

        speed = 0.0

        # World-space (preferred)
        if (ball.world_x is not None and self._prev.world_x is not None):
            dist = np.hypot(ball.world_x - self._prev.world_x,
                            ball.world_y - self._prev.world_y)
            speed = dist * self._fps

        # Pixel-space fallback with homography conversion attempt
        elif homography and homography.is_ready():
            try:
                wx1, wy1 = homography.image_to_world(self._prev.x, self._prev.y)
                wx2, wy2 = homography.image_to_world(ball.x, ball.y)
                dist  = np.hypot(wx2 - wx1, wy2 - wy1)
                speed = dist * self._fps
            except Exception:
                pass

        self._prev = ball
        self._buf.append(speed)
        return float(np.mean(self._buf))


class RallyDetector:
    """
    Detects rally start/end and accumulates per-rally stats.

    A rally is active when the ball is visible for at least N
    consecutive frames. It ends when the ball disappears for
    RALLY_GAP_FRAMES consecutive frames.
    """

    def __init__(self, fps: float = 30.0):
        self._fps             = fps
        self._rally_active    = False
        self._rally_id        = 0
        self._rally_start     = 0
        self._missing_streak  = 0
        self._shot_count      = 0
        self._max_speed       = 0.0
        self._completed: List[RallyStats] = []

    def update(
        self,
        frame_number: int,
        ball: Optional[Ball],
        ball_speed_ms: float,
    ) -> Optional[RallyStats]:
        """
        Called every frame. Returns a completed RallyStats when a
        rally ends, otherwise None.
        """
        if ball is not None:
            self._missing_streak = 0
            if not self._rally_active:
                # Rally start
                self._rally_active = True
                self._rally_start  = frame_number
                self._shot_count   = 0
                self._max_speed    = 0.0
            if ball_speed_ms > self._max_speed:
                self._max_speed = ball_speed_ms
        else:
            self._missing_streak += 1
            if self._rally_active and self._missing_streak >= config.RALLY_GAP_FRAMES:
                # Rally ended
                self._rally_active = False
                completed = RallyStats(
                    rally_id=self._rally_id,
                    start_frame=self._rally_start,
                    end_frame=frame_number,
                    shot_count=self._shot_count,
                    max_ball_speed_ms=self._max_speed,
                )
                self._rally_id += 1
                self._completed.append(completed)
                return completed

        return None

    @property
    def completed_rallies(self) -> List[RallyStats]:
        return list(self._completed)