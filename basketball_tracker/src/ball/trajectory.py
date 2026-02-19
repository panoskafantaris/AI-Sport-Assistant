"""
Ball trajectory fitting + landing estimation.

We model the ball path as a parabola in the image plane (gravity
creates a natural second-order curve). A landing point is estimated
by finding where the parabola's y-coordinate would reach the expected
bounce height (near zero velocity in vertical direction).

For out/in calls we use the court homography to convert the predicted
landing point to world coordinates and check against court bounds.
"""
from __future__ import annotations
from collections import deque
from typing import Deque, Optional, Tuple
import numpy as np

from ..models.ball  import Ball, BallTrajectory, LandingEstimate, BallStatus
from ..models.court import CourtBoundary, CourtHomography
import config


class TrajectoryTracker:
    """
    Maintains a rolling window of ball detections and fits a parabola.
    """

    def __init__(self):
        self._window: Deque[Ball] = deque(maxlen=config.BALL_TRAJECTORY_WINDOW)
        self._trajectory = BallTrajectory()

    def update(self, ball: Optional[Ball]) -> BallTrajectory:
        """Add a detection (or None for missed frames) and refit."""
        if ball is not None:
            self._window.append(ball)

        if len(self._window) >= 3:
            self._fit()
        return self._trajectory

    @property
    def trajectory(self) -> BallTrajectory:
        return self._trajectory

    # ── Internals ──────────────────────────────────────────────────────────────

    def _fit(self):
        balls = list(self._window)
        t  = np.array([b.timestamp_ms for b in balls])
        t -= t[0]          # normalise to start at 0
        t /= 1000.0        # convert ms → seconds for numeric stability
        xs = np.array([b.x for b in balls])
        ys = np.array([b.y for b in balls])

        self._trajectory.detections = balls
        self._trajectory.coeff_x    = np.polyfit(t, xs, deg=min(2, len(balls)-1))
        self._trajectory.coeff_y    = np.polyfit(t, ys, deg=min(2, len(balls)-1))

        # Fit in world coords only if ALL detections in window have valid coords
        world_balls = [b for b in balls if b.world_x is not None and b.world_y is not None]
        if len(world_balls) >= 3:
            wt  = np.array([b.timestamp_ms for b in world_balls])
            wt -= wt[0]
            wt /= 1000.0
            wxs = np.array([b.world_x for b in world_balls], dtype=float)
            wys = np.array([b.world_y for b in world_balls], dtype=float)
            deg = min(2, len(world_balls) - 1)
            self._trajectory.world_coeff_x = np.polyfit(wt, wxs, deg)
            self._trajectory.world_coeff_y = np.polyfit(wt, wys, deg)
        else:
            self._trajectory.world_coeff_x = None
            self._trajectory.world_coeff_y = None


class LandingPredictor:
    """
    Predicts where the ball will land and whether it is IN or OUT.
    """

    def predict(
        self,
        trajectory: BallTrajectory,
        court: Optional[CourtBoundary],
        homography: Optional[CourtHomography],
    ) -> Optional[LandingEstimate]:
        if not trajectory.is_fitted or len(trajectory.detections) < 3:
            return None

        # Estimate landing from parabola: find t where vy ≈ 0 or y peaks
        land_px, land_t = self._find_landing_pixel(trajectory)
        if land_px is None:
            return None

        px, py = land_px

        # World coordinates
        wx, wy = None, None
        if homography and homography.is_ready():
            try:
                wx, wy = homography.image_to_world(px, py)
            except Exception:
                pass

        # In/Out decision
        status, conf = self._call_in_out(px, py, wx, wy, court)
        frames_ahead = max(0, int(land_t * 30))  # rough: 30fps

        return LandingEstimate(
            pixel_x=px, pixel_y=py,
            world_x=wx, world_y=wy,
            status=status,
            confidence=conf,
            frames_until_landing=frames_ahead,
        )

    # ── Internals ──────────────────────────────────────────────────────────────

    def _find_landing_pixel(
        self, traj: BallTrajectory
    ) -> Tuple[Optional[Tuple[float, float]], float]:
        """
        Find the t where the parabola y-derivative = 0 (apex)
        and the next downward crossing (landing after bounce).
        """
        cy = traj.coeff_y
        if len(cy) < 2:
            return None, 0.0

        # Derivative of parabola at²+bt+c is 2at+b=0 → t=-b/(2a)
        if len(cy) == 3 and abs(cy[0]) > 1e-6:
            t_apex = -cy[1] / (2 * cy[0])
        else:
            t_apex = 0.5  # fallback

        # Predict horizon_seconds ahead
        horizon = config.BALL_LANDING_HORIZON / 30.0
        t_land  = t_apex + horizon

        try:
            px = float(np.polyval(traj.coeff_x, t_land))
            py = float(np.polyval(traj.coeff_y, t_land))
            return (px, py), t_land
        except Exception:
            return None, 0.0

    def _call_in_out(
        self,
        px: float, py: float,
        wx: Optional[float], wy: Optional[float],
        court: Optional[CourtBoundary],
    ) -> Tuple[BallStatus, float]:
        margin = config.BALL_OUT_MARGIN_PX

        # World-space call (preferred – precise)
        if wx is not None and wy is not None:
            cw = config.COURT_WIDTH_SINGLE
            cl = config.COURT_LENGTH
            if (-margin/100 <= wx <= cw + margin/100 and
                    -margin/100 <= wy <= cl + margin/100):
                return BallStatus.IN, 0.92
            return BallStatus.OUT, 0.88

        # Pixel-space fallback
        if court and court.is_valid():
            inside = court.contains_point(px, py, margin=margin)
            return (BallStatus.IN, 0.70) if inside else (BallStatus.OUT, 0.65)

        return BallStatus.UNKNOWN, 0.0