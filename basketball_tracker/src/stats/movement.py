"""
Phase 4 – Player movement & speed analysis.

Converts pixel-space position history → real-world metres
using court homography, then computes speed, acceleration,
distance covered, and heatmap data.
"""
from __future__ import annotations
from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple
import numpy as np

from ..models.player import Player, PlayerRole
from ..models.court  import CourtHomography
import config


class MovementAnalyser:
    """
    Per-player movement statistics across all frames.
    """

    def __init__(self, fps: float = 30.0):
        self._fps   = fps
        self._speed_buf: Dict[int, deque] = defaultdict(
            lambda: deque(maxlen=config.SPEED_SMOOTHING_WINDOW)
        )
        self._total_dist: Dict[int, float] = defaultdict(float)
        self._last_world: Dict[int, Tuple[float, float]] = {}

    def update(
        self,
        players: List[Player],
        homography: Optional[CourtHomography] = None,
    ) -> List[Player]:
        """
        Enrich each Player with speed and world-space positions.
        Modifies in-place and returns the same list.
        """
        for p in players:
            tid = p.track_id
            fx, fy = p.bbox.foot_center

            # Convert to world if possible
            wx, wy = None, None
            if homography and homography.is_ready():
                try:
                    wx, wy = homography.image_to_world(fx, fy)
                    p.world_position_history = getattr(p, "world_position_history", [])
                    p.world_position_history.append((wx, wy))
                except Exception:
                    pass

            # Speed in metres/second
            if wx is not None and tid in self._last_world:
                lx, ly = self._last_world[tid]
                dist_m = np.hypot(wx - lx, wy - ly)
                speed  = dist_m * self._fps           # m/s

                self._speed_buf[tid].append(speed)
                self._total_dist[tid] += dist_m

                p.speed_ms = float(np.mean(self._speed_buf[tid]))
            elif tid in self._last_world:
                # Pixel-space fallback
                lx, ly = self._last_world.get(tid, (fx, fy))
                p.speed_px_per_frame = float(np.hypot(fx - lx, fy - ly))

            # Update last position
            if wx is not None:
                self._last_world[tid] = (wx, wy)
            else:
                self._last_world[tid] = (fx, fy)

        return players

    def total_distance(self, track_id: int) -> float:
        """Cumulative distance covered by a player (metres)."""
        return self._total_dist.get(track_id, 0.0)

    def summary(self) -> Dict[int, dict]:
        return {
            tid: {
                "total_distance_m": round(self._total_dist[tid], 2),
                "avg_speed_ms":     round(float(np.mean(list(buf))), 2) if buf else 0.0,
                "max_speed_ms":     round(float(max(buf)), 2) if buf else 0.0,
            }
            for tid, buf in self._speed_buf.items()
        }


class HeatmapAccumulator:
    """
    Accumulates player positions into a 2-D world-space grid.
    Call .to_image() to get a coloured heatmap overlay.
    """

    def __init__(self, grid_w: int = 100, grid_h: int = 200):
        self._grid = defaultdict(lambda: np.zeros((grid_h, grid_w), np.float32))
        self._gw = grid_w
        self._gh = grid_h

    def update(self, players: List[Player]):
        cw = config.COURT_WIDTH_SINGLE
        cl = config.COURT_LENGTH

        for p in players:
            if not p.world_position_history:
                continue
            wx, wy = p.world_position_history[-1]
            gx = int(np.clip(wx / cw * (self._gw - 1), 0, self._gw - 1))
            gy = int(np.clip(wy / cl * (self._gh - 1), 0, self._gh - 1))
            self._grid[p.role.value][gy, gx] += 1.0

    def to_image(self, role: str, size: Tuple[int, int]) -> Optional[np.ndarray]:
        import cv2
        grid = self._grid.get(role)
        if grid is None:
            return None
        norm = cv2.normalize(grid, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        coloured = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
        return cv2.resize(coloured, size)