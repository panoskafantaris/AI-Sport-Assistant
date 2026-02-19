"""
Visualizer – draws all analysis overlays onto video frames.

Separate draw methods for each phase keep this file readable.
"""
from __future__ import annotations
from typing import List, Optional
import cv2
import numpy as np

from .models.player import Player, PlayerRole
from .models.ball   import Ball, LandingEstimate, BallStatus
from .models.court  import CourtBoundary
import config

_ROLE_COLORS = {
    PlayerRole.PLAYER_NEAR:  (0, 220, 50),
    PlayerRole.PLAYER_FAR:   (255, 160, 0),
    PlayerRole.BALL_BOY:     (180, 180, 180),
    PlayerRole.UMPIRE:       (180, 180, 180),
    PlayerRole.SPECTATOR:    (100, 100, 100),
    PlayerRole.UNKNOWN:      (200, 200, 200),
}
_STATUS_COLORS = {
    BallStatus.IN:      (0, 255, 80),
    BallStatus.OUT:     (0, 0, 255),
    BallStatus.UNKNOWN: (200, 200, 0),
}


class Visualizer:

    # ── Court ──────────────────────────────────────────────────────────────────

    @staticmethod
    def draw_court(frame: np.ndarray, court: Optional[CourtBoundary]) -> np.ndarray:
        if court is None or not court.is_valid():
            return frame
        pts = court.corners.astype(np.int32)
        cv2.polylines(frame, [pts], isClosed=True, color=(0, 255, 255), thickness=2)
        for i, (x, y) in enumerate(pts):
            cv2.circle(frame, (int(x), int(y)), 6, (0, 255, 255), -1)
            cv2.putText(frame, str(i + 1), (int(x) + 8, int(y) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        return frame

    # ── Players ────────────────────────────────────────────────────────────────

    @staticmethod
    def draw_players(frame: np.ndarray, players: List[Player]) -> np.ndarray:
        for p in players:
            color = _ROLE_COLORS.get(p.role, (200, 200, 200))
            x1, y1, x2, y2 = p.bbox.to_int_tuple()
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, config.BOX_THICKNESS)

            label = f"#{p.track_id} {p.role.value.replace('_',' ')}"
            if p.speed_ms > 0.1:
                label += f" {p.speed_ms:.1f}m/s"

            stroke = p.__dict__.get("stroke", "")
            if stroke and stroke != "unknown":
                label += f" [{stroke}]"

            cv2.putText(frame, label, (x1, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, config.FONT_SCALE,
                        color, config.FONT_THICKNESS)

            # Draw movement trail
            hist = p.position_history[-12:]
            for j in range(1, len(hist)):
                alpha = j / len(hist)
                tc = tuple(int(c * alpha) for c in color)
                cv2.line(frame, (int(hist[j-1][0]), int(hist[j-1][1])),
                         (int(hist[j][0]),   int(hist[j][1])), tc, 1)
        return frame

    # ── Ball ───────────────────────────────────────────────────────────────────

    @staticmethod
    def draw_ball(frame: np.ndarray, ball: Optional[Ball]) -> np.ndarray:
        if ball is None:
            return frame
        cx, cy, r = int(ball.x), int(ball.y), max(4, int(ball.radius))
        cv2.circle(frame, (cx, cy), r, (0, 255, 255), 2)
        cv2.circle(frame, (cx, cy), 2, (0, 200, 255), -1)
        return frame

    @staticmethod
    def draw_landing(frame: np.ndarray, landing: Optional[LandingEstimate]) -> np.ndarray:
        if landing is None:
            return frame
        color = _STATUS_COLORS.get(landing.status, (200, 200, 0))
        px, py = int(landing.pixel_x), int(landing.pixel_y)
        # X mark at landing point
        d = 10
        cv2.line(frame, (px - d, py - d), (px + d, py + d), color, 2)
        cv2.line(frame, (px + d, py - d), (px - d, py + d), color, 2)
        label = f"{landing.status.value.upper()} ({landing.confidence:.0%})"
        cv2.putText(frame, label, (px + 12, py),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
        return frame

    # ── Info overlay ───────────────────────────────────────────────────────────

    @staticmethod
    def draw_info(
        frame: np.ndarray,
        frame_number: int,
        ball_speed_ms: float,
        rally_count: int,
        surface: str = "",
    ) -> np.ndarray:
        surface_tag = f"  |  {surface.upper()}" if surface and surface != "unknown" else ""
        lines = [
            f"Frame: {frame_number}{surface_tag}",
            f"Ball: {ball_speed_ms:.1f} m/s  ({ball_speed_ms * 3.6:.0f} km/h)",
            f"Rallies: {rally_count}",
        ]
        y = 24
        for line in lines:
            cv2.putText(frame, line, (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1,
                        cv2.LINE_AA)
            y += 22
        return frame