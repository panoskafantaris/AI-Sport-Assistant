"""
Phase 2 – Player detector.

Detects all persons with YOLO, then filters to retain only
the actual players (excluding ball boys, umpire, line judges,
and crowd) using position + size heuristics + court boundary.
"""
from __future__ import annotations
from typing import List, Optional
import numpy as np
from ultralytics import YOLO

from ..models.player import Player, PlayerRole, BoundingBox
from ..models.court  import CourtBoundary
import config


class PlayerDetector:
    """Detects and classifies persons on the tennis court."""

    def __init__(self):
        self._model = YOLO(config.PERSON_MODEL)

    # ── Public API ─────────────────────────────────────────────────────────────

    def detect(
        self,
        frame: np.ndarray,
        court: Optional[CourtBoundary] = None,
    ) -> List[Player]:
        """
        Run YOLO detection and return only relevant players.

        Args:
            frame:  BGR video frame.
            court:  Current court boundary for on-court filtering.

        Returns:
            List of Player objects with assigned roles.
        """
        raw = self._yolo_detect(frame)
        players = self._filter_and_classify(raw, frame, court)
        return players

    # ── Internals ──────────────────────────────────────────────────────────────

    def _yolo_detect(self, frame: np.ndarray) -> List[Player]:
        results = self._model.predict(
            frame,
            conf=config.DETECTION_CONF,
            iou=config.DETECTION_IOU,
            imgsz=config.DETECTION_IMG_SIZE,
            classes=[config.PERSON_CLASS_ID],
            verbose=False,
        )
        players = []
        for res in results:
            for box in res.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                tid  = int(box.id[0]) if box.id is not None else -1
                players.append(Player(
                    track_id=tid,
                    bbox=BoundingBox(x1, y1, x2, y2),
                    confidence=conf,
                ))
        return players

    def _filter_and_classify(
        self,
        players: List[Player],
        frame: np.ndarray,
        court: Optional[CourtBoundary],
    ) -> List[Player]:
        H, W = frame.shape[:2]
        classified: List[Player] = []

        for p in players:
            role = self._classify_role(p, H, W, court)
            p.role = role
            classified.append(p)

        # Only keep real players + sort by confidence
        active = [p for p in classified if p.role in (
            PlayerRole.PLAYER_NEAR, PlayerRole.PLAYER_FAR, PlayerRole.UNKNOWN
        )]
        active.sort(key=lambda p: p.confidence, reverse=True)
        return active

    def _classify_role(
        self,
        player: Player,
        frame_h: int,
        frame_w: int,
        court: Optional[CourtBoundary],
    ) -> PlayerRole:
        bbox = player.bbox
        cx, cy = bbox.center
        fx, fy = bbox.foot_center
        box_h  = bbox.height

        # ── Reject spectators: outside court + large crowd zone ──────────────
        if court and court.is_valid():
            on_court = court.contains_point(fx, fy, margin=config.COURT_MARGIN_PX)
            if not on_court:
                return PlayerRole.SPECTATOR

        # ── Reject umpire: typically stationary high-up near net ─────────────
        if cy < frame_h * config.UMPIRE_ZONE_TOP_RATIO:
            return PlayerRole.UMPIRE

        # ── Heuristic: ball boys are shorter AND near the court edges ─────────
        if court and court.is_valid():
            median_h = self._median_person_height_approx(frame_h)
            if box_h < median_h * config.BALL_BOY_HEIGHT_RATIO:
                return PlayerRole.BALL_BOY

        # ── Determine side (near / far) by vertical position ─────────────────
        court_mid_y = frame_h * 0.5
        if court and court.is_valid():
            # Use court midpoint if available
            ys = court.corners[:, 1]
            court_mid_y = (ys.min() + ys.max()) / 2

        if fy > court_mid_y:
            return PlayerRole.PLAYER_NEAR
        else:
            return PlayerRole.PLAYER_FAR

    @staticmethod
    def _median_person_height_approx(frame_h: int) -> float:
        """Rough expected height of a standing person in the frame."""
        return frame_h * 0.25   # heuristic: full-body ≈ 25% frame height