"""
Player tracker – persistent identity across frames using YOLO + ByteTrack.

Wraps detection + tracking in a single pass so track IDs are stable.
"""
from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import numpy as np
from ultralytics import YOLO

from ..models.player import Player, PlayerRole, BoundingBox
from ..models.court  import CourtBoundary
import config


class PlayerTracker:
    """
    Tracks players using YOLO's built-in ByteTrack integration.

    Maintains per-track history and propagates role assignments.
    """

    def __init__(self):
        self._model = YOLO(config.PERSON_MODEL)
        # Persistent role memory: track_id → PlayerRole
        self._role_memory: Dict[int, PlayerRole] = {}
        # Position history: track_id → list of (x, y) foot centers
        self._history: Dict[int, List[Tuple[float, float]]] = {}

    def track(
        self,
        frame: np.ndarray,
        court: Optional[CourtBoundary] = None,
    ) -> List[Player]:
        """
        Run tracking on a frame and return Players with stable IDs.
        """
        results = self._model.track(
            frame,
            persist=True,
            conf=config.TRACK_HIGH_THRESH,
            iou=config.DETECTION_IOU,
            imgsz=config.DETECTION_IMG_SIZE,
            classes=[config.PERSON_CLASS_ID],
            tracker=f"{config.TRACKER_TYPE}.yaml",
            verbose=False,
        )

        players: List[Player] = []
        H, W = frame.shape[:2]

        for res in results:
            for box in res.boxes:
                if box.id is None:
                    continue
                tid  = int(box.id[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                bbox = BoundingBox(x1, y1, x2, y2)

                # Update history
                if tid not in self._history:
                    self._history[tid] = []
                self._history[tid].append(bbox.foot_center)

                # Classify role (persist once assigned)
                role = self._get_or_assign_role(tid, bbox, H, W, court)

                p = Player(
                    track_id=tid,
                    bbox=bbox,
                    confidence=conf,
                    role=role,
                    position_history=list(self._history[tid]),
                )
                players.append(p)

        return self._retain_active_players(players, court, H)

    def reset(self):
        self._role_memory.clear()
        self._history.clear()

    # ── Internals ──────────────────────────────────────────────────────────────

    def _get_or_assign_role(
        self,
        tid: int,
        bbox: BoundingBox,
        H: int,
        W: int,
        court: Optional[CourtBoundary],
    ) -> PlayerRole:
        # Return cached role (once decided as PLAYER, keep it)
        if tid in self._role_memory:
            cached = self._role_memory[tid]
            if cached in (PlayerRole.PLAYER_NEAR, PlayerRole.PLAYER_FAR):
                return cached

        role = self._classify(bbox, H, W, court)
        self._role_memory[tid] = role
        return role

    def _classify(
        self,
        bbox: BoundingBox,
        H: int,
        W: int,
        court: Optional[CourtBoundary],
    ) -> PlayerRole:
        fx, fy = bbox.foot_center
        cx, cy = bbox.center

        # Outside court → spectator
        if court and court.is_valid():
            if not court.contains_point(fx, fy, margin=config.COURT_MARGIN_PX):
                return PlayerRole.SPECTATOR

        # Top zone → umpire chair
        if cy < H * config.UMPIRE_ZONE_TOP_RATIO:
            return PlayerRole.UMPIRE

        # Very short → ball boy
        if bbox.height < H * 0.15:
            return PlayerRole.BALL_BOY

        # Near vs far side
        court_mid = H * 0.5
        if court and court.is_valid():
            ys = court.corners[:, 1]
            court_mid = (ys.min() + ys.max()) / 2

        return PlayerRole.PLAYER_NEAR if fy > court_mid else PlayerRole.PLAYER_FAR

    @staticmethod
    def _retain_active_players(
        players: List[Player],
        court: Optional[CourtBoundary],
        frame_h: int,
    ) -> List[Player]:
        """Keep only genuine players; limit to expected max count."""
        genuine = [
            p for p in players
            if p.role in (PlayerRole.PLAYER_NEAR, PlayerRole.PLAYER_FAR)
        ]
        # Sort by confidence descending; cap at max expected players
        genuine.sort(key=lambda p: p.confidence, reverse=True)
        return genuine[:config.MAX_PLAYERS_DOUBLES]