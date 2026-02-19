"""
Phase 4 – Player kinesiology via YOLOv8-pose.

Extracts skeleton keypoints per player and derives:
  - Joint angles (elbow, knee, shoulder)
  - Stroke classification (forehand, backhand, serve, overhead)
  - Stance width and balance indicator
"""
from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import numpy as np

from ..models.player import Player
import config

# Lazy import to avoid loading pose model unless used
_pose_model = None


def _get_pose_model():
    global _pose_model
    if _pose_model is None:
        from ultralytics import YOLO
        _pose_model = YOLO(config.POSE_MODEL)
    return _pose_model


# COCO pose keypoint indices
KP = {
    "nose":        0,
    "l_shoulder": 5,  "r_shoulder": 6,
    "l_elbow":    7,  "r_elbow":    8,
    "l_wrist":    9,  "r_wrist":   10,
    "l_hip":     11,  "r_hip":     12,
    "l_knee":    13,  "r_knee":    14,
    "l_ankle":   15,  "r_ankle":   16,
}


class KinesiologyAnalyser:
    """
    Runs pose estimation on player crops and extracts biomechanical features.
    """

    def __init__(self, enabled: bool = True):
        self._enabled = enabled

    def analyse(self, frame: np.ndarray, players: List[Player]) -> List[Player]:
        """Enrich players with pose keypoints and stroke features."""
        if not self._enabled:
            return players

        model = _get_pose_model()
        H, W  = frame.shape[:2]

        for p in players:
            x1, y1, x2, y2 = p.bbox.to_int_tuple()
            # Pad crop slightly
            pad = 10
            crop = frame[
                max(0, y1 - pad):min(H, y2 + pad),
                max(0, x1 - pad):min(W, x2 + pad),
            ]
            if crop.size == 0:
                continue

            results = model.predict(crop, verbose=False, conf=0.3)
            if not results or results[0].keypoints is None:
                continue

            kpts = results[0].keypoints.xy.cpu().numpy()  # (N, 17, 2)
            if len(kpts) == 0:
                continue

            kp = kpts[0]  # first (best) person in crop
            p.pose_keypoints = kp
            # Store derived features in extras via player attr
            p.__dict__["stroke"]       = self._classify_stroke(kp)
            p.__dict__["stance_width"] = self._stance_width(kp, p.bbox.height)
            p.__dict__["knee_bend"]    = self._joint_angle(kp, "l_hip", "l_knee", "l_ankle")

        return players

    # ── Biomechanics helpers ──────────────────────────────────────────────────

    @staticmethod
    def _joint_angle(kp: np.ndarray, a: str, b: str, c: str) -> float:
        """Angle at joint b in degrees (0-180)."""
        pa = kp[KP[a]]
        pb = kp[KP[b]]
        pc = kp[KP[c]]
        if np.any(pa == 0) or np.any(pb == 0) or np.any(pc == 0):
            return 0.0
        v1 = pa - pb
        v2 = pc - pb
        cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        return float(np.degrees(np.arccos(np.clip(cos_a, -1, 1))))

    @staticmethod
    def _stance_width(kp: np.ndarray, bbox_h: float) -> float:
        """Normalised ankle distance (0-1 relative to bbox height)."""
        la = kp[KP["l_ankle"]]
        ra = kp[KP["r_ankle"]]
        if np.any(la == 0) or np.any(ra == 0):
            return 0.0
        return float(np.linalg.norm(la - ra)) / (bbox_h + 1e-6)

    @staticmethod
    def _classify_stroke(kp: np.ndarray) -> str:
        """
        Heuristic stroke classification based on wrist / shoulder geometry.
        """
        lw = kp[KP["l_wrist"]]
        rw = kp[KP["r_wrist"]]
        ls = kp[KP["l_shoulder"]]
        rs = kp[KP["r_shoulder"]]

        if np.any(rw == 0) or np.any(lw == 0):
            return "unknown"

        # Dominant wrist above shoulder → serve or overhead
        r_above = rw[1] < rs[1]
        l_above = lw[1] < ls[1]
        if r_above and l_above:
            return "serve_or_overhead"

        # Dominant wrist crossed to other side → backhand
        if rw[0] < ls[0]:
            return "backhand"

        # Dominant wrist to same side → forehand
        if rw[0] > rs[0]:
            return "forehand"

        return "neutral"