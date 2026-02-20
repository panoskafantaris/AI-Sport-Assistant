"""
Scoring functions for seed evaluation and line intersection.

quick_score: weighted alignment of projected template lines.
Uses DetectionMasks for multi-modal detection (white + edge).
"""
from __future__ import annotations
from typing import Optional, Tuple
import cv2
import numpy as np

from ..v2.line_filter import DetectedLine
from . import template as T
from .detection_masks import DetectionMasks

_WEIGHTS = {
    "near_baseline":  1.0,
    "far_baseline":   3.0,
    "left_sideline":  1.0,
    "right_sideline": 1.0,
    "center_service": 0.5,
    "net":            1.0,
}


def quick_score(
    corners: np.ndarray,
    masks: DetectionMasks,
    fh: int, fw: int,
    spacing: int = 5,
) -> float:
    """Template alignment score with per-line mask + weighting."""
    H_mat, _ = cv2.findHomography(T.BOUNDARY_CORNERS, corners, 0)
    if H_mat is None:
        return 0.0

    total_w, hits_w = 0.0, 0.0
    for line in T.SCORED_LINES:
        w = _WEIGHTS.get(line.name, 1.0)
        mask = masks.get(line.detect_mode)
        pts_w = np.array(
            [[[line.x1, line.y1]], [[line.x2, line.y2]]],
            dtype=np.float32)
        pts_i = cv2.perspectiveTransform(pts_w, H_mat)
        p1, p2 = pts_i[0, 0], pts_i[1, 0]
        n = max(2, int(np.linalg.norm(p2 - p1) / spacing))
        line_hits, line_total = 0, 0
        for j in range(n):
            t = j / max(n - 1, 1)
            px = int(p1[0] + t * (p2[0] - p1[0]))
            py = int(p1[1] + t * (p2[1] - p1[1]))
            if 0 <= px < fw and 0 <= py < fh:
                line_total += 1
                if mask[py, px] > 0:
                    line_hits += 1
        if line_total > 0:
            total_w += w
            hits_w += w * (line_hits / line_total)

    return hits_w / max(total_w, 1e-6)


def line_intersect(
    a: DetectedLine, b: DetectedLine,
) -> Optional[Tuple[float, float]]:
    """Intersection of two infinite lines."""
    denom = ((a.x1 - a.x2) * (b.y1 - b.y2) -
             (a.y1 - a.y2) * (b.x1 - b.x2))
    if abs(denom) < 1e-6:
        return None
    t = ((a.x1 - b.x1) * (b.y1 - b.y2) -
         (a.y1 - b.y1) * (b.x1 - b.x2)) / denom
    return (a.x1 + t * (a.x2 - a.x1), a.y1 + t * (a.y2 - a.y1))


def in_bounds(pt, fh, fw, margin=0.25):
    x, y = pt
    return (-fw*margin <= x <= fw*(1+margin) and
            -fh*margin <= y <= fh*(1+margin))