"""
Confidence scoring — multi-modal line alignment.

Each line uses the appropriate detection mask:
  - Paint lines (baselines, sidelines, center) → white mask
  - Net → Canny edge mask

Boundary lines weighted 3x, center service 1x, net 2x.
"""
from __future__ import annotations
import cv2
import numpy as np

from . import template as T
from .detection_masks import DetectionMasks

_LINE_WEIGHTS = {
    "near_baseline":  3.0,
    "far_baseline":   3.0,
    "left_sideline":  3.0,
    "right_sideline": 3.0,
    "center_service": 1.0,
    "net":            2.0,
}


def compute_confidence(
    corners: np.ndarray,
    masks: DetectionMasks,
    frame_h: int,
    frame_w: int,
) -> float:
    """Overall confidence: 70% edge alignment + 30% geometry."""
    ea = edge_alignment(corners, masks)
    geo = geometric_plausibility(corners, frame_h, frame_w)
    return round(0.70 * ea + 0.30 * geo, 4)


def edge_alignment(
    corners: np.ndarray,
    masks: DetectionMasks,
    spacing: int = 4,
) -> float:
    """Weighted alignment using per-line mask selection."""
    H_mat, _ = cv2.findHomography(T.BOUNDARY_CORNERS, corners, 0)
    if H_mat is None:
        return 0.0

    total_w, hits_w = 0.0, 0.0
    for line in T.SCORED_LINES:
        w = _LINE_WEIGHTS.get(line.name, 1.0)
        mask = masks.get(line.detect_mode)
        frac = _line_alignment(line, H_mat, mask, spacing)
        total_w += w
        hits_w += w * frac

    return hits_w / max(total_w, 1e-6)


def _line_alignment(line, H_mat, mask, spacing):
    """Fraction of projected line samples hitting the mask."""
    fh, fw = mask.shape[:2]
    pts_w = np.array(
        [[[line.x1, line.y1]], [[line.x2, line.y2]]],
        dtype=np.float32)
    pts_i = cv2.perspectiveTransform(pts_w, H_mat)
    p1, p2 = pts_i[0, 0], pts_i[1, 0]

    n = max(2, int(np.linalg.norm(p2 - p1) / spacing))
    hits, total = 0, 0
    for i in range(n):
        t = i / max(n - 1, 1)
        px = int(p1[0] + t * (p2[0] - p1[0]))
        py = int(p1[1] + t * (p2[1] - p1[1]))
        if 0 <= px < fw and 0 <= py < fh:
            total += 1
            if mask[py, px] > 0:
                hits += 1
    return hits / max(total, 1)


def geometric_plausibility(
    corners: np.ndarray, H: int, W: int,
) -> float:
    """Perspective plausibility score (0-1)."""
    bl, br, tr, tl = corners
    scores = []

    # Near side wider than far (perspective)
    near_w = float(np.linalg.norm(br - bl))
    far_w = float(np.linalg.norm(tr - tl))
    ratio = near_w / max(far_w, 1)
    scores.append(min(max((ratio - 1.0) / 2.0, 0.0), 1.0))

    # Area 10-55% of frame
    area = float(cv2.contourArea(corners.astype(np.float32)))
    area_frac = area / (W * H)
    if 0.10 <= area_frac <= 0.55:
        scores.append(1.0)
    else:
        scores.append(max(0.0, 1.0 - abs(area_frac - 0.30) * 4))

    # Horizontally centered
    cx = float(np.mean(corners[:, 0]))
    scores.append(max(0.0, 1.0 - abs(cx - W / 2) / W * 3))

    # Vertical span 30-80%
    top_y = min(tl[1], tr[1])
    bot_y = max(bl[1], br[1])
    height_frac = (bot_y - top_y) / H
    if 0.30 <= height_frac <= 0.80:
        scores.append(1.0)
    else:
        scores.append(max(0.0, 1.0 - abs(height_frac - 0.55) * 3))

    return float(np.mean(scores))


def per_line_alignment(
    corners: np.ndarray,
    masks: DetectionMasks,
    spacing: int = 5,
) -> dict:
    """Per-line alignment breakdown (for debugging)."""
    H_mat, _ = cv2.findHomography(T.BOUNDARY_CORNERS, corners, 0)
    if H_mat is None:
        return {}

    result = {}
    for line in T.SCORED_LINES:
        mask = masks.get(line.detect_mode)
        result[line.name] = round(
            _line_alignment(line, H_mat, mask, spacing), 3)
    return result