"""
Court boundary corner construction from classified lines.

Two modes:
  1. Direct: use classified baselines + sidelines → 4 corners
  2. Candidate generation: try multiple line combos → best quad

Mode 1 is used when the classifier is confident.
Mode 2 provides fallback candidates for scoring.
"""
from __future__ import annotations
from typing import List, Optional, Tuple
import cv2
import numpy as np

from .line_filter import DetectedLine
from .line_classifier import ClassifiedLines


def corners_from_classified(
    cl: ClassifiedLines,
    frame_h: int, frame_w: int,
) -> Optional[np.ndarray]:
    """Build 4 corners [TL, TR, BR, BL] from classified lines."""
    far  = cl.far_baseline
    near = cl.near_baseline
    left = cl.left_sideline
    right = cl.right_sideline

    if any(l is None for l in (far, near, left, right)):
        return None

    corners = build_boundary_corners(far, near, left, right)
    if corners is not None and _valid_quad(corners, frame_w, frame_h):
        return corners
    return None


def corners_from_baselines(
    cl: ClassifiedLines,
    frame_h: int, frame_w: int,
) -> Optional[np.ndarray]:
    """
    Build 4 corners using ONLY the two baselines (technique D+B).

    Sidelines are derived by connecting corresponding baseline
    endpoints.  Corners are then computed as line-line intersections
    for mathematical consistency.

    This is more robust than Hough sidelines because:
      - Sidelines are often partially occluded (net, ball boys, chairs)
      - Hough V-lines are short and noisy
      - Baseline paint is thick and RANSAC-refined to sub-pixel accuracy
      - Connecting accurate endpoint pairs gives correct perspective slope
    """
    far  = cl.far_baseline
    near = cl.near_baseline

    if far is None or near is None:
        return None

    # Ensure left endpoint has smaller x (order endpoints)
    far  = _order_lr(far)
    near = _order_lr(near)

    # Derive sidelines from corresponding baseline endpoints
    #   left  sideline: far left end  → near left end
    #   right sideline: far right end → near right end
    left  = DetectedLine(far.x1, far.y1, near.x1, near.y1)
    right = DetectedLine(far.x2, far.y2, near.x2, near.y2)

    # Corners via line-line intersection (technique B)
    # This clips baselines at the exact sideline crossing point,
    # even if baseline paint extends past the corner.
    corners = build_boundary_corners(far, near, left, right)
    if corners is not None and _valid_quad(corners, frame_w, frame_h):
        return corners
    return None


def _order_lr(line: DetectedLine) -> DetectedLine:
    """Ensure x1 <= x2 (left endpoint first)."""
    if line.x1 <= line.x2:
        return line
    return DetectedLine(line.x2, line.y2, line.x1, line.y1)


def generate_candidates(
    h_lines: List[DetectedLine],
    v_lines: List[DetectedLine],
    frame_h: int, frame_w: int,
) -> List[np.ndarray]:
    """
    Generate boundary candidates from line combinations.
    Returns a list of valid corner arrays [TL, TR, BR, BL].
    """
    mid_y = frame_h * 0.55

    # Baseline candidates: long H lines
    far_opts = sorted(
        [l for l in h_lines
         if l.length >= frame_w * 0.12 and l.midpoint[1] < mid_y],
        key=lambda l: l.midpoint[1])[:3]

    near_opts = sorted(
        [l for l in h_lines
         if l.length >= frame_w * 0.12 and l.midpoint[1] >= mid_y],
        key=lambda l: l.midpoint[1], reverse=True)[:2]

    near_fallback = _synth_near(frame_h, frame_w)
    near_opts.append(near_fallback)

    # Sideline candidates: V lines spanning enough height
    side_cands = []
    if far_opts:
        far_y = far_opts[0].midpoint[1]
        court_h = (frame_h * 0.82) - far_y
        min_span = court_h * 0.20  # 20% of estimated court height
        for l in v_lines:
            span = abs(l.y2 - l.y1)
            mx = l.midpoint[0]
            if span >= min_span and frame_w * 0.02 < mx < frame_w * 0.98:
                side_cands.append(l)

    side_cands.sort(key=lambda l: l.midpoint[0])
    left_opts = side_cands[:2] if side_cands else []
    right_opts = side_cands[-2:] if len(side_cands) >= 2 else []

    # Build all valid combinations
    candidates = []
    for far in far_opts:
        for near in near_opts:
            if far.midpoint[1] >= near.midpoint[1] - 30:
                continue
            for left in left_opts:
                for right in right_opts:
                    if left.midpoint[0] >= right.midpoint[0] - 30:
                        continue
                    c = build_boundary_corners(far, near, left, right)
                    if c is not None and _valid_quad(c, frame_w, frame_h):
                        candidates.append(c)
    return candidates


def build_boundary_corners(
    far: DetectedLine, near: DetectedLine,
    left: DetectedLine, right: DetectedLine,
) -> Optional[np.ndarray]:
    """4 corners from boundary lines → [TL, TR, BR, BL]."""
    tl = _intersect(far, left)
    tr = _intersect(far, right)
    br = _intersect(near, right)
    bl = _intersect(near, left)
    if any(p is None for p in (tl, tr, br, bl)):
        return None
    return np.array([tl, tr, br, bl], dtype=np.float32)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _intersect(a: DetectedLine, b: DetectedLine):
    denom = ((a.x1 - a.x2) * (b.y1 - b.y2) -
             (a.y1 - a.y2) * (b.x1 - b.x2))
    if abs(denom) < 1e-6:
        return None
    t = ((a.x1 - b.x1) * (b.y1 - b.y2) -
         (a.y1 - b.y1) * (b.x1 - b.x2)) / denom
    return (a.x1 + t * (a.x2 - a.x1), a.y1 + t * (a.y2 - a.y1))


def _synth_near(fh, fw):
    """Synthetic near baseline at bottom of court zone."""
    y = fh * 0.80
    return DetectedLine(fw * 0.05, y, fw * 0.95, y)


def _valid_quad(corners, W, H):
    tl, tr, br, bl = corners
    # Corners within frame (with margin)
    m = 0.15
    for cx, cy in corners:
        if not (-W * m <= cx <= W * (1 + m) and
                -H * m <= cy <= H * (1 + m)):
            return False
    # Top above bottom
    if tl[1] >= bl[1] - 20 or tr[1] >= br[1] - 20:
        return False
    # Left of right
    if tl[0] >= tr[0] - 20 or bl[0] >= br[0] - 20:
        return False
    # Near side wider than far (perspective)
    nw = np.linalg.norm(np.array(br) - np.array(bl))
    fw_val = np.linalg.norm(np.array(tr) - np.array(tl))
    if nw < fw_val * 0.8:
        return False
    # Minimum height
    if np.linalg.norm(np.array(bl) - np.array(tl)) < H * 0.20:
        return False
    # Reasonable area
    area = float(cv2.contourArea(corners.astype(np.float32)))
    if not (0.08 <= area / (W * H) <= 0.75):
        return False
    return True