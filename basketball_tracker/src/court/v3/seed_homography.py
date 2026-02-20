"""
Baseline-scan seed — finds court corners by scanning white mask.

For each candidate baseline pair:
  1. Scan white mask at baseline y to find left/right extent
  2. Map these 4 corners to template corners
  3. Add V-line keypoints for intermediate lines
  4. Compute homography, score by template alignment

Key insight: scanning at the baseline y directly finds where paint
ends, giving accurate corner x-positions without V-line extrapolation.
"""
from __future__ import annotations
from typing import List, Optional, Tuple
import cv2
import numpy as np

from ..v2.line_filter import DetectedLine
from . import template as T
from .scoring import quick_score, line_intersect, in_bounds
from .baseline_scanner import scan_baseline_extent, has_paint, fallback_scan_far_baseline
from .detection_masks import DetectionMasks


def compute_seed(
    h_lines: List[DetectedLine],
    v_lines: List[DetectedLine],
    white_mask: np.ndarray,
    frame_h: int, frame_w: int,
    masks: DetectionMasks = None,
) -> Tuple[Optional[np.ndarray], float]:
    """Try all valid H-line pairs, scan for corners, return best."""
    if len(h_lines) < 2:
        return None, 0.0

    # Build minimal masks if not provided
    if masks is None:
        from .detection_masks import build_white_mask, build_edge_mask
        masks = DetectionMasks(
            white=cv2.dilate(white_mask,
                cv2.getStructuringElement(cv2.MORPH_RECT, (9,9))),
            edge=np.zeros_like(white_mask))

    h_sorted = sorted(h_lines, key=lambda l: l.midpoint[1])

    # Find near baseline (bottommost thick line)
    near_candidates = [l for l in h_sorted
                       if l.midpoint[1] > frame_h * 0.6
                       and has_paint(l, white_mask, frame_h)]
    
    # Augment H-lines with a scan-based far baseline if needed
    augmented_h = list(h_sorted)
    if near_candidates:
        scanned = fallback_scan_far_baseline(
            white_mask, frame_h, frame_w)
        if scanned is not None:
            # Add only if no existing H-line is close to this y
            scanned_y = scanned.midpoint[1]
            has_nearby = any(abs(l.midpoint[1] - scanned_y) < 30
                           for l in h_sorted)
            if not has_nearby:
                augmented_h.append(scanned)
                augmented_h.sort(key=lambda l: l.midpoint[1])

    best_corners = None
    best_score = 0.0

    for fi in range(len(augmented_h)):
        for ni in range(fi + 1, len(augmented_h)):
            far_h = augmented_h[fi]
            near_h = augmented_h[ni]

            if not _valid_pair(far_h, near_h, white_mask, frame_h, frame_w):
                continue

            far_ext = scan_baseline_extent(
                far_h, white_mask, frame_h, frame_w)
            near_ext = scan_baseline_extent(
                near_h, white_mask, frame_h, frame_w)
            if far_ext is None or near_ext is None:
                continue

            mid_h = [augmented_h[j] for j in range(fi + 1, ni)]
            corners = _build_homography(
                far_h, near_h, far_ext, near_ext,
                mid_h, v_lines, frame_h, frame_w)
            if corners is None:
                continue

            score = quick_score(corners, masks, frame_h, frame_w)
            if score > best_score:
                best_score = score
                best_corners = corners

    return best_corners, round(best_score, 4)


def _valid_pair(far, near, white_mask, fh, fw):
    """Must span >=45% of frame, far in top 40%, both lines have paint."""
    gap = near.midpoint[1] - far.midpoint[1]
    if gap < fh * 0.45:
        return False
    if far.length < fw * 0.08 or near.length < fw * 0.08:
        return False
    if near.length < far.length * 0.5:
        return False
    if far.midpoint[1] > fh * 0.40:
        return False
    # Both lines must have real paint (rejects structural edges)
    if not has_paint(far, white_mask, fh):
        return False
    if not has_paint(near, white_mask, fh):
        return False
    return True


def _build_homography(
    far_h, near_h, far_ext, near_ext,
    mid_h, v_lines, fh, fw,
) -> Optional[np.ndarray]:
    """Build homography from baseline endpoints + V-line keypoints."""
    far_y = far_h.midpoint[1]
    near_y = near_h.midpoint[1]
    far_lx, far_rx = far_ext
    near_lx, near_rx = near_ext

    # 4 corner correspondences
    img_pts = [
        (near_lx, near_y),    # BL
        (near_rx, near_y),    # BR
        (far_rx, far_y),      # TR
        (far_lx, far_y),      # TL
    ]
    world_pts = [
        (0.0, 0.0),
        (T.SINGLES_WIDTH, 0.0),
        (T.SINGLES_WIDTH, T.COURT_LENGTH),
        (0.0, T.COURT_LENGTH),
    ]

    # Add V-line keypoints for intermediate lines
    _add_vline_keypoints(
        img_pts, world_pts,
        far_h, near_h, mid_h, v_lines, fh, fw)

    img_arr = np.array(img_pts, dtype=np.float32)
    world_arr = np.array(world_pts, dtype=np.float32)

    H, _ = cv2.findHomography(world_arr, img_arr, cv2.RANSAC, 10.0)
    if H is None:
        return None

    corners_w = T.BOUNDARY_CORNERS.reshape(-1, 1, 2)
    corners_i = cv2.perspectiveTransform(corners_w, H)
    return corners_i.reshape(4, 2).astype(np.float32)


def _add_vline_keypoints(
    img_pts, world_pts,
    far_h, near_h, mid_h, v_lines, fh, fw,
):
    """Add V-line × service-line intersections as extra keypoints."""
    if not v_lines or not mid_h:
        return

    v_sorted = sorted(v_lines, key=lambda l: l.midpoint[0])
    v_groups = _group_v_lines(v_sorted, fw)
    far_y = far_h.midpoint[1]
    near_y = near_h.midpoint[1]

    for svc in mid_h:
        rel = (svc.midpoint[1] - far_y) / max(near_y - far_y, 1)
        if 0.15 < rel < 0.40:
            svc_wy = T.FAR_SERVICE_Y
        elif 0.60 < rel < 0.85:
            svc_wy = T.SERVICE_BOX_DEPTH
        else:
            continue

        for v_line, v_wx in v_groups:
            pt = line_intersect(svc, v_line)
            if pt and in_bounds(pt, fh, fw):
                img_pts.append(pt)
                world_pts.append((v_wx, svc_wy))


def _group_v_lines(v_sorted, fw):
    """Group V-lines into left / center / right."""
    mid_x = fw / 2
    left = [l for l in v_sorted if l.midpoint[0] < mid_x * 0.7]
    right = [l for l in v_sorted if l.midpoint[0] > mid_x * 1.3]
    center = [l for l in v_sorted
              if mid_x * 0.8 < l.midpoint[0] < mid_x * 1.2]

    groups = []
    if left:
        groups.append((max(left, key=lambda l: l.length), 0.0))
    if center:
        groups.append((
            min(center, key=lambda l: abs(l.midpoint[0] - mid_x)),
            T.CENTER_X))
    if right:
        groups.append((max(right, key=lambda l: l.length),
                       T.SINGLES_WIDTH))
    return groups