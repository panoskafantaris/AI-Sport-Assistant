"""
Template assignment — maps image keypoints to template world points.

Uses vertical ordering and x-position to assign each sampled image
point to the most plausible template keypoint. The court has 4 rows
(far baseline, far service, near service, near baseline) × 3 columns
(left, center, right) = 12 keypoints total.
"""
from __future__ import annotations
from typing import List, Optional
import cv2
import numpy as np

from . import template as T


# Template rows: (world_y, [keypoint_indices])
# Sorted top-of-image first (far = high world_y appears at top)
TEMPLATE_ROWS = [
    (T.COURT_LENGTH,      [9, 10, 11]),   # far baseline
    (T.FAR_SERVICE_Y,     [6, 7, 8]),     # far service
    (T.SERVICE_BOX_DEPTH, [3, 4, 5]),     # near service
    (0.0,                  [0, 1, 2]),     # near baseline
]


def assign_template_points(
    sample_img: np.ndarray,    # (4, 2) image points
    world_pts: np.ndarray,     # (12, 2) all template points
    frame_h: int,
    frame_w: int,
) -> Optional[np.ndarray]:
    """
    Assign 4 sampled image points to template keypoints.

    Uses vertical ordering: image y increases → world y decreases
    (far baseline is at top of image, near baseline at bottom).

    Returns (4, 2) world coordinates or None if invalid.
    """
    # Sort by image y (top to bottom)
    order = np.argsort(sample_img[:, 1])
    sorted_img = sample_img[order]

    # Normalize y to [0, 1]
    y_norm = sorted_img[:, 1] / frame_h

    assigned_world = np.zeros((4, 2), dtype=np.float32)
    used_rows = set()

    for i in range(4):
        px, py = sorted_img[i]
        x_norm = px / frame_w

        # Pick best unused row
        best_row = None
        best_dist = float('inf')
        for ri, (row_y, _) in enumerate(TEMPLATE_ROWS):
            if ri in used_rows:
                continue
            expected_y_norm = 1.0 - row_y / T.COURT_LENGTH
            dist = abs(y_norm[i] - expected_y_norm)
            if dist < best_dist:
                best_dist = dist
                best_row = ri

        if best_row is None:
            return None
        used_rows.add(best_row)

        _, kp_indices = TEMPLATE_ROWS[best_row]
        col_idx = _assign_column(x_norm, kp_indices)
        assigned_world[i] = world_pts[col_idx]

    if not _valid_assignment(sorted_img, assigned_world):
        return None

    return assigned_world


def _assign_column(x_norm: float, kp_indices: List[int]) -> int:
    """Pick left (idx 0), center (idx 1), or right (idx 2)."""
    if x_norm < 0.35:
        return kp_indices[0]
    elif x_norm > 0.65:
        return kp_indices[2]
    else:
        return kp_indices[1]


def _valid_assignment(
    img_pts: np.ndarray,
    world_pts: np.ndarray,
) -> bool:
    """Check 4 correspondences form a valid (non-degenerate) homography."""
    unique_y = len(set(round(wy, 1) for _, wy in world_pts))
    if unique_y < 2:
        return False
    area = cv2.contourArea(world_pts.astype(np.float32))
    return area > 1.0