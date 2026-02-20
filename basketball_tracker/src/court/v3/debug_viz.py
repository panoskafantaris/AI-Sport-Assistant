"""
Debug visualization for Court Detection V3.

Draws:
  - Raw H-lines (red) and V-lines (blue) from Hough
  - Projected template lines (green/orange/magenta)
  - Boundary polygon with corner labels
"""
from __future__ import annotations
from typing import List, Optional
import cv2
import numpy as np

from ..v2.line_filter import DetectedLine
from . import template as T

_CORNER_LABELS = ["BL", "BR", "TR", "TL"]
_CORNER_COLOURS = [
    (255, 0, 200),    # BL magenta
    (255, 100, 0),    # BR blue
    (0, 200, 255),    # TR yellow
    (0, 255, 0),      # TL green
]

_TMPL_COLOURS = {
    "near_baseline":  (0, 0, 255),
    "far_baseline":   (0, 0, 255),
    "near_service":   (0, 180, 255),
    "far_service":    (0, 180, 255),
    "left_sideline":  (0, 255, 100),
    "right_sideline": (0, 255, 100),
    "center_service": (0, 255, 200),
    "net":            (255, 0, 255),
}


def draw_result(
    frame: np.ndarray,
    corners: Optional[np.ndarray],
    confidence: float,
    frame_number: int,
    h_lines: Optional[List[DetectedLine]] = None,
    v_lines: Optional[List[DetectedLine]] = None,
    line_scores: Optional[dict] = None,
) -> np.ndarray:
    """Draw detection result on frame."""
    out = frame.copy()
    H, W = out.shape[:2]

    # Draw raw lines (thin)
    if h_lines:
        for ln in h_lines:
            cv2.line(out, (int(ln.x1), int(ln.y1)),
                     (int(ln.x2), int(ln.y2)), (0, 0, 180), 1)
    if v_lines:
        for ln in v_lines:
            cv2.line(out, (int(ln.x1), int(ln.y1)),
                     (int(ln.x2), int(ln.y2)), (180, 0, 0), 1)

    # Draw projected template + boundary
    if corners is not None:
        _draw_template_lines(out, corners, line_scores)
        _draw_boundary(out, corners)

    # Status bar
    cv2.rectangle(out, (0, 0), (W, 40), (20, 20, 20), -1)
    if corners is not None:
        status = f"DETECTED conf={confidence:.3f}"
        color = (0, 255, 0)
    else:
        status = "FAILED"
        color = (0, 0, 255)
    cv2.putText(out, f"Frame {frame_number}  |  {status}",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return out


def _draw_template_lines(
    frame: np.ndarray,
    corners: np.ndarray,
    line_scores: Optional[dict] = None,
) -> None:
    """Project and draw all template lines."""
    H_mat, _ = cv2.findHomography(T.BOUNDARY_CORNERS, corners, 0)
    if H_mat is None:
        return

    for line in T.SCORED_LINES:
        pts_w = np.array(
            [[[line.x1, line.y1]], [[line.x2, line.y2]]],
            dtype=np.float32)
        pts_i = cv2.perspectiveTransform(pts_w, H_mat)
        p1 = tuple(pts_i[0, 0].astype(int))
        p2 = tuple(pts_i[1, 0].astype(int))

        colour = _TMPL_COLOURS.get(line.name, (200, 200, 200))
        thick = 3 if "baseline" in line.name else 2
        cv2.line(frame, p1, p2, colour, thick)

        mx = (p1[0] + p2[0]) // 2
        my = (p1[1] + p2[1]) // 2
        label = line.name.replace("_", " ")
        if line_scores and line.name in line_scores:
            label += f" {line_scores[line.name]:.0%}"
        cv2.putText(frame, label, (mx - 40, my - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, colour, 1)


def _draw_boundary(frame: np.ndarray, corners: np.ndarray) -> None:
    pts = corners.reshape(-1, 1, 2).astype(np.int32)
    cv2.polylines(frame, [pts], True, (0, 255, 255), 3)

    for i, (cx, cy) in enumerate(corners):
        cv2.circle(frame, (int(cx), int(cy)), 10,
                   _CORNER_COLOURS[i], -1)
        cv2.circle(frame, (int(cx), int(cy)), 10,
                   (255, 255, 255), 2)
        cv2.putText(frame, _CORNER_LABELS[i],
                    (int(cx) + 14, int(cy) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    _CORNER_COLOURS[i], 2)