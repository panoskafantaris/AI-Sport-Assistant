"""
Debug visualization — 4-panel composite with line classification.

  A: Frame with ROI trapezoid overlay
  B: White line mask
  C: Classified lines (colour-coded by role)
  D: Best boundary + projected template
"""
from __future__ import annotations
from pathlib import Path
from typing import List, Optional
import cv2
import numpy as np

from . import template as T
from .line_classifier import ClassifiedLines

# Colour key for classified lines
_ROLE_COLOURS = {
    "far_baseline":   (0, 0, 255),     # red
    "near_baseline":  (0, 0, 255),     # red
    "far_service":    (0, 180, 255),   # orange
    "near_service":   (0, 180, 255),   # orange
    "net":            (255, 0, 255),   # magenta (distinctive!)
    "left_sideline":  (255, 100, 0),   # blue
    "right_sideline": (255, 100, 0),   # blue
    "center_service": (200, 200, 0),   # cyan
}


def save_debug_composite(
    frame: np.ndarray,
    white_mask: np.ndarray,
    roi_mask: np.ndarray,
    lines, h_lines, v_lines,
    cl: Optional[ClassifiedLines],
    best_corners: Optional[np.ndarray],
    best_score: float,
    frame_number: int,
    debug_dir: Path,
):
    H, W = frame.shape[:2]
    ph, pw = H // 2, W // 2

    def rsz(img):
        return cv2.resize(img, (pw, ph))

    def bgr(g):
        return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)

    # Panel A: ROI overlay
    A = frame.copy()
    tint = np.zeros_like(A)
    tint[roi_mask == 0] = (0, 0, 80)
    tint[roi_mask > 0] = (0, 40, 0)
    A = cv2.addWeighted(A, 0.7, tint, 0.3, 0)
    cv2.putText(A, "A: ROI (red=excluded, green=included)",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

    # Panel B: white mask
    B = bgr(white_mask.copy())
    n_white = np.sum(white_mask > 0)
    cv2.putText(B, f"B: White lines ({n_white} px)",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

    # Panel C: classified lines
    C = frame.copy()
    # Draw raw lines faintly
    for ln in lines:
        cv2.line(C, (int(ln.x1), int(ln.y1)),
                 (int(ln.x2), int(ln.y2)), (80, 80, 80), 1)
    # Draw classified lines with colours
    if cl:
        _draw_classified(C, cl)
    else:
        for ln in h_lines:
            cv2.line(C, (int(ln.x1), int(ln.y1)),
                     (int(ln.x2), int(ln.y2)), (0, 220, 0), 2)
        for ln in v_lines:
            cv2.line(C, (int(ln.x1), int(ln.y1)),
                     (int(ln.x2), int(ln.y2)), (255, 80, 0), 2)
    cv2.putText(C, f"C: Classified lines  H={len(h_lines)} V={len(v_lines)}",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

    # Panel D: result
    D = frame.copy()
    if best_corners is not None:
        pts = best_corners.reshape(-1, 1, 2).astype(np.int32)
        cv2.polylines(D, [pts], True, (0, 255, 255), 3)
        for i, (cx, cy) in enumerate(best_corners):
            cv2.circle(D, (int(cx), int(cy)), 9, (0, 0, 255), -1)
            cv2.putText(D, ["TL", "TR", "BR", "BL"][i],
                        (int(cx) + 10, int(cy) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        _draw_template(D, best_corners)
        cv2.putText(D, f"D: conf={best_score:.3f}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
    else:
        cv2.putText(D, "D: FAILED",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)

    panel = np.vstack([np.hstack([rsz(A), rsz(B)]),
                       np.hstack([rsz(C), rsz(D)])])
    path = debug_dir / f"{frame_number:06d}_v2_composite.jpg"
    cv2.imwrite(str(path), panel)


def _draw_classified(frame, cl: ClassifiedLines):
    """Draw each classified line with its role colour and label."""
    for role_name in ("far_baseline", "far_service", "net",
                      "near_service", "near_baseline",
                      "left_sideline", "right_sideline", "center_service"):
        ln = getattr(cl, role_name)
        if ln is None:
            continue
        colour = _ROLE_COLOURS.get(role_name, (200, 200, 200))
        thick = 3 if role_name == "net" else 2
        cv2.line(frame, (int(ln.x1), int(ln.y1)),
                 (int(ln.x2), int(ln.y2)), colour, thick)
        mx, my = ln.midpoint
        label = role_name.replace("_", " ")
        cv2.putText(frame, label, (int(mx) - 40, int(my) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1)


def _draw_template(frame, corners):
    H_mat, _ = cv2.findHomography(corners, T.BOUNDARY_CORNERS, cv2.RANSAC)
    if H_mat is None:
        return
    try:
        H_inv = np.linalg.inv(H_mat)
    except np.linalg.LinAlgError:
        return
    for line in T.COURT_LINES:
        pts_w = np.array([[[line.x1, line.y1]], [[line.x2, line.y2]]],
                         dtype=np.float32)
        pts_i = cv2.perspectiveTransform(pts_w, H_inv)
        p1, p2 = tuple(pts_i[0, 0].astype(int)), \
                  tuple(pts_i[1, 0].astype(int))
        is_bndry = "baseline" in line.name or "sideline" in line.name
        is_net = "net" in line.name
        colour = ((255, 0, 255) if is_net else
                  (0, 255, 100) if is_bndry else (100, 255, 100))
        cv2.line(frame, p1, p2, colour, 2 if (is_bndry or is_net) else 1)