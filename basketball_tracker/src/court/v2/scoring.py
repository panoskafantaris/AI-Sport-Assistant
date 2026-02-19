"""
Court boundary candidate scoring.

  1. Edge alignment: template lines match white pixels
  2. Court colour overlap: boundary interior on court surface
  3. Geometric plausibility: perspective, area, centering
  4. Line coverage: fraction of template lines with support

Colour weight is low (10%) since blue court = blue stands at
venues like Qatar.
"""
from __future__ import annotations
import cv2
import numpy as np
from . import template as T


def edge_alignment_score(
    corners: np.ndarray, white_mask: np.ndarray, spacing: int = 4,
) -> float:
    """Fraction of projected template line samples near white pixels."""
    H_mat, _ = cv2.findHomography(corners, T.BOUNDARY_CORNERS, cv2.RANSAC)
    if H_mat is None:
        return 0.0
    try:
        H_inv = np.linalg.inv(H_mat)
    except np.linalg.LinAlgError:
        return 0.0

    fh, fw = white_mask.shape[:2]
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    dilated = cv2.dilate(white_mask, k)

    total, hits = 0, 0
    for line in T.COURT_LINES:
        pts_w = np.array([[[line.x1, line.y1]], [[line.x2, line.y2]]],
                         dtype=np.float32)
        pts_i = cv2.perspectiveTransform(pts_w, H_inv)
        p1, p2 = pts_i[0, 0], pts_i[1, 0]
        n = max(2, int(np.linalg.norm(p2 - p1) / spacing))
        for i in range(n):
            t = i / max(n - 1, 1)
            px = int(p1[0] + t * (p2[0] - p1[0]))
            py = int(p1[1] + t * (p2[1] - p1[1]))
            if 0 <= px < fw and 0 <= py < fh:
                total += 1
                if dilated[py, px] > 0:
                    hits += 1
    return hits / max(total, 1)


def court_colour_overlap_score(
    corners: np.ndarray, colour_mask: np.ndarray,
) -> float:
    """Fraction of boundary interior on court-coloured surface."""
    fh, fw = colour_mask.shape[:2]
    fill = np.zeros((fh, fw), np.uint8)
    cv2.fillPoly(fill, [corners.reshape(-1, 1, 2).astype(np.int32)], 255)
    overlap = cv2.bitwise_and(colour_mask, fill)
    fill_area = max(int(np.sum(fill > 0)), 1)
    ov_area = float(np.sum(overlap > 0))
    # If colour mask is mostly empty, return neutral 0.5
    if np.sum(colour_mask > 0) < fh * fw * 0.02:
        return 0.5
    return ov_area / fill_area


def geometric_score(corners: np.ndarray, H: int, W: int) -> float:
    """Perspective plausibility: perspective ratio, area, centering."""
    tl, tr, br, bl = corners
    scores = []

    # Near wider than far
    nw = float(np.linalg.norm(br - bl))
    fw = float(np.linalg.norm(tr - tl))
    scores.append(min(max((nw / max(fw, 1) - 1.0) / 2.0, 0.0), 1.0))

    # Area 10-55% of frame
    area = float(cv2.contourArea(corners.astype(np.float32)))
    af = area / (W * H)
    scores.append(1.0 if 0.10 <= af <= 0.55
                  else max(0.0, 1.0 - abs(af - 0.30) * 4))

    # Centering
    cx = np.mean(corners[:, 0])
    scores.append(max(0.0, 1.0 - abs(cx - W / 2) / W * 3))

    # Height span 30-80%
    hf = (max(bl[1], br[1]) - min(tl[1], tr[1])) / H
    scores.append(1.0 if 0.30 <= hf <= 0.80
                  else max(0.0, 1.0 - abs(hf - 0.55) * 3))

    return float(np.mean(scores))


def line_coverage_score(
    corners: np.ndarray, white_mask: np.ndarray,
    min_support: float = 0.06,
) -> float:
    """Fraction of template lines with white pixel support."""
    H_mat, _ = cv2.findHomography(corners, T.BOUNDARY_CORNERS, cv2.RANSAC)
    if H_mat is None:
        return 0.0
    try:
        H_inv = np.linalg.inv(H_mat)
    except np.linalg.LinAlgError:
        return 0.0

    fh, fw = white_mask.shape[:2]
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    dilated = cv2.dilate(white_mask, k)

    supported = 0
    for line in T.COURT_LINES:
        pts_w = np.array([[[line.x1, line.y1]], [[line.x2, line.y2]]],
                         dtype=np.float32)
        pts_i = cv2.perspectiveTransform(pts_w, H_inv)
        p1, p2 = pts_i[0, 0], pts_i[1, 0]
        n = max(2, int(np.linalg.norm(p2 - p1) / 5))
        hits = 0
        for i in range(n):
            t = i / max(n - 1, 1)
            px = int(p1[0] + t * (p2[0] - p1[0]))
            py = int(p1[1] + t * (p2[1] - p1[1]))
            if 0 <= px < fw and 0 <= py < fh and dilated[py, px] > 0:
                hits += 1
        if hits / max(n, 1) >= min_support:
            supported += 1

    return supported / max(len(T.COURT_LINES), 1)