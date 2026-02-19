"""
Court line detection + clustering.

Span-based merge: computes full extent across all fragments
in a cluster, correctly handling lines broken by player occlusion.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
import cv2
import numpy as np

from .roi import build_court_trapezoid, extract_white_lines


@dataclass
class DetectedLine:
    """A line segment in image space."""
    x1: float; y1: float; x2: float; y2: float

    @property
    def midpoint(self) -> Tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    @property
    def length(self) -> float:
        return float(np.hypot(self.x2 - self.x1, self.y2 - self.y1))

    @property
    def angle_deg(self) -> float:
        return float(np.degrees(
            np.arctan2(self.y2 - self.y1, self.x2 - self.x1)))

    @property
    def is_roughly_horizontal(self) -> bool:
        a = abs(self.angle_deg) % 180
        return a < 35 or a > 145

    @property
    def is_roughly_vertical(self) -> bool:
        a = abs(self.angle_deg) % 180
        return 55 < a < 125

    @property
    def top_y(self) -> float:
        return min(self.y1, self.y2)

    @property
    def bot_y(self) -> float:
        return max(self.y1, self.y2)


class LineDetector:
    """Detects court lines with spatial ROI + Hough."""

    def __init__(self, white_thresh=180, min_len=20):
        self.white_thresh = white_thresh
        self.min_len = min_len

    def detect(self, frame: np.ndarray
               ) -> Tuple[List[DetectedLine], np.ndarray, np.ndarray]:
        H, W = frame.shape[:2]
        roi = build_court_trapezoid(H, W)
        white = extract_white_lines(frame, roi, self.white_thresh)
        edges = cv2.Canny(cv2.GaussianBlur(white, (3, 3), 0), 30, 100)
        lines = self._hough(edges, W, H)
        return lines, roi, white

    def _hough(self, edges, W, H) -> List[DetectedLine]:
        raw = cv2.HoughLinesP(
            edges, rho=1, theta=np.deg2rad(1),
            threshold=18,
            minLineLength=self.min_len,
            maxLineGap=80,    # bridge player bodies (60-80px wide)
        )
        if raw is None:
            return []
        out = []
        for x1, y1, x2, y2 in raw[:, 0]:
            ln = DetectedLine(float(x1), float(y1), float(x2), float(y2))
            if ln.is_roughly_horizontal or ln.is_roughly_vertical:
                out.append(ln)
        return out


def cluster_lines(
    lines: List[DetectedLine], cluster_dist: float = 35.0,
) -> Tuple[List[DetectedLine], List[DetectedLine]]:
    """Split into H/V groups and merge nearby parallels."""
    h = [l for l in lines if l.is_roughly_horizontal]
    v = [l for l in lines if l.is_roughly_vertical]
    return _merge(h, "h", cluster_dist), _merge(v, "v", cluster_dist)


def _merge(lines, axis, dist):
    if not lines:
        return []
    key = ((lambda l: l.midpoint[1]) if axis == "h"
           else (lambda l: l.midpoint[0]))
    lines = sorted(lines, key=key)
    merged, cluster = [], [lines[0]]
    for ln in lines[1:]:
        if abs(key(ln) - key(cluster[-1])) < dist:
            cluster.append(ln)
        else:
            merged.append(_span_merge(cluster, axis))
            cluster = [ln]
    merged.append(_span_merge(cluster, axis))
    return merged


def _span_merge(cluster, axis):
    """
    Merge using FULL SPAN across all fragments.
    Two 200px fragments 100px apart become one 500px line.
    """
    all_pts = []
    for l in cluster:
        all_pts.append((l.x1, l.y1))
        all_pts.append((l.x2, l.y2))
    pts = np.array(all_pts)

    if axis == "h":
        i_min, i_max = np.argmin(pts[:, 0]), np.argmax(pts[:, 0])
    else:
        i_min, i_max = np.argmin(pts[:, 1]), np.argmax(pts[:, 1])

    p1, p2 = pts[i_min], pts[i_max]
    return DetectedLine(float(p1[0]), float(p1[1]),
                        float(p2[0]), float(p2[1]))