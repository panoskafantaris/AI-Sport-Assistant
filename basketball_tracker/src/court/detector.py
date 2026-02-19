"""
Phase 1 – Court line detector.

Strategy for a STABLE tennis camera:
1. Convert to grayscale → Canny edges.
2. Probabilistic Hough to find all line segments.
3. Cluster lines by angle → horizontals (baselines, service lines, net)
   and verticals/diagonals (sidelines, centre service line).
4. Find intersection points to locate the 4 outer corners + inner lines.
5. Expose CourtBoundary with high-confidence polygon.
"""
from __future__ import annotations
from typing import List, Optional, Tuple
import cv2
import numpy as np

import sys
sys.path.insert(0, str(__file__).rsplit("/src", 1)[0])

from ..models.court import CourtBoundary, CourtLine
import config


class CourtLineDetector:
    """Detects tennis court lines and builds a precise boundary polygon."""

    def __init__(self):
        self._last_boundary: Optional[CourtBoundary] = None

    # ── Public API ─────────────────────────────────────────────────────────────

    def detect(self, frame: np.ndarray) -> Optional[CourtBoundary]:
        """Run detection on a single frame. Returns CourtBoundary or None."""
        lines = self._hough_lines(frame)
        if not lines:
            return self._last_boundary  # return cached on failure

        h_lines, v_lines = self._cluster_by_angle(lines)
        if len(h_lines) < 2 or len(v_lines) < 2:
            return self._last_boundary

        boundary = self._build_boundary(frame, h_lines, v_lines)
        if boundary and boundary.is_valid():
            self._last_boundary = boundary
        return self._last_boundary

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _hough_lines(self, frame: np.ndarray) -> List[CourtLine]:
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur  = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150, apertureSize=3)

        raw = cv2.HoughLinesP(
            edges,
            rho=config.HOUGH_RHO,
            theta=np.deg2rad(config.HOUGH_THETA),
            threshold=config.HOUGH_THRESHOLD,
            minLineLength=config.HOUGH_MIN_LENGTH,
            maxLineGap=config.HOUGH_MAX_GAP,
        )
        if raw is None:
            return []
        return [CourtLine(x1, y1, x2, y2) for x1, y1, x2, y2 in raw[:, 0]]

    def _cluster_by_angle(
        self, lines: List[CourtLine]
    ) -> Tuple[List[CourtLine], List[CourtLine]]:
        """
        Separate into near-horizontal (court width lines) and
        near-vertical (court length lines).
        """
        h_lines, v_lines = [], []
        for ln in lines:
            ang = abs(ln.angle_deg) % 180
            if ang < 30 or ang > 150:        # near-horizontal
                h_lines.append(ln)
            elif 60 < ang < 120:             # near-vertical
                v_lines.append(ln)
        h_lines = self._merge_close_lines(h_lines, axis="h")
        v_lines = self._merge_close_lines(v_lines, axis="v")
        return h_lines, v_lines

    def _merge_close_lines(
        self, lines: List[CourtLine], axis: str
    ) -> List[CourtLine]:
        """Merge nearly-parallel, close lines into representative lines."""
        if not lines:
            return []
        # Sort by mid-y (horizontal) or mid-x (vertical)
        key = (lambda ln: ln.midpoint[1]) if axis == "h" else (lambda ln: ln.midpoint[0])
        lines = sorted(lines, key=key)
        merged: List[CourtLine] = []
        cluster: List[CourtLine] = [lines[0]]

        for ln in lines[1:]:
            ref = cluster[-1]
            dist = abs(key(ln) - key(ref))
            angle_diff = abs(ln.angle_deg - ref.angle_deg)
            if dist < config.LINE_CLUSTER_DIST and angle_diff < config.LINE_CLUSTER_ANGLE:
                cluster.append(ln)
            else:
                merged.append(self._average_line(cluster))
                cluster = [ln]
        merged.append(self._average_line(cluster))
        return merged

    @staticmethod
    def _average_line(cluster: List[CourtLine]) -> CourtLine:
        xs = [(ln.x1 + ln.x2) / 2 for ln in cluster]
        ys = [(ln.y1 + ln.y2) / 2 for ln in cluster]
        ang = np.mean([ln.angle_deg for ln in cluster])
        # Extend to max length of cluster members
        L = max(ln.length for ln in cluster)
        cx, cy = np.mean(xs), np.mean(ys)
        dx = np.cos(np.deg2rad(ang)) * L / 2
        dy = np.sin(np.deg2rad(ang)) * L / 2
        return CourtLine(cx - dx, cy - dy, cx + dx, cy + dy)

    def _build_boundary(
        self,
        frame: np.ndarray,
        h_lines: List[CourtLine],
        v_lines: List[CourtLine],
    ) -> Optional[CourtBoundary]:
        """
        Identify the 4 outermost court corners from line intersections.
        """
        H, W = frame.shape[:2]
        # Find all intersections between horizontal and vertical lines
        pts = []
        for hl in h_lines:
            for vl in v_lines:
                pt = self._intersect(hl, vl)
                if pt and 0 <= pt[0] <= W and 0 <= pt[1] <= H:
                    pts.append(pt)

        if len(pts) < 4:
            return None

        pts = np.array(pts, dtype=np.float32)
        # Keep extreme corners: top-left, top-right, bottom-right, bottom-left
        corners = self._four_corners(pts)
        return CourtBoundary(corners=corners, confidence=0.9)

    @staticmethod
    def _intersect(
        a: CourtLine, b: CourtLine
    ) -> Optional[Tuple[float, float]]:
        """Line-line intersection using Cramer's rule."""
        x1, y1, x2, y2 = a.x1, a.y1, a.x2, a.y2
        x3, y3, x4, y4 = b.x1, b.y1, b.x2, b.y2
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denom) < 1e-6:
            return None
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        x = x1 + t * (x2 - x1)
        y = y1 + t * (y2 - y1)
        return (x, y)

    @staticmethod
    def _four_corners(pts: np.ndarray) -> np.ndarray:
        """
        Pick the 4 outer-most corners: TL, TR, BR, BL ordering.
        Uses sum/diff trick then convex hull extremes.
        """
        s = pts.sum(axis=1)
        d = np.diff(pts, axis=1).flatten()
        tl = pts[np.argmin(s)]
        br = pts[np.argmax(s)]
        tr = pts[np.argmin(d)]
        bl = pts[np.argmax(d)]
        return np.array([tl, tr, br, bl], dtype=np.float32)