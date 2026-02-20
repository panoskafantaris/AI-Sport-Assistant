"""
Keypoint detector — extracts candidate court keypoints.

Strategy:
  1. Reuse V2 line detection (ROI + white mask + Hough + clustering)
  2. Compute ALL pairwise H×V intersections
  3. Filter: must be in-frame, near white pixels, not on net mesh
  4. Return candidate keypoints for RANSAC matching

Each candidate is a potential match to one of the 12 template keypoints.
"""
from __future__ import annotations
from typing import List, Tuple
import cv2
import numpy as np

from dataclasses import dataclass

from ..v2.line_filter import LineDetector, DetectedLine, cluster_lines
from ..v2.net_detector import detect_net


@dataclass
class Candidate:
    """A candidate keypoint with image coordinates and source info."""
    x: float
    y: float
    h_line_idx: int
    v_line_idx: int
    white_score: float = 0.0


class KeypointDetector:
    """Detects candidate court keypoints from line intersections."""

    def __init__(self, white_thresh: int = 180):
        self._line_det = LineDetector(white_thresh=white_thresh)

    def detect(self, frame: np.ndarray) -> Tuple[
        List[Candidate],
        List[DetectedLine],  # h_lines
        List[DetectedLine],  # v_lines
        np.ndarray,          # white_mask
        np.ndarray,          # roi_mask
    ]:
        """
        Detect candidate keypoints from a single frame.

        Returns candidates plus intermediate data for debugging.
        """
        H, W = frame.shape[:2]

        # Step 1: detect + cluster lines (reuse V2)
        raw_lines, roi_mask, white_mask = self._line_det.detect(frame)
        h_lines, v_lines = cluster_lines(raw_lines)

        # Step 2: identify and remove net line
        h_court = self._remove_net(h_lines, white_mask)

        # Step 3: compute all H×V intersections
        candidates = self._intersect_all(
            h_court, v_lines, white_mask, H, W)

        return candidates, h_lines, v_lines, white_mask, roi_mask

    def _remove_net(
        self,
        h_lines: List[DetectedLine],
        white_mask: np.ndarray,
    ) -> List[DetectedLine]:
        """Remove net line from horizontal lines (it's not paint)."""
        net_line, court_h = detect_net(h_lines, white_mask, min_score=0.4)
        return court_h

    def _intersect_all(
        self,
        h_lines: List[DetectedLine],
        v_lines: List[DetectedLine],
        white_mask: np.ndarray,
        H: int, W: int,
    ) -> List[Candidate]:
        """Compute all H×V intersections and filter valid ones."""
        # Dilate white mask for proximity check
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
        dilated = cv2.dilate(white_mask, kernel)

        candidates = []
        margin = 0.10  # allow 10% outside frame

        for hi, h_line in enumerate(h_lines):
            for vi, v_line in enumerate(v_lines):
                pt = _line_intersect(h_line, v_line)
                if pt is None:
                    continue
                x, y = pt

                # Must be in or near frame
                if not (-W * margin <= x <= W * (1 + margin) and
                        -H * margin <= y <= H * (1 + margin)):
                    continue

                # Score: white pixel density in local neighborhood
                ws = _white_score(dilated, x, y, H, W)

                candidates.append(Candidate(
                    x=x, y=y,
                    h_line_idx=hi,
                    v_line_idx=vi,
                    white_score=ws,
                ))

        # Keep candidates with some white pixel support
        candidates = [c for c in candidates if c.white_score > 0.05]
        return candidates


# ── Helpers ──────────────────────────────────────────────────────────────────

def _line_intersect(
    a: DetectedLine, b: DetectedLine,
) -> tuple | None:
    """Line-line intersection via Cramer's rule."""
    denom = ((a.x1 - a.x2) * (b.y1 - b.y2) -
             (a.y1 - a.y2) * (b.x1 - b.x2))
    if abs(denom) < 1e-6:
        return None
    t = ((a.x1 - b.x1) * (b.y1 - b.y2) -
         (a.y1 - b.y1) * (b.x1 - b.x2)) / denom
    x = a.x1 + t * (a.x2 - a.x1)
    y = a.y1 + t * (a.y2 - a.y1)
    return (x, y)


def _white_score(
    dilated_mask: np.ndarray,
    x: float, y: float,
    H: int, W: int,
    radius: int = 12,
) -> float:
    """Fraction of white pixels in a square patch around (x, y)."""
    ix, iy = int(round(x)), int(round(y))
    x1 = max(0, ix - radius)
    x2 = min(W, ix + radius + 1)
    y1 = max(0, iy - radius)
    y2 = min(H, iy + radius + 1)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    patch = dilated_mask[y1:y2, x1:x2]
    return float(np.mean(patch > 0))