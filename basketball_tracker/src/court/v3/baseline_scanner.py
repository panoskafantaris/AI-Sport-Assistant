"""
Baseline endpoint scanner — finds line extents from white mask.

Instead of relying on Hough line endpoints (which overshoot) or
V-line intersections (which accumulate extrapolation error), this
module directly scans the white mask at a line's y-position to
find where white paint starts and ends.
"""
from __future__ import annotations
from typing import Optional, Tuple
import numpy as np

from ..v2.line_filter import DetectedLine


def has_paint(line, white_mask, fh, min_density=0.025, band=4):
    """
    Check if a line has real white paint (not a structural edge).

    Structural edges: thick=0, density<0.02 (e.g., court surround edge)
    Painted lines: thick>=1, density>=0.03 (baselines, service lines)
    """
    y = int(line.midpoint[1])
    y0 = max(0, y - band)
    y1 = min(fh, y + band + 1)
    x0 = max(0, int(min(line.x1, line.x2)))
    x1 = min(white_mask.shape[1], int(max(line.x1, line.x2)) + 1)
    if x1 <= x0 or y1 <= y0:
        return False
    region = white_mask[y0:y1, x0:x1]
    density = np.count_nonzero(region) / max(region.size, 1)
    return density >= min_density


def fallback_scan_far_baseline(
    white_mask: np.ndarray,
    fh: int, fw: int,
) -> Optional[DetectedLine]:
    """
    Scan white mask directly for far baseline when Hough misses it.

    Search y=5%-35% of frame height (standard broadcast far baseline zone).
    Returns a synthetic DetectedLine at the strongest horizontal white band.
    """
    y_start = int(fh * 0.05)
    y_end = int(fh * 0.35)
    best_y = None
    best_count = 0

    for y in range(y_start, y_end, 2):
        band = white_mask[max(0, y-2):y+3, :]
        count = np.count_nonzero(np.max(band, axis=0))
        if count > best_count:
            best_count = count
            best_y = y

    if best_y is None or best_count < fw * 0.15:
        return None

    band = white_mask[max(0, best_y-2):best_y+3, :]
    profile = np.max(band, axis=0)
    white_x = np.where(profile > 0)[0]
    if len(white_x) < 30:
        return None

    return DetectedLine(
        float(white_x[0]), float(best_y),
        float(white_x[-1]), float(best_y))


def scan_baseline_extent(
    line: DetectedLine,
    white_mask: np.ndarray,
    fh: int, fw: int,
    band_half: int = 5,
    min_run: int = 20,
    margin_frac: float = 0.15,
) -> Optional[Tuple[float, float]]:
    """
    Scan white mask at the line's y-position to find left/right extent.

    The scan is constrained to the Hough line's endpoint region
    (with margin) to avoid picking up overlay text (scoreboards,
    sponsor text like "QATAR") as court paint.

    Returns (left_x, right_x) or None.
    """
    y_center = int(line.midpoint[1])
    y_top = max(0, y_center - band_half)
    y_bot = min(fh, y_center + band_half + 1)

    # Constrain scan to Hough line region + margin
    line_left = min(line.x1, line.x2)
    line_right = max(line.x1, line.x2)
    line_span = line_right - line_left
    margin = max(line_span * margin_frac, 50)
    scan_left = max(0, int(line_left - margin))
    scan_right = min(fw, int(line_right + margin))

    # Collapse vertical band to 1D horizontal profile (within bounds)
    band = white_mask[y_top:y_bot, scan_left:scan_right]
    profile = np.max(band, axis=0)

    is_white = profile > 0
    if not np.any(is_white):
        return None

    white_x = np.where(is_white)[0]
    if len(white_x) < min_run:
        return None

    # Find the longest contiguous run (gap > 10px = different segment)
    diffs = np.diff(white_x)
    breaks = np.where(diffs > 10)[0]

    if len(breaks) == 0:
        left_x = float(white_x[0]) + scan_left
        right_x = float(white_x[-1]) + scan_left
    else:
        # Multiple segments — find the longest
        segments = []
        start = 0
        for brk in breaks:
            segments.append((start, brk))
            start = brk + 1
        segments.append((start, len(white_x) - 1))

        best_seg = max(segments, key=lambda s: s[1] - s[0])
        left_x = float(white_x[best_seg[0]]) + scan_left
        right_x = float(white_x[best_seg[1]]) + scan_left

    if right_x - left_x < 50:
        return None

    return (left_x, right_x)