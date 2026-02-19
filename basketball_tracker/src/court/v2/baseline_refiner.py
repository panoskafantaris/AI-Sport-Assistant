"""
Baseline refinement — snaps a detected line to actual paint center.

Technique: **Vertical paint profile centering**

Hough lines detect the TOP edge of thick paint (baselines are ~8-10px
at broadcast resolution). A naive white-pixel RANSAC fits to the
top-half of the paint, producing a line ~10-15px too high.

This module instead:
  1. Uses an asymmetric band (more below than above the Hough line)
  2. For each x-column, finds the vertical center of the paint band
  3. RANSAC-fits those paint centers → sub-pixel accuracy on true center

The result sits exactly on the paint midline regardless of line
thickness, initial Hough offset, or player occlusion gaps.
"""
from __future__ import annotations
from typing import List, Optional, Tuple
import cv2
import numpy as np

from .line_filter import DetectedLine

# ── Band geometry ─────────────────────────────────────────────────────────────
# Hough systematically detects the top edge of thick paint.
# Extend the search band more downward to capture the full paint.
_BAND_ABOVE = 10   # pixels above detected line
_BAND_BELOW = 35   # pixels below detected line (captures full paint)
_COL_STEP   = 3    # sample every Nth column (speed vs accuracy)
_MIN_PAINT_PX = 3  # minimum white pixels in a column to count as paint
_MIN_CENTERS  = 30 # minimum paint-center points needed for RANSAC


def refine_baseline(
    line: DetectedLine,
    white_mask: np.ndarray,
    frame: np.ndarray = None,
    white_thresh: int = 175,
) -> DetectedLine:
    """
    Refine a near-horizontal baseline to the paint center.

    Args:
        line:         Approximate baseline from Hough/clustering.
        white_mask:   ROI-masked white mask (fallback source).
        frame:        Original BGR frame. If provided, extracts
                      white pixels directly (bypasses ROI cutout).
        white_thresh: Grayscale threshold for paint detection.

    Returns:
        Refined DetectedLine at the true paint center.
    """
    if frame is not None:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = None

    H, W = (frame.shape[:2] if frame is not None
             else white_mask.shape[:2])

    slope = _line_slope(line)

    # Collect paint-center points across columns
    # Extend well past Hough endpoints — ROI cutouts hide baseline ends
    extent = max(line.length * 0.4, 200)  # at least 200px extension
    centers: List[Tuple[float, float]] = []
    x_start = max(0, int(min(line.x1, line.x2) - extent))
    x_end   = min(W, int(max(line.x1, line.x2) + extent))

    for x in range(x_start, x_end, _COL_STEP):
        # Predicted Y at this x from the Hough line (extrapolate)
        pred_y = line.y1 + slope * (x - line.x1)
        # Skip if prediction goes off frame
        if pred_y < 0 or pred_y > H:
            continue
        top = max(0, int(pred_y - _BAND_ABOVE))
        bot = min(H, int(pred_y + _BAND_BELOW))
        if bot <= top:
            continue

        # Extract vertical column brightness
        if gray is not None:
            col = gray[top:bot, x].astype(np.float32)
        else:
            col = white_mask[top:bot, x].astype(np.float32)

        cy = _paint_center_in_column(col, top, white_thresh)
        if cy is not None:
            centers.append((float(x), cy))

    if len(centers) < _MIN_CENTERS:
        return line

    xs = np.array([c[0] for c in centers])
    ys = np.array([c[1] for c in centers])

    refined = _ransac_fit(xs, ys)
    if refined is None:
        return line

    slope_r, intercept_r = refined

    # Extend to full visible paint extent
    x_min, x_max = float(xs.min()), float(xs.max())
    y1 = slope_r * x_min + intercept_r
    y2 = slope_r * x_max + intercept_r
    return DetectedLine(x_min, y1, x_max, y2)


# ── Paint center extraction ──────────────────────────────────────────────────

def _paint_center_in_column(
    col: np.ndarray, top_offset: int, thresh: int,
) -> Optional[float]:
    """
    Find the vertical center of the white paint band in one column.

    Scans for a contiguous bright region, returns its center Y
    in frame coordinates.  None if no paint found.
    """
    bright = col >= thresh
    if np.sum(bright) < _MIN_PAINT_PX:
        return None

    indices = np.where(bright)[0]
    if len(indices) == 0:
        return None

    # Take the longest contiguous run (the actual paint band)
    runs = _contiguous_runs(indices)
    best_run = max(runs, key=len)

    if len(best_run) < _MIN_PAINT_PX:
        return None

    # Paint center = midpoint of the run, in frame coords
    center_local = (best_run[0] + best_run[-1]) / 2.0
    return center_local + top_offset


def _contiguous_runs(indices: np.ndarray) -> List[np.ndarray]:
    """Split sorted indices into contiguous runs."""
    if len(indices) == 0:
        return []
    breaks = np.where(np.diff(indices) > 1)[0] + 1
    return np.split(indices, breaks)


# ── RANSAC line fit ───────────────────────────────────────────────────────────

def _ransac_fit(
    xs: np.ndarray, ys: np.ndarray,
    n_iters: int = 100, threshold: float = 2.0,
) -> Optional[Tuple[float, float]]:
    """
    RANSAC fit of y = slope * x + intercept on paint centers.

    Tight threshold (2px) rejects outlier columns where reflections
    or adjacent markings shift the apparent paint center.
    """
    n = len(xs)
    if n < 2:
        return None

    best_inliers = 0
    best_mask = None

    for _ in range(n_iters):
        idx = np.random.choice(n, 2, replace=False)
        x1, y1 = xs[idx[0]], ys[idx[0]]
        x2, y2 = xs[idx[1]], ys[idx[1]]
        dx = x2 - x1
        if abs(dx) < 1e-6:
            continue

        s = (y2 - y1) / dx
        b = y1 - s * x1
        residuals = np.abs(ys - (s * xs + b))
        mask = residuals < threshold
        count = int(np.sum(mask))

        if count > best_inliers:
            best_inliers = count
            best_mask = mask

    if best_mask is None or best_inliers < 10:
        return None

    # Least-squares refit on inliers
    ix = xs[best_mask]
    iy = ys[best_mask]
    A = np.vstack([ix, np.ones(len(ix))]).T
    result = np.linalg.lstsq(A, iy, rcond=None)
    slope, intercept = result[0]
    return (float(slope), float(intercept))


def _line_slope(line: DetectedLine) -> float:
    dx = line.x2 - line.x1
    if abs(dx) < 1e-6:
        return 0.0
    return (line.y2 - line.y1) / dx