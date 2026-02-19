"""
Spatial ROI for court line detection.

Defines a trapezoid that captures the court while excluding:
  - Top 7%:    banners, broadcast graphics
  - Bottom 18%: ad boards ("QATAR", "ATP TOUR"), sponsors
  - Bottom-left: scoreboard overlay
  - Corners: coaching boxes / ball kids

Also provides white-line extraction and line measurement utilities.
"""
from __future__ import annotations
import cv2
import numpy as np

# ── ROI geometry (fraction of frame) ─────────────────────────────────────────
_TOP_Y   = 0.07    # top of trapezoid (below banners)
_BOT_Y   = 0.82    # bottom of trapezoid (above ad boards)
_TOP_IN  = 0.22    # inset from sides at top (narrower — perspective)
_BOT_IN  = 0.03    # inset from sides at bottom (wider)

# Scoreboard cutout (bottom-left)
_SCORE_X = 0.22    # right edge of scoreboard zone
_SCORE_Y = 0.55    # top of scoreboard zone (generous)


def build_court_trapezoid(H: int, W: int) -> np.ndarray:
    """Return a binary mask selecting the court region."""
    mask = np.zeros((H, W), np.uint8)

    top_y = int(H * _TOP_Y)
    bot_y = int(H * _BOT_Y)
    top_l = int(W * _TOP_IN)
    top_r = int(W * (1 - _TOP_IN))
    bot_l = int(W * _BOT_IN)
    bot_r = int(W * (1 - _BOT_IN))

    pts = np.array([[top_l, top_y], [top_r, top_y],
                     [bot_r, bot_y], [bot_l, bot_y]])
    cv2.fillPoly(mask, [pts], 255)

    # Cut scoreboard zone
    cv2.rectangle(mask, (0, int(H * _SCORE_Y)),
                  (int(W * _SCORE_X), H), 0, -1)

    return mask


def extract_white_lines(
    frame: np.ndarray, roi: np.ndarray, thresh: int = 180,
) -> np.ndarray:
    """Extract bright (white) pixels within the ROI."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, bright = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)
    return cv2.bitwise_and(bright, roi)


# ── Line measurement ─────────────────────────────────────────────────────────

def measure_line_thickness(white_mask: np.ndarray, y: float,
                           half_band: int = 25) -> float:
    """
    Median vertical white extent at y across 5 sample strips.

    Net mesh: 10-35px.  Court paint: 2-6px.
    Samples at 20%, 35%, 50%, 65%, 80% of width to avoid
    localized features (e.g. ATP TOUR sign only at center).
    """
    H, W = white_mask.shape[:2]
    yi = int(y)
    top = max(0, yi - half_band)
    bot = min(H, yi + half_band + 1)
    strip = white_mask[top:bot, :]
    if strip.size == 0:
        return 0.0

    thicknesses = []
    for frac in (0.20, 0.35, 0.50, 0.65, 0.80):
        col = int(W * frac)
        col_data = strip[:, max(0, col - 3):col + 4]
        profile = np.max(col_data, axis=1)
        thicknesses.append(float(np.sum(profile > 0)))

    return float(np.median(thicknesses))


def measure_white_density(white_mask: np.ndarray, y: float,
                          half_band: int = 4) -> float:
    """Fraction of white pixels in a thin horizontal band at y."""
    H, W = white_mask.shape[:2]
    yi = int(y)
    top = max(0, yi - half_band)
    bot = min(H, yi + half_band + 1)
    band = white_mask[top:bot, :]
    if band.size == 0:
        return 0.0
    return float(np.mean(band > 0))