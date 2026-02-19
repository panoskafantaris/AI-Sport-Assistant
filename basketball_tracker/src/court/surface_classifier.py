"""
Tennis court surface classifier (Option 4).

Classifies the court surface as clay, hard, or grass using HSV colour
analysis of the centre region of the frame.

Each surface type provides:
  - A dominant HSV colour range for masking
  - A refined court mask that constrains Hough line detection
  - Confidence score (0-1) for how clearly the surface was identified

Surface HSV profiles (tuned for broadcast footage):
  Clay  – red/orange  H: 5-20,  S: 80-255, V: 80-220
  Hard  – blue/green  H: 85-140, S: 60-255, V: 60-230   (US Open blue, AO blue)
  Grass – green       H: 35-80,  S: 50-255, V: 60-200
"""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple
import cv2
import numpy as np


class SurfaceType(Enum):
    CLAY  = "clay"
    HARD  = "hard"
    GRASS = "grass"
    UNKNOWN = "unknown"


# HSV ranges per surface: list of (lower, upper) pairs (multiple ranges = OR logic)
_SURFACE_RANGES: dict[SurfaceType, list[Tuple[np.ndarray, np.ndarray]]] = {
    SurfaceType.CLAY: [
        (np.array([0,  80,  80]), np.array([18, 255, 220])),   # orange-red
        (np.array([170, 80, 80]), np.array([180, 255, 220])),  # red wraparound
    ],
    SurfaceType.HARD: [
        # Court surface is vivid saturated blue (S>100, V>80)
        # Stands are pale/washed-out blue (low S) — excluded by S>=100
        (np.array([85,  100, 80]), np.array([140, 255, 220])), # vivid blue court
        (np.array([140, 80,  80]), np.array([170, 200, 210])), # purple-blue (some venues)
    ],
    SurfaceType.GRASS: [
        (np.array([35,  50,  60]), np.array([80, 255, 200])),  # green
    ],
}


@dataclass
class SurfaceResult:
    surface:    SurfaceType
    confidence: float           # 0-1
    mask:       np.ndarray      # binary mask of court-coloured pixels
    hsv_mean:   Tuple[float, float, float]  # mean HSV of detected region


class SurfaceClassifier:
    """
    Classifies the court surface type and produces a colour-based court mask.
    The mask is used by AutoCourtDetector to constrain Hough line detection.
    """

    def __init__(self, centre_crop_ratio: float = 0.5):
        """
        Args:
            centre_crop_ratio: Fraction of frame (centred) used for classification.
                               0.5 = inner 50% — avoids crowd/scoreboard contamination.
        """
        self.centre_crop_ratio = centre_crop_ratio

    # ── Public API ─────────────────────────────────────────────────────────────

    def classify(self, frame: np.ndarray) -> SurfaceResult:
        """
        Classify the court surface and return a full-frame colour mask.
        """
        hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        crop = self._centre_crop(hsv)

        scores: dict[SurfaceType, float] = {}
        masks:  dict[SurfaceType, np.ndarray] = {}

        for surface, ranges in _SURFACE_RANGES.items():
            full_mask = self._build_mask(hsv, ranges)
            crop_mask = self._centre_crop(full_mask)
            score     = float(np.mean(crop_mask > 0))
            scores[surface] = score
            masks[surface]  = full_mask

        best     = max(scores, key=scores.get)
        best_conf= scores[best]

        if best_conf < 0.08:    # < 8% of centre crop matched → unknown scene
            best      = SurfaceType.UNKNOWN
            best_mask = np.zeros(frame.shape[:2], np.uint8)
        else:
            best_mask = self._refine_mask(masks[best])

        hsv_mean = self._mean_hsv(hsv, best_mask)

        return SurfaceResult(
            surface    = best,
            confidence = round(best_conf, 3),
            mask       = best_mask,
            hsv_mean   = hsv_mean,
        )

    # ── Internals ──────────────────────────────────────────────────────────────

    def _centre_crop(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        r    = self.centre_crop_ratio
        y1   = int(h * (1 - r) / 2)
        y2   = int(h * (1 + r) / 2)
        x1   = int(w * (1 - r) / 2)
        x2   = int(w * (1 + r) / 2)
        return img[y1:y2, x1:x2]

    @staticmethod
    def _build_mask(
        hsv: np.ndarray,
        ranges: list[Tuple[np.ndarray, np.ndarray]]
    ) -> np.ndarray:
        mask = np.zeros(hsv.shape[:2], np.uint8)
        for lo, hi in ranges:
            mask |= cv2.inRange(hsv, lo, hi)
        return mask

    @staticmethod
    def _refine_mask(mask: np.ndarray) -> np.ndarray:
        """Remove noise and fill holes in the colour mask."""
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=1)
        return mask

    @staticmethod
    def _mean_hsv(
        hsv: np.ndarray, mask: np.ndarray
    ) -> Tuple[float, float, float]:
        if mask is None or not np.any(mask):
            return (0.0, 0.0, 0.0)
        mean = cv2.mean(hsv, mask=mask)
        return (round(mean[0], 1), round(mean[1], 1), round(mean[2], 1))