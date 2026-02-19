"""
Court Detection V2 — main detector.

Pipeline:
  1. Build spatial ROI trapezoid
  2. Extract white lines within ROI
  3. Hough line detection + clustering
  4. Classify lines (baselines, net, service, sidelines)
  5. Build corners from classified lines (primary)
  6. If that fails, try combinatorial candidates
  7. Score and optionally refine
"""
from __future__ import annotations
from typing import Optional, Tuple
import numpy as np

from .line_filter import LineDetector, cluster_lines
from .line_classifier import classify_lines, ClassifiedLines
from .intersection import (corners_from_classified,
                           corners_from_baselines,
                           generate_candidates)
from .model_fitter import ModelFitter
from .refiner import HomographyRefiner

AUTO_THRESHOLD = 0.50


class CourtDetectorV2:
    """Spatial-ROI based court detector with line classification."""

    def __init__(self, refine: bool = True, debug: bool = False):
        self._line_det   = LineDetector()
        self._fitter     = ModelFitter()
        self._refiner    = HomographyRefiner() if refine else None
        self._debug      = debug

    def detect(
        self, frame: np.ndarray, frame_number: int = 0,
    ) -> Tuple[Optional[np.ndarray], float, dict, ClassifiedLines]:
        """
        Returns:
          (corners, confidence, surface_info, classified_lines)
          corners: (4,2) float32 [TL, TR, BR, BL] or None
        """
        H, W = frame.shape[:2]
        surface_info = {"surface": "unknown", "confidence": 0.0}

        # Step 1-2: detect lines
        raw_lines, roi_mask, white_mask = self._line_det.detect(frame)

        # Step 3: cluster
        h_lines, v_lines = cluster_lines(raw_lines)
        white_px = int(np.sum(white_mask > 0))
        print(f"  [V2] {len(raw_lines)} raw → {len(h_lines)}H"
              f" {len(v_lines)}V  white={white_px}px")

        # Step 4: classify
        cl = classify_lines(h_lines, v_lines, white_mask, H, W,
                            frame=frame)
        print(f"  [V2] Classification:\n{cl.summary()}")

        # Step 5: primary — corners from baselines (technique D+B)
        corners = corners_from_baselines(cl, H, W)
        if corners is not None:
            print(f"  [V2] Corners: baseline-derived")

        # Step 5b: fallback — use classified sidelines
        if corners is None:
            corners = corners_from_classified(cl, H, W)
            if corners is not None:
                print(f"  [V2] Corners: classified sidelines")

        # Step 6: fallback — combinatorial candidates
        if corners is None:
            candidates = generate_candidates(h_lines, v_lines, H, W)
            print(f"  [V2] {len(candidates)} candidates")
            if candidates:
                corners, _ = self._fitter.find_best(
                    candidates, white_mask, roi_mask, H, W)

        if corners is None:
            return None, 0.0, surface_info, cl

        # Step 7: score
        conf = self._fitter.score_candidate(
            corners, white_mask, roi_mask, H, W)

        # Optional refinement
        if self._refiner and conf > 0.3:
            corners, ref_score = self._refiner.refine(corners, white_mask)
            conf = max(conf, ref_score)

        return corners, conf, surface_info, cl