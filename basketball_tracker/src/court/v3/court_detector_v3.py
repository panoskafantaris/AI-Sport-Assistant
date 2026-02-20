"""
Court Detection V3 — Calibration-seeded + multi-modal alignment.

Pipeline:
  1. Build detection masks (white paint + Canny edges for net)
  2. If calibration corners → use as seed (skip refinement)
  3. Otherwise: line detection → multi-candidate seed → refine
  4. Temporal Kalman filter for smoothing
  5. Score against SCORED_LINES (no service lines)
"""
from __future__ import annotations
from typing import Optional, Tuple
import numpy as np

from ..v2.line_filter import LineDetector, cluster_lines
from ..v2.line_classifier import classify_lines
from .seed_homography import compute_seed
from .refinement import HomographyRefiner
from .temporal_tracker import TemporalTracker
from .confidence import compute_confidence, per_line_alignment
from .detection_masks import DetectionMasks


class CourtDetectorV3:
    """Calibration-seeded court detector with multi-modal scoring."""

    def __init__(
        self,
        refine: bool = True,
        temporal: bool = True,
        white_thresh: int = 180,
        calibration_corners: Optional[np.ndarray] = None,
    ):
        self._line_det = LineDetector(white_thresh=white_thresh)
        self._refiner = HomographyRefiner() if refine else None
        self._tracker = TemporalTracker() if temporal else None
        self._cal_corners = calibration_corners
        self._cal_used = False

    def set_calibration(self, corners: np.ndarray) -> None:
        self._cal_corners = corners.astype(np.float32)
        self._cal_used = False

    def reset_temporal(self) -> None:
        if self._tracker:
            self._tracker.reset()
        self._cal_used = False

    def detect(
        self, frame: np.ndarray, frame_number: int = 0,
    ) -> Tuple[Optional[np.ndarray], float, dict]:
        H, W = frame.shape[:2]
        info = {"frame": frame_number, "method": "none"}

        # Step 1: build detection masks
        masks = DetectionMasks.from_frame(frame)
        info["masks"] = masks

        # Step 2: detect lines (for auto-seed + debug)
        raw_lines, roi_mask, white_mask = self._line_det.detect(frame)
        h_lines, v_lines = cluster_lines(raw_lines)
        info.update(n_h_lines=len(h_lines), n_v_lines=len(v_lines),
                    h_lines=h_lines, v_lines=v_lines,
                    white_mask=white_mask)

        cl = classify_lines(
            h_lines, v_lines, white_mask, H, W, frame=frame)
        info["classified"] = cl

        print(f"  [V3] {len(h_lines)}H {len(v_lines)}V")

        # Step 3: get seed corners
        corners = None

        # Priority 1: calibration (first frame only)
        if self._cal_corners is not None and not self._cal_used:
            corners = self._cal_corners.copy()
            info["method"] = "calibration"
            self._cal_used = True
            print(f"  [V3] Seed: calibration")

        # Priority 2: multi-candidate from lines
        if corners is None:
            corners, seed_score = compute_seed(
                h_lines, v_lines, white_mask, H, W, masks=masks)
            info["seed_score"] = seed_score
            if corners is not None:
                info["method"] = "multi_seed"
                print(f"  [V3] Seed: score={seed_score:.3f}")

        # Priority 3: temporal prior
        if corners is None and self._tracker and self._tracker.is_initialized:
            corners = self._tracker.get_prior_corners()
            if corners is not None:
                info["method"] = "prior"

        if corners is None:
            if self._tracker:
                smoothed = self._tracker.update(None)
                if smoothed is not None:
                    return smoothed, 0.3, info
            return None, 0.0, info

        # Step 4: refinement (skip for calibration seed)
        if self._refiner is not None and info["method"] != "calibration":
            tight = info["method"] == "prior"
            corners, ref_score = self._refiner.refine(
                corners, masks, tight=tight)
            info["refinement_score"] = ref_score
            print(f"  [V3] Refined: {ref_score:.3f}"
                  f"{'  (tight)' if tight else ''}")
        elif info["method"] == "calibration":
            print(f"  [V3] Skipping refinement (calibration seed)")

        # Step 5: confidence
        conf = compute_confidence(corners, masks, H, W)
        info["line_scores"] = per_line_alignment(corners, masks)
        info["confidence"] = conf

        # Step 6: temporal smoothing
        if self._tracker:
            smoothed = self._tracker.update(corners, conf)
            if smoothed is not None:
                corners = smoothed

        print(f"  [V3] Final conf={conf:.3f}")
        return corners, conf, info