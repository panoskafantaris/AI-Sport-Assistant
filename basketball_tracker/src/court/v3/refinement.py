"""
Homography refinement — multi-start Nelder-Mead optimizer.

Optimizes 4 corner positions to maximize alignment of projected
template lines with detection masks (white paint + edge for net).
"""
from __future__ import annotations
from typing import Tuple
import cv2
import numpy as np
from scipy.optimize import minimize

from . import template as T
from .scoring import quick_score
from .detection_masks import DetectionMasks


class HomographyRefiner:
    """Refines boundary corners to maximize template-line alignment."""

    def __init__(
        self,
        max_iter: int = 500,
        sample_spacing: int = 4,
        n_restarts: int = 8,
        perturb_px: float = 80.0,
    ):
        self.max_iter = max_iter
        self.spacing = sample_spacing
        self.n_restarts = n_restarts
        self.perturb_px = perturb_px

    def refine(
        self,
        corners: np.ndarray,
        masks: DetectionMasks,
        tight: bool = False,
    ) -> Tuple[np.ndarray, float]:
        """
        Refine corners via multi-start optimization.

        Args:
            tight: small perturbations only (±15px) for prior frames.
        """
        fh, fw = masks.white.shape[:2]
        best_corners = corners.copy()
        best_score = quick_score(corners, masks, fh, fw)

        if tight:
            perturb = 15.0
            starts = [corners.flatten()]
            for _ in range(3):
                noise = np.random.randn(8) * perturb
                starts.append(corners.flatten() + noise)
        else:
            starts = [corners.flatten()]
            for _ in range(self.n_restarts - 3):
                noise = np.random.randn(8) * self.perturb_px
                starts.append(corners.flatten() + noise)
            for shift_y in [-60, -120]:
                directed = corners.flatten().copy()
                directed[5] += shift_y  # TR y up
                directed[7] += shift_y  # TL y up
                starts.append(directed)

        for x0 in starts:
            result = minimize(
                lambda p: self._cost(p, masks, fh, fw),
                x0.astype(np.float64),
                method="Nelder-Mead",
                options={
                    "maxiter": self.max_iter,
                    "xatol": 0.5,
                    "fatol": 0.0005,
                    "adaptive": True,
                },
            )
            c = result.x.reshape(4, 2).astype(np.float32)
            s = quick_score(c, masks, fh, fw)
            if s > best_score:
                best_score = s
                best_corners = c

        return best_corners, round(best_score, 4)

    def _cost(self, params, masks, fh, fw) -> float:
        c = params.reshape(4, 2).astype(np.float32)
        alignment = quick_score(c, masks, fh, fw)
        penalty = self._geometry_penalty(c, fh, fw)
        return 1.0 - alignment + penalty

    @staticmethod
    def _geometry_penalty(corners, fh, fw) -> float:
        bl, br, tr, tl = corners
        penalty = 0.0
        for cx, cy in corners:
            if cx < -fw*0.2 or cx > fw*1.2:
                penalty += 0.1
            if cy < -fh*0.2 or cy > fh*1.2:
                penalty += 0.1
        if bl[1] <= tl[1] or br[1] <= tr[1]:
            penalty += 0.5
        near_w = np.linalg.norm(br - bl)
        far_w = np.linalg.norm(tr - tl)
        if near_w < far_w * 0.9:
            penalty += 0.3
        return penalty