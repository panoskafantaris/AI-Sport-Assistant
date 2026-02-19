"""
Court boundary refinement via Nelder-Mead optimization.
"""
from __future__ import annotations
from typing import Tuple
import cv2
import numpy as np
from scipy.optimize import minimize
from . import template as T


class HomographyRefiner:
    def __init__(self, max_iter=200):
        self.max_iter = max_iter

    def refine(self, corners: np.ndarray, white_mask: np.ndarray,
               ) -> Tuple[np.ndarray, float]:
        fh, fw = white_mask.shape[:2]
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        dilated = cv2.dilate(white_mask, k)
        x0 = corners.flatten()

        def cost(params):
            c = params.reshape(4, 2).astype(np.float32)
            H_mat, _ = cv2.findHomography(c, T.BOUNDARY_CORNERS, cv2.RANSAC)
            if H_mat is None:
                return 1.0
            try:
                H_inv = np.linalg.inv(H_mat)
            except np.linalg.LinAlgError:
                return 1.0
            total, hits = 0, 0
            for line in T.COURT_LINES:
                pts_w = np.array(
                    [[[line.x1, line.y1]], [[line.x2, line.y2]]],
                    dtype=np.float32)
                pts_i = cv2.perspectiveTransform(pts_w, H_inv)
                p1, p2 = pts_i[0, 0], pts_i[1, 0]
                n = max(2, int(np.linalg.norm(p2 - p1) / 5))
                for i in range(n):
                    t = i / max(n - 1, 1)
                    px = int(p1[0] + t * (p2[0] - p1[0]))
                    py = int(p1[1] + t * (p2[1] - p1[1]))
                    if 0 <= px < fw and 0 <= py < fh:
                        total += 1
                        if dilated[py, px] > 0:
                            hits += 1
            return 1.0 - hits / max(total, 1)

        result = minimize(cost, x0, method="Nelder-Mead",
                          options={"maxiter": self.max_iter,
                                   "xatol": 1.0, "fatol": 0.001,
                                   "adaptive": True})
        refined = result.x.reshape(4, 2).astype(np.float32)
        return refined, round(1.0 - result.fun, 4)