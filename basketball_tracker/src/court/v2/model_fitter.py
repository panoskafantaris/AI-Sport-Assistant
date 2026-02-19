"""
Model fitter — scores boundary candidates and picks the best.

Colour weight is only 10% since blue court = blue stands at some venues.
"""
from __future__ import annotations
from typing import List, Optional, Tuple
import numpy as np
from .scoring import (edge_alignment_score, court_colour_overlap_score,
                      geometric_score, line_coverage_score)


class ModelFitter:
    def __init__(self, weights=(0.40, 0.10, 0.30, 0.20)):
        self.w_e, self.w_c, self.w_g, self.w_l = weights

    def score_candidate(self, corners, white_mask, colour_mask, fh, fw):
        se = edge_alignment_score(corners, white_mask)
        sc = court_colour_overlap_score(corners, colour_mask)
        sg = geometric_score(corners, fh, fw)
        sl = line_coverage_score(corners, white_mask)
        total = self.w_e*se + self.w_c*sc + self.w_g*sg + self.w_l*sl
        print(f"    Score: edge={se:.2f} col={sc:.2f}"
              f" geo={sg:.2f} cov={sl:.2f} → {total:.3f}")
        return round(total, 4)

    def find_best(self, candidates, white_mask, colour_mask, fh, fw
                  ) -> Tuple[Optional[np.ndarray], float]:
        best, bs = None, 0.0
        for c in candidates:
            s = self.score_candidate(c, white_mask, colour_mask, fh, fw)
            if s > bs:
                bs, best = s, c
        return best, bs