"""
Net detector — identifies the net among horizontal lines.

Two signals: thickness (net mesh = 10-35px, court lines = 1-4px)
and density (net mesh = 0.15-0.8, court lines = 0.01-0.05).
Combined via geometric mean so BOTH must be elevated.
"""
from __future__ import annotations
from typing import List, Optional, Tuple
import numpy as np

from .line_filter import DetectedLine
from .roi import measure_line_thickness, measure_white_density


def score_net_likelihood(thickness: float, density: float) -> float:
    """Combined net score (0-1). Geometric mean of two signals."""
    t = min(max((thickness - 5) / 15.0, 0.0), 1.0)
    d = min(max((density - 0.03) / 0.15, 0.0), 1.0)
    return (t * d) ** 0.5


def detect_net(
    h_lines: List[DetectedLine],
    white_mask: np.ndarray,
    min_score: float = 0.5,
) -> Tuple[Optional[DetectedLine], List[DetectedLine]]:
    """
    Find the net among horizontal lines.

    Returns (net_line, remaining_court_lines).
    """
    scored = []
    for ln in h_lines:
        thick = measure_line_thickness(white_mask, ln.midpoint[1])
        density = measure_white_density(white_mask, ln.midpoint[1])
        ns = score_net_likelihood(thick, density)
        scored.append((ln, thick, density, ns))
        print(f"    H y={ln.midpoint[1]:.0f}  thick={thick:.1f}"
              f"  dens={density:.3f}  net={ns:.2f}"
              f"  len={ln.length:.0f}")

    net_cands = [(ln, s) for ln, t, d, s in scored if s >= min_score]
    net_line = None
    if net_cands:
        net_cands.sort(key=lambda x: x[1], reverse=True)
        net_line = net_cands[0][0]

    court_h = [ln for ln, t, d, s in scored
               if net_line is None or ln is not net_line]
    return net_line, court_h