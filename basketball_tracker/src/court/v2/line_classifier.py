"""
Court line classifier — assigns roles to detected H/V lines.

Strategy when NO net is detected (common on broadcast):
  - Use perspective cues: near baseline is WIDER than far baseline
  - Far baseline = topmost long H line
  - Near baseline = widest H line in the lower half
  - Service lines fill the gaps between them

Structural validation enforces correct vertical ordering.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np

from .line_filter import DetectedLine
from .net_detector import detect_net
from .baseline_refiner import refine_baseline


@dataclass
class ClassifiedLines:
    """All identified court lines with their roles."""
    far_baseline:   Optional[DetectedLine] = None
    far_service:    Optional[DetectedLine] = None
    net:            Optional[DetectedLine] = None
    near_service:   Optional[DetectedLine] = None
    near_baseline:  Optional[DetectedLine] = None
    left_sideline:  Optional[DetectedLine] = None
    right_sideline: Optional[DetectedLine] = None
    center_service: Optional[DetectedLine] = None
    unclassified_h: List[DetectedLine] = field(default_factory=list)
    unclassified_v: List[DetectedLine] = field(default_factory=list)

    @property
    def baselines(self):
        return (self.far_baseline, self.near_baseline)

    @property
    def sidelines(self):
        return (self.left_sideline, self.right_sideline)

    def summary(self) -> str:
        parts = []
        for name in ("far_baseline", "far_service", "net",
                      "near_service", "near_baseline",
                      "left_sideline", "right_sideline",
                      "center_service"):
            ln = getattr(self, name)
            if ln:
                parts.append(f"  {name:<16} y={ln.midpoint[1]:.0f}"
                             f"  len={ln.length:.0f}")
            else:
                parts.append(f"  {name:<16} —")
        return "\n".join(parts)


# ── Minimum span thresholds (fraction of frame width) ────────────────────────
_BASELINE_MIN_SPAN = 0.15   # baselines must span >= 15% of frame width
_SERVICE_MIN_SPAN  = 0.06   # service lines can be shorter


def classify_lines(
    h_lines: List[DetectedLine],
    v_lines: List[DetectedLine],
    white_mask: np.ndarray,
    frame_h: int, frame_w: int,
    frame: np.ndarray = None,
) -> ClassifiedLines:
    """Classify all detected H/V lines by their court role."""
    result = ClassifiedLines()
    h_sorted = sorted(h_lines, key=lambda l: l.midpoint[1])

    # Step 1: try net detection
    result.net, court_h = detect_net(h_sorted, white_mask)

    # Step 2: filter short H lines
    min_svc = frame_w * _SERVICE_MIN_SPAN
    court_h = [l for l in court_h if l.length >= min_svc]

    # Step 3: assign baselines
    min_base = frame_w * _BASELINE_MIN_SPAN
    if result.net is not None:
        _assign_with_net(result, court_h, min_base, frame_h, frame_w)
    else:
        _assign_no_net(result, court_h, min_base, frame_h, frame_w)

    # Step 3b: RANSAC-refine baselines to snap to white paint
    if result.near_baseline is not None:
        result.near_baseline = refine_baseline(
            result.near_baseline, white_mask, frame=frame)
    if result.far_baseline is not None:
        result.far_baseline = refine_baseline(
            result.far_baseline, white_mask, frame=frame)

    # Step 4: assign service lines
    _assign_service_lines(result, court_h)

    # Step 5: assign vertical lines
    _assign_verticals(result, v_lines, frame_w)

    return result


def _assign_with_net(result, court_h, min_base, fh, fw):
    """Assign baselines relative to a detected net."""
    net_y = result.net.midpoint[1]
    above = sorted(
        [l for l in court_h
         if l.midpoint[1] < net_y - 15 and l.length >= min_base],
        key=lambda l: l.midpoint[1])
    below = sorted(
        [l for l in court_h
         if l.midpoint[1] > net_y + 15 and l.length >= min_base],
        key=lambda l: l.length, reverse=True)

    if above:
        result.far_baseline = above[0]
    if below:
        result.near_baseline = below[0]  # widest below net
    _ensure_baselines(result, fh, fw)


def _assign_no_net(result, court_h, min_base, fh, fw):
    """
    No net found — use perspective cues.

    In perspective: near baseline is always WIDER than far baseline.
    - far baseline = topmost line that's long enough
    - near baseline = bottommost line that is wider than far baseline

    The "bottommost + wider" rule prevents service lines from being
    chosen over a slightly shorter (player-occluded) near baseline.
    """
    long_h = sorted(
        [l for l in court_h if l.length >= min_base],
        key=lambda l: l.midpoint[1])

    if len(long_h) >= 2:
        result.far_baseline = long_h[0]
        far_len = long_h[0].length

        # Near baseline: bottommost line wider than far baseline
        wider_below = [l for l in long_h[1:]
                       if l.length >= far_len * 0.85]
        if wider_below:
            # Pick bottommost among those wider than far
            result.near_baseline = max(
                wider_below, key=lambda l: l.midpoint[1])
        else:
            # Fallback: just pick bottommost long line
            result.near_baseline = long_h[-1]
    elif long_h:
        if long_h[0].midpoint[1] < fh * 0.45:
            result.far_baseline = long_h[0]
        else:
            result.near_baseline = long_h[0]

    _ensure_baselines(result, fh, fw)


def _ensure_baselines(result, fh, fw):
    """Synthesize missing baselines as frame-edge defaults."""
    if result.near_baseline is None:
        result.near_baseline = DetectedLine(
            fw * 0.03, fh * 0.80, fw * 0.97, fh * 0.80)
    if result.far_baseline is None and result.net:
        net_y = result.net.midpoint[1]
        near_y = result.near_baseline.midpoint[1]
        far_y = max(near_y - 2 * (near_y - net_y), fh * 0.05)
        mx, hw = result.net.midpoint[0], result.net.length * 0.35
        result.far_baseline = DetectedLine(mx - hw, far_y,
                                           mx + hw, far_y)


def _assign_service_lines(result, court_h):
    """Service lines between baselines and net/midpoint."""
    if not (result.far_baseline and result.near_baseline):
        return
    far_y = result.far_baseline.midpoint[1]
    near_y = result.near_baseline.midpoint[1]
    mid_y = (result.net.midpoint[1] if result.net
             else (far_y + near_y) / 2)

    for ln in court_h:
        if ln is result.far_baseline or ln is result.near_baseline:
            continue
        y = ln.midpoint[1]
        if far_y + 10 < y < mid_y - 10 and result.far_service is None:
            result.far_service = ln
        elif mid_y + 10 < y < near_y - 10 and result.near_service is None:
            result.near_service = ln
        else:
            result.unclassified_h.append(ln)


def _assign_verticals(result, v_lines, fw):
    """Assign sidelines and center service line."""
    vs = sorted(v_lines, key=lambda l: l.midpoint[0])
    if len(vs) >= 2:
        result.left_sideline = vs[0]
        result.right_sideline = vs[-1]
        if len(vs) >= 3:
            mid_x = (vs[0].midpoint[0] + vs[-1].midpoint[0]) / 2
            cands = [l for l in vs[1:-1]
                     if abs(l.midpoint[0] - mid_x) < fw * 0.15]
            if cands:
                result.center_service = min(
                    cands, key=lambda l: abs(l.midpoint[0] - mid_x))
        for l in vs:
            if l not in (result.left_sideline, result.right_sideline,
                         result.center_service):
                result.unclassified_v.append(l)
    elif vs:
        result.unclassified_v.extend(vs)