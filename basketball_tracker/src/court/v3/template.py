"""
Tennis court template — world coordinates + canonical keypoints.

    FL ──────────────────── FR      y = 23.77  (far baseline)
    │                        │
    │                        │      y = 11.885 (net — mesh, not paint)
    │                        │
    │   cs_top ──── cs_bot   │      center service line
    │                        │
    BL ────────bc──── BR      y = 0  (near baseline)

   x=0                    x=8.23

SCORED_LINES: lines we measure alignment against (no service lines).
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List
import numpy as np

# ── Court dimensions (metres) ──────────────────────────────────
COURT_LENGTH      = 23.77
SINGLES_WIDTH     = 8.23
DOUBLES_WIDTH     = 10.97
SERVICE_BOX_DEPTH = 6.40
NET_Y             = COURT_LENGTH / 2       # 11.885
CENTER_X          = SINGLES_WIDTH / 2      # 4.115
FAR_SERVICE_Y     = COURT_LENGTH - SERVICE_BOX_DEPTH  # 17.37


@dataclass(frozen=True)
class TemplateLine:
    name: str
    x1: float; y1: float; x2: float; y2: float
    detect_mode: str = "white"  # "white" for paint, "edge" for net


# Lines scored for alignment (service lines removed)
SCORED_LINES: List[TemplateLine] = [
    TemplateLine("near_baseline",  0.0, 0.0,
                 SINGLES_WIDTH, 0.0),
    TemplateLine("far_baseline",   0.0, COURT_LENGTH,
                 SINGLES_WIDTH, COURT_LENGTH),
    TemplateLine("left_sideline",  0.0, 0.0,
                 0.0, COURT_LENGTH),
    TemplateLine("right_sideline", SINGLES_WIDTH, 0.0,
                 SINGLES_WIDTH, COURT_LENGTH),
    TemplateLine("center_service", CENTER_X, SERVICE_BOX_DEPTH,
                 CENTER_X, FAR_SERVICE_Y),
    TemplateLine("net", 0.0, NET_Y, SINGLES_WIDTH, NET_Y,
                 detect_mode="edge"),
]


# ── Boundary corners [BL, BR, TR, TL] in world coords ─────────
BOUNDARY_CORNERS = np.array([
    [0.0,           0.0],            # BL (near-left)
    [SINGLES_WIDTH, 0.0],            # BR (near-right)
    [SINGLES_WIDTH, COURT_LENGTH],   # TR (far-right)
    [0.0,           COURT_LENGTH],   # TL (far-left)
], dtype=np.float32)


# ── Canonical keypoints (line intersections) ───────────────────
KEYPOINTS: List[tuple] = [
    ("near_base_left",     0.0,           0.0),
    ("near_base_center",   CENTER_X,      0.0),
    ("near_base_right",    SINGLES_WIDTH,  0.0),
    ("near_svc_left",      0.0,           SERVICE_BOX_DEPTH),
    ("near_svc_center",    CENTER_X,      SERVICE_BOX_DEPTH),
    ("near_svc_right",     SINGLES_WIDTH,  SERVICE_BOX_DEPTH),
    ("far_svc_left",       0.0,           FAR_SERVICE_Y),
    ("far_svc_center",     CENTER_X,      FAR_SERVICE_Y),
    ("far_svc_right",      SINGLES_WIDTH,  FAR_SERVICE_Y),
    ("far_base_left",      0.0,           COURT_LENGTH),
    ("far_base_center",    CENTER_X,      COURT_LENGTH),
    ("far_base_right",     SINGLES_WIDTH,  COURT_LENGTH),
]

KEYPOINTS_WORLD = np.array(
    [[x, y] for _, x, y in KEYPOINTS], dtype=np.float32)
KEYPOINT_NAMES = [name for name, _, _ in KEYPOINTS]
NUM_KEYPOINTS = len(KEYPOINTS)