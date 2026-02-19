"""
Tennis court geometry template — world coordinates in metres.

    FL ──────────────────── FR      y = 23.77  (far baseline)
    │                        │
    │   fsl ──────── fsr     │      y = 17.37  (far service line)
    │   │     ╎       │      │
    │   │     ╎       │      │      y = 11.885 (net)
    │   │     ╎       │      │
    │   nsl ──────── nsr     │      y = 6.40   (near service line)
    │                        │
    BL ──────────────────── BR      y = 0      (near baseline)

   x=0                    x=8.23   (singles width)
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np

COURT_LENGTH      = 23.77
SINGLES_WIDTH     = 8.23
DOUBLES_WIDTH     = 10.97
SERVICE_BOX_DEPTH = 6.40
NET_Y             = 11.885
CENTER_X          = SINGLES_WIDTH / 2

@dataclass(frozen=True)
class TemplateLine:
    name: str
    x1: float; y1: float; x2: float; y2: float

COURT_LINES: List[TemplateLine] = [
    TemplateLine("near_baseline",   0.0, 0.0,         SINGLES_WIDTH, 0.0),
    TemplateLine("far_baseline",    0.0, COURT_LENGTH, SINGLES_WIDTH, COURT_LENGTH),
    TemplateLine("left_sideline",   0.0, 0.0,          0.0,          COURT_LENGTH),
    TemplateLine("right_sideline",  SINGLES_WIDTH, 0.0, SINGLES_WIDTH, COURT_LENGTH),
    TemplateLine("near_service",    0.0, SERVICE_BOX_DEPTH,
                 SINGLES_WIDTH, SERVICE_BOX_DEPTH),
    TemplateLine("far_service",     0.0, COURT_LENGTH - SERVICE_BOX_DEPTH,
                 SINGLES_WIDTH, COURT_LENGTH - SERVICE_BOX_DEPTH),
    TemplateLine("center_service",  CENTER_X, SERVICE_BOX_DEPTH,
                 CENTER_X, COURT_LENGTH - SERVICE_BOX_DEPTH),
    TemplateLine("net",             0.0, NET_Y, SINGLES_WIDTH, NET_Y),
]

BOUNDARY_CORNERS = np.array([
    [0.0,           0.0],
    [SINGLES_WIDTH, 0.0],
    [SINGLES_WIDTH, COURT_LENGTH],
    [0.0,           COURT_LENGTH],
], dtype=np.float32)

# Expected Y ratios (fraction of court length) for horizontal lines
# Sorted far→near in world coords (top→bottom in image)
H_LINE_ROLES = [
    ("far_baseline",  1.0),     # y/L = 1.0
    ("far_service",   0.731),   # (23.77-6.40)/23.77
    ("net",           0.500),   # 11.885/23.77
    ("near_service",  0.269),   # 6.40/23.77
    ("near_baseline", 0.0),     # y/L = 0.0
]