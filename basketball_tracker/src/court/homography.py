"""
Court homography – maps image pixels ↔ real-world court metres.

Standard tennis court key points (world coordinates in metres,
origin at near-left corner of baseline):

    BL=(0,0)     BR=(W,0)         near baseline
    SL=(0,SBD)   SR=(W,SBD)       near service line
    NET_L=(0,N)  NET_R=(W,N)      net
    SFL=(0,L-SBD) SFR=(W,L-SBD)  far service line
    FL=(0,L)     FR=(W,L)         far baseline

We compute H from (at least) 4 corresponding pixel ↔ world pairs.
"""
from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np

from ..models.court import CourtBoundary, CourtHomography
import config


# World-space corners of the outer singles court (metres)
# Order: near-left, near-right, far-right, far-left
_WORLD_CORNERS_SINGLES = np.array([
    [0.0,              0.0             ],  # near-left baseline
    [config.COURT_WIDTH_SINGLE, 0.0   ],  # near-right baseline
    [config.COURT_WIDTH_SINGLE, config.COURT_LENGTH],  # far-right
    [0.0,              config.COURT_LENGTH],           # far-left
], dtype=np.float32)

_WORLD_CORNERS_DOUBLES = np.array([
    [0.0,              0.0             ],
    [config.COURT_WIDTH_DOUBLE, 0.0   ],
    [config.COURT_WIDTH_DOUBLE, config.COURT_LENGTH],
    [0.0,              config.COURT_LENGTH],
], dtype=np.float32)


class HomographyCalc:
    """Computes and wraps the court homography matrix."""

    def __init__(self, doubles: bool = False):
        self._world = _WORLD_CORNERS_DOUBLES if doubles else _WORLD_CORNERS_SINGLES
        self._homography = CourtHomography()

    def compute(self, boundary: CourtBoundary) -> Optional[CourtHomography]:
        """
        Compute homography from detected court corners to world coords.

        Args:
            boundary: CourtBoundary with corners in TL, TR, BR, BL order.

        Returns:
            CourtHomography or None on failure.
        """
        if not boundary.is_valid():
            return None

        src = boundary.corners.astype(np.float32)  # image points
        dst = self._world                           # world points

        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        if H is None:
            return None

        H_inv, _ = cv2.findHomography(dst, src, cv2.RANSAC, 5.0)
        self._homography = CourtHomography(H=H, H_inv=H_inv)
        return self._homography

    @property
    def homography(self) -> CourtHomography:
        return self._homography

    def world_rect_to_image(self, world_corners: np.ndarray) -> np.ndarray:
        """
        Project a rectangle defined in world coords back to image pixels.
        Useful for drawing service boxes, etc.
        """
        if self._homography.H_inv is None:
            raise RuntimeError("Homography not computed")
        pts = world_corners.reshape(-1, 1, 2).astype(np.float32)
        return cv2.perspectiveTransform(pts, self._homography.H_inv)

    def service_box_corners(self) -> Optional[np.ndarray]:
        """Return image-space corners of the 4 service boxes."""
        if self._homography.H_inv is None:
            return None
        sbd = config.SERVICE_BOX_DEPTH
        net = config.NET_OFFSET
        cw  = config.COURT_WIDTH_SINGLE / 2

        world_pts = np.array([
            [0,   net - sbd], [cw,  net - sbd],  # near-left service box TL, TR
            [cw,  net      ], [0,   net      ],   # near-left service box BR, BL
            [cw,  net - sbd], [config.COURT_WIDTH_SINGLE, net - sbd],
            [config.COURT_WIDTH_SINGLE, net], [cw, net],
        ], dtype=np.float32)

        img_pts = cv2.perspectiveTransform(
            world_pts.reshape(-1, 1, 2), self._homography.H_inv
        ).reshape(-1, 2)
        return img_pts