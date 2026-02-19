"""
Court-related data models.
"""
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np


@dataclass
class CourtLine:
    """A detected line segment on the court."""
    x1: float
    y1: float
    x2: float
    y2: float
    label: str = ""          # e.g. "baseline", "sideline", "service"

    @property
    def midpoint(self) -> Tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    @property
    def length(self) -> float:
        return float(np.hypot(self.x2 - self.x1, self.y2 - self.y1))

    @property
    def angle_deg(self) -> float:
        return float(np.degrees(np.arctan2(self.y2 - self.y1, self.x2 - self.x1)))


@dataclass
class CourtBoundary:
    """
    Precise tennis court boundary defined by key point intersections.

    Points ordered: near-left, near-right, far-right, far-left
    (i.e. two baselines + two sidelines forming the outer rectangle).
    """
    corners: np.ndarray                    # shape (4,2) float32, image coords
    inner_lines: List[CourtLine] = field(default_factory=list)
    confidence: float = 0.0

    # Optional: service box corners, net line, etc.
    net_line: Optional[CourtLine] = None
    service_corners: Optional[np.ndarray] = None  # 4 pts for service boxes

    def is_valid(self) -> bool:
        return self.corners is not None and self.corners.shape == (4, 2)

    def contains_point(self, x: float, y: float, margin: float = 0.0) -> bool:
        """Check if a pixel coordinate is inside the court polygon."""
        import cv2
        if not self.is_valid():
            return False
        pts = self.corners.reshape((-1, 1, 2)).astype(np.float32)
        dist = cv2.pointPolygonTest(pts, (x, y), measureDist=False)
        return dist >= -margin


@dataclass
class CourtHomography:
    """
    Mapping between image pixels and real-world court coordinates (meters).
    """
    H: Optional[np.ndarray] = None        # 3×3 homography matrix (image → world)
    H_inv: Optional[np.ndarray] = None    # world → image

    def is_ready(self) -> bool:
        return self.H is not None

    def image_to_world(self, px: float, py: float) -> Tuple[float, float]:
        """Convert image pixel to court metres."""
        if self.H is None:
            raise RuntimeError("Homography not calibrated")
        pt = np.array([[[px, py]]], dtype=np.float32)
        import cv2
        world = cv2.perspectiveTransform(pt, self.H)[0][0]
        return float(world[0]), float(world[1])

    def world_to_image(self, wx: float, wy: float) -> Tuple[float, float]:
        if self.H_inv is None:
            raise RuntimeError("Homography not calibrated")
        pt = np.array([[[wx, wy]]], dtype=np.float32)
        import cv2
        img = cv2.perspectiveTransform(pt, self.H_inv)[0][0]
        return float(img[0]), float(img[1])