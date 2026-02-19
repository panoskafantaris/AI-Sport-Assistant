"""
Player-related data models.
"""
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, Tuple


class PlayerRole(Enum):
    PLAYER_NEAR  = "player_near"   # Player on near (bottom) side
    PLAYER_FAR   = "player_far"    # Player on far (top) side
    BALL_BOY     = "ball_boy"
    UMPIRE       = "umpire"
    LINE_JUDGE   = "line_judge"
    SPECTATOR    = "spectator"
    UNKNOWN      = "unknown"


@dataclass
class BoundingBox:
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> Tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    @property
    def foot_center(self) -> Tuple[float, float]:
        """Bottom-center — better for court position."""
        return ((self.x1 + self.x2) / 2, self.y2)

    def to_int_tuple(self) -> Tuple[int, int, int, int]:
        return (int(self.x1), int(self.y1), int(self.x2), int(self.y2))


@dataclass
class Player:
    """A tracked player across frames."""
    track_id: int
    bbox: BoundingBox
    confidence: float
    role: PlayerRole = PlayerRole.UNKNOWN

    # Pixel history of foot_center positions
    position_history: List[Tuple[float, float]] = field(default_factory=list)

    # Real-world positions (metres) from homography
    world_position_history: List[Tuple[float, float]] = field(default_factory=list)

    # Kinematics (populated by stats module)
    speed_px_per_frame: float = 0.0
    speed_ms: float = 0.0           # metres/second
    acceleration_ms2: float = 0.0

    # Pose keypoints (17×3: x,y,conf per joint) from YOLOv8-pose
    pose_keypoints: Optional[object] = None

    def to_dict(self) -> dict:
        cx, cy = self.bbox.center
        return {
            "track_id": self.track_id,
            "role": self.role.value,
            "bbox": list(self.bbox.to_int_tuple()),
            "center": [round(cx, 1), round(cy, 1)],
            "confidence": round(self.confidence, 3),
            "speed_ms": round(self.speed_ms, 2),
        }