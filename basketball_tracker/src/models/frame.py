"""
Frame-level and aggregate result models.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

from .court  import CourtBoundary, CourtHomography
from .player import Player
from .ball   import Ball, BallTrajectory, LandingEstimate


@dataclass
class VideoMetadata:
    width: int
    height: int
    fps: float
    total_frames: int
    duration_s: float
    path: str = ""


@dataclass
class RallyStats:
    rally_id: int
    start_frame: int
    end_frame: int
    shot_count: int = 0
    winner_role: str = "unknown"
    max_ball_speed_ms: float = 0.0


@dataclass
class FrameData:
    """All data extracted for a single video frame."""
    frame_number: int
    timestamp_ms: float

    players: List[Player] = field(default_factory=list)
    ball: Optional[Ball] = None
    trajectory: Optional[BallTrajectory] = None
    landing: Optional[LandingEstimate] = None

    court: Optional[CourtBoundary] = None
    homography: Optional[CourtHomography] = None

    # Per-frame stats (populated by stats modules)
    extras: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "frame": self.frame_number,
            "timestamp_ms": round(self.timestamp_ms, 1),
            "players": [p.to_dict() for p in self.players],
            "ball": self.ball.to_dict() if self.ball else None,
            "landing": self.landing.to_dict() if self.landing else None,
            "extras": self.extras,
        }


@dataclass
class TrackingResult:
    """Aggregate result for a full video."""
    metadata: VideoMetadata
    frames: List[FrameData] = field(default_factory=list)
    rallies: List[RallyStats] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "video": {
                "path": self.metadata.path,
                "width": self.metadata.width,
                "height": self.metadata.height,
                "fps": self.metadata.fps,
                "total_frames": self.metadata.total_frames,
                "duration_s": round(self.metadata.duration_s, 2),
            },
            "frames": [f.to_dict() for f in self.frames],
            "rallies": [
                {
                    "id": r.rally_id,
                    "start_frame": r.start_frame,
                    "end_frame": r.end_frame,
                    "shots": r.shot_count,
                    "max_ball_speed_ms": round(r.max_ball_speed_ms, 2),
                }
                for r in self.rallies
            ],
        }