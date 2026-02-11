"""
Data models for basketball tracker.
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple


class Team(Enum):
    """Team classification for tracked players."""
    UNKNOWN = 0
    TEAM_A = 1
    TEAM_B = 2
    REFEREE = 3
    
    def to_string(self) -> str:
        """Convert to readable string."""
        return self.name.replace("_", " ").title()


@dataclass
class BoundingBox:
    """Represents a bounding box in pixel coordinates."""
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
    def center(self) -> Tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)
    
    @property
    def area(self) -> float:
        return self.width * self.height
    
    def to_tuple(self) -> Tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)
    
    def to_int_tuple(self) -> Tuple[int, int, int, int]:
        return (int(self.x1), int(self.y1), int(self.x2), int(self.y2))


@dataclass
class Detection:
    """A single detection in a frame."""
    bbox: BoundingBox
    confidence: float
    class_id: int
    class_name: str = "person"


@dataclass
class TrackedObject:
    """A tracked object with persistent ID."""
    track_id: int
    bbox: BoundingBox
    confidence: float
    class_id: int
    class_name: str = "person"
    
    # Optional: track history for trajectory analysis
    history: List[Tuple[float, float]] = field(default_factory=list)
    
    # Team classification (Phase 2)
    team: Team = Team.UNKNOWN
    jersey_color_rgb: Optional[Tuple[int, int, int]] = None


@dataclass
class FrameData:
    """All data associated with a single frame."""
    frame_number: int
    timestamp_ms: float
    detections: List[Detection] = field(default_factory=list)
    tracked_objects: List[TrackedObject] = field(default_factory=list)


@dataclass
class VideoMetadata:
    """Metadata about the input video."""
    filepath: str
    width: int
    height: int
    fps: float
    total_frames: int
    duration_seconds: float
    
    def to_dict(self) -> dict:
        return {
            "filepath": self.filepath,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "total_frames": self.total_frames,
            "duration_seconds": self.duration_seconds
        }


@dataclass
class TrackingResult:
    """Complete tracking result for a video."""
    metadata: VideoMetadata
    frames: List[FrameData] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON export."""
        return {
            "metadata": self.metadata.to_dict(),
            "frames": [
                {
                    "frame_number": f.frame_number,
                    "timestamp_ms": f.timestamp_ms,
                    "tracked_objects": [
                        {
                            "track_id": obj.track_id,
                            "bbox": obj.bbox.to_tuple(),
                            "confidence": obj.confidence,
                            "class_name": obj.class_name,
                            "team": obj.team.name,
                            "jersey_color_rgb": obj.jersey_color_rgb
                        }
                        for obj in f.tracked_objects
                    ]
                }
                for f in self.frames
            ]
        }