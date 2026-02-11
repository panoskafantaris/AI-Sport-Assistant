"""
Pytest fixtures for basketball tracker tests.
"""
import cv2
import numpy as np
import pytest
import tempfile
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import BoundingBox, Detection, TrackedObject, FrameData, VideoMetadata, Team


@pytest.fixture
def sample_frame():
    """Create a sample BGR frame for testing."""
    # Create a 640x480 frame with some shapes to detect
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[:] = (100, 120, 100)  # Grayish-green background
    
    # Draw some rectangles to simulate people
    cv2.rectangle(frame, (100, 100), (180, 300), (0, 0, 255), -1)
    cv2.rectangle(frame, (300, 150), (380, 350), (255, 0, 0), -1)
    cv2.rectangle(frame, (500, 120), (580, 320), (0, 255, 0), -1)
    
    return frame


@pytest.fixture
def sample_frame_small():
    """Create a small frame for quick tests."""
    return np.zeros((240, 320, 3), dtype=np.uint8)


@pytest.fixture
def sample_bbox():
    """Create a sample bounding box."""
    return BoundingBox(x1=100, y1=100, x2=200, y2=300)


@pytest.fixture
def sample_detection(sample_bbox):
    """Create a sample detection."""
    return Detection(
        bbox=sample_bbox,
        confidence=0.85,
        class_id=0,
        class_name="person"
    )


@pytest.fixture
def sample_tracked_object(sample_bbox):
    """Create a sample tracked object."""
    return TrackedObject(
        track_id=1,
        bbox=sample_bbox,
        confidence=0.85,
        class_id=0,
        class_name="person",
        history=[(150.0, 200.0), (152.0, 202.0), (154.0, 204.0)]
    )


@pytest.fixture
def sample_frame_data(sample_tracked_object):
    """Create sample frame data."""
    return FrameData(
        frame_number=10,
        timestamp_ms=333.33,
        tracked_objects=[sample_tracked_object]
    )


@pytest.fixture
def sample_video_metadata():
    """Create sample video metadata."""
    return VideoMetadata(
        filepath="/path/to/video.mp4",
        width=1920,
        height=1080,
        fps=30.0,
        total_frames=900,
        duration_seconds=30.0
    )


@pytest.fixture
def temp_video_file():
    """Create a temporary video file for testing."""
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
        temp_path = Path(f.name)
    
    # Create a simple video with OpenCV
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(temp_path), fourcc, 30.0, (640, 480))
    
    # Write 30 frames (1 second)
    for i in range(30):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # Add moving rectangle
        x = 100 + i * 10
        cv2.rectangle(frame, (x, 100), (x + 80, 300), (0, 255, 0), -1)
        writer.write(frame)
    
    writer.release()
    
    yield temp_path
    
    # Cleanup
    if temp_path.exists():
        temp_path.unlink()


@pytest.fixture
def temp_output_dir():
    """Create a temporary output directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def multiple_tracked_objects():
    """Create multiple tracked objects for testing."""
    return [
        TrackedObject(
            track_id=1,
            bbox=BoundingBox(100, 100, 180, 300),
            confidence=0.9,
            class_id=0,
            class_name="person",
            history=[(140, 200)]
        ),
        TrackedObject(
            track_id=2,
            bbox=BoundingBox(300, 150, 380, 350),
            confidence=0.85,
            class_id=0,
            class_name="person",
            history=[(340, 250)]
        ),
        TrackedObject(
            track_id=3,
            bbox=BoundingBox(500, 120, 580, 320),
            confidence=0.88,
            class_id=0,
            class_name="person",
            history=[(540, 220)]
        )
    ]


@pytest.fixture
def tracked_objects_with_teams():
    """Create tracked objects with team assignments."""
    return [
        TrackedObject(
            track_id=1,
            bbox=BoundingBox(100, 100, 180, 300),
            confidence=0.9,
            class_id=0,
            class_name="person",
            history=[(140, 200)],
            team=Team.TEAM_A,
            jersey_color_rgb=(255, 50, 50)
        ),
        TrackedObject(
            track_id=2,
            bbox=BoundingBox(300, 150, 380, 350),
            confidence=0.85,
            class_id=0,
            class_name="person",
            history=[(340, 250)],
            team=Team.TEAM_B,
            jersey_color_rgb=(50, 50, 255)
        ),
        TrackedObject(
            track_id=3,
            bbox=BoundingBox(500, 120, 580, 320),
            confidence=0.88,
            class_id=0,
            class_name="person",
            history=[(540, 220)],
            team=Team.REFEREE,
            jersey_color_rgb=(128, 128, 128)
        )
    ]


@pytest.fixture
def basketball_court_frame():
    """Create a frame simulating a basketball court with players."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    # Court color (orange-brown)
    frame[:] = (30, 100, 180)
    
    # Court lines
    cv2.line(frame, (0, 240), (640, 240), (255, 255, 255), 2)
    cv2.circle(frame, (320, 240), 50, (255, 255, 255), 2)
    
    # Red team players
    cv2.rectangle(frame, (100, 100), (160, 280), (0, 0, 255), -1)
    cv2.rectangle(frame, (200, 150), (260, 330), (0, 0, 255), -1)
    
    # Blue team players
    cv2.rectangle(frame, (400, 100), (460, 280), (255, 0, 0), -1)
    cv2.rectangle(frame, (500, 150), (560, 330), (255, 0, 0), -1)
    
    return frame