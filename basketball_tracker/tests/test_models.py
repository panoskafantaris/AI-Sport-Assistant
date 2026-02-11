"""
Tests for data models.
"""
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import (
    BoundingBox, 
    Detection, 
    TrackedObject, 
    FrameData,
    VideoMetadata,
    TrackingResult
)


class TestBoundingBox:
    """Tests for BoundingBox class."""
    
    def test_creation(self):
        bbox = BoundingBox(x1=10, y1=20, x2=110, y2=220)
        assert bbox.x1 == 10
        assert bbox.y1 == 20
        assert bbox.x2 == 110
        assert bbox.y2 == 220
    
    def test_width_height(self, sample_bbox):
        assert sample_bbox.width == 100
        assert sample_bbox.height == 200
    
    def test_center(self, sample_bbox):
        cx, cy = sample_bbox.center
        assert cx == 150.0
        assert cy == 200.0
    
    def test_area(self, sample_bbox):
        assert sample_bbox.area == 20000
    
    def test_to_tuple(self, sample_bbox):
        assert sample_bbox.to_tuple() == (100, 100, 200, 300)
    
    def test_to_int_tuple(self):
        bbox = BoundingBox(x1=10.5, y1=20.7, x2=110.3, y2=220.9)
        assert bbox.to_int_tuple() == (10, 20, 110, 220)


class TestDetection:
    """Tests for Detection class."""
    
    def test_creation(self, sample_detection):
        assert sample_detection.confidence == 0.85
        assert sample_detection.class_id == 0
        assert sample_detection.class_name == "person"
    
    def test_bbox_access(self, sample_detection):
        assert sample_detection.bbox.width == 100


class TestTrackedObject:
    """Tests for TrackedObject class."""
    
    def test_creation(self, sample_tracked_object):
        assert sample_tracked_object.track_id == 1
        assert sample_tracked_object.confidence == 0.85
        assert len(sample_tracked_object.history) == 3
    
    def test_empty_history(self, sample_bbox):
        obj = TrackedObject(
            track_id=1,
            bbox=sample_bbox,
            confidence=0.9,
            class_id=0
        )
        assert obj.history == []


class TestFrameData:
    """Tests for FrameData class."""
    
    def test_creation(self, sample_frame_data):
        assert sample_frame_data.frame_number == 10
        assert sample_frame_data.timestamp_ms == 333.33
        assert len(sample_frame_data.tracked_objects) == 1
    
    def test_empty_frame(self):
        frame = FrameData(frame_number=0, timestamp_ms=0.0)
        assert frame.detections == []
        assert frame.tracked_objects == []


class TestVideoMetadata:
    """Tests for VideoMetadata class."""
    
    def test_creation(self, sample_video_metadata):
        assert sample_video_metadata.width == 1920
        assert sample_video_metadata.height == 1080
        assert sample_video_metadata.fps == 30.0
    
    def test_to_dict(self, sample_video_metadata):
        d = sample_video_metadata.to_dict()
        assert d["width"] == 1920
        assert d["fps"] == 30.0
        assert "filepath" in d


class TestTrackingResult:
    """Tests for TrackingResult class."""
    
    def test_creation(self, sample_video_metadata, sample_frame_data):
        result = TrackingResult(
            metadata=sample_video_metadata,
            frames=[sample_frame_data]
        )
        assert len(result.frames) == 1
    
    def test_to_dict(self, sample_video_metadata, sample_frame_data):
        result = TrackingResult(
            metadata=sample_video_metadata,
            frames=[sample_frame_data]
        )
        d = result.to_dict()
        
        assert "metadata" in d
        assert "frames" in d
        assert len(d["frames"]) == 1
        assert d["frames"][0]["frame_number"] == 10