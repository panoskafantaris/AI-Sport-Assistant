"""
Tests for scene detector module.
"""
import cv2
import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.team_classifier.scene_detector import SceneDetector


@pytest.fixture
def scene_detector():
    """Create a scene detector instance."""
    return SceneDetector(threshold=0.4, cooldown_frames=5)


@pytest.fixture
def court_frame():
    """Create a basketball court-like frame."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    # Orange/brown court color
    frame[:] = (30, 100, 180)
    # Add some court lines
    cv2.line(frame, (0, 240), (640, 240), (255, 255, 255), 3)
    cv2.circle(frame, (320, 240), 60, (255, 255, 255), 3)
    return frame


@pytest.fixture
def crowd_frame():
    """Create a crowd/different scene frame."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    # Dark background with colorful spots (crowd)
    frame[:] = (40, 40, 60)
    # Random colored rectangles to simulate crowd
    for _ in range(50):
        x, y = np.random.randint(0, 600), np.random.randint(0, 440)
        color = tuple(np.random.randint(0, 255, 3).tolist())
        cv2.rectangle(frame, (x, y), (x+40, y+40), color, -1)
    return frame


@pytest.fixture
def similar_court_frame(court_frame):
    """Create a slightly different court frame."""
    frame = court_frame.copy()
    # Add slight variation
    cv2.rectangle(frame, (100, 100), (150, 200), (0, 0, 255), -1)  # Player
    return frame


class TestSceneDetector:
    """Tests for SceneDetector class."""
    
    def test_initialization(self, scene_detector):
        """Test scene detector initializes correctly."""
        assert scene_detector.threshold == 0.4
        assert scene_detector.cooldown_frames == 5
    
    def test_first_frame_is_scene_change(self, scene_detector, court_frame):
        """Test that first frame is always a scene change."""
        result = scene_detector.is_scene_change(court_frame)
        assert result is True
    
    def test_similar_frames_no_change(self, scene_detector, court_frame, similar_court_frame):
        """Test that similar frames don't trigger scene change."""
        # First frame
        scene_detector.is_scene_change(court_frame)
        
        # Wait for cooldown
        for _ in range(6):
            scene_detector.is_scene_change(court_frame)
        
        # Similar frame should not be a scene change
        result = scene_detector.is_scene_change(similar_court_frame)
        assert result is False
    
    def test_different_scene_triggers_change(self, scene_detector, court_frame, crowd_frame):
        """Test that very different frame triggers scene change."""
        # First frame
        scene_detector.is_scene_change(court_frame)
        
        # Wait for cooldown
        for _ in range(6):
            scene_detector.is_scene_change(court_frame)
        
        # Very different frame should trigger scene change
        result = scene_detector.is_scene_change(crowd_frame)
        assert result is True
    
    def test_cooldown_prevents_immediate_change(self, scene_detector, court_frame, crowd_frame):
        """Test cooldown prevents rapid scene changes."""
        # First frame
        scene_detector.is_scene_change(court_frame)
        
        # Immediately try different scene (within cooldown)
        result = scene_detector.is_scene_change(crowd_frame)
        assert result is False  # Should be blocked by cooldown
    
    def test_reset_clears_state(self, scene_detector, court_frame):
        """Test reset clears detector state."""
        # Process a frame
        scene_detector.is_scene_change(court_frame)
        
        # Reset
        scene_detector.reset()
        
        # First frame after reset should be scene change
        result = scene_detector.is_scene_change(court_frame)
        assert result is True
    
    def test_frames_since_change_counter(self, scene_detector, court_frame):
        """Test frames since change counter."""
        # First frame
        scene_detector.is_scene_change(court_frame)
        assert scene_detector.get_frames_since_change() == 0
        
        # Process more frames
        scene_detector.is_scene_change(court_frame)
        scene_detector.is_scene_change(court_frame)
        
        assert scene_detector.get_frames_since_change() >= 2
    
    def test_compute_histogram_returns_array(self, scene_detector, court_frame):
        """Test histogram computation returns valid array."""
        hist = scene_detector._compute_histogram(court_frame)
        
        assert hist is not None
        assert len(hist) > 0
        assert isinstance(hist, np.ndarray)
    
    def test_histogram_difference_identical(self, scene_detector, court_frame):
        """Test histogram difference for identical frames."""
        hist1 = scene_detector._compute_histogram(court_frame)
        hist2 = scene_detector._compute_histogram(court_frame)
        
        diff = scene_detector._histogram_difference(hist1, hist2)
        
        assert diff < 0.1  # Should be very small for identical frames
    
    def test_histogram_difference_different(self, scene_detector, court_frame, crowd_frame):
        """Test histogram difference for different frames."""
        hist1 = scene_detector._compute_histogram(court_frame)
        hist2 = scene_detector._compute_histogram(crowd_frame)
        
        diff = scene_detector._histogram_difference(hist1, hist2)
        
        assert diff > 0.3  # Should be significant for different frames