"""
Tests for team classifier module (integration).
"""
import cv2
import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.team_classifier.classifier import TeamClassifier
from src.models import Team, TrackedObject, BoundingBox


@pytest.fixture
def team_classifier():
    """Create a team classifier instance."""
    return TeamClassifier(
        calibration_frames=5,
        enable_scene_detection=False  # Disable for easier testing
    )


@pytest.fixture
def team_classifier_with_scene():
    """Create a team classifier with scene detection."""
    return TeamClassifier(
        calibration_frames=5,
        enable_scene_detection=True
    )


def create_player_frame(jersey_color_bgr, bbox):
    """Create a frame with a player of given jersey color."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[:] = (34, 139, 34)  # Green court
    
    x1, y1, x2, y2 = bbox
    height = y2 - y1
    
    # Head
    head_y2 = int(y1 + height * 0.15)
    cv2.rectangle(frame, (x1, y1), (x2, head_y2), (180, 200, 230), -1)
    
    # Jersey
    jersey_y1 = head_y2
    jersey_y2 = int(y1 + height * 0.55)
    cv2.rectangle(frame, (x1, jersey_y1), (x2, jersey_y2), jersey_color_bgr, -1)
    
    # Legs
    cv2.rectangle(frame, (x1, jersey_y2), (x2, y2), (50, 50, 50), -1)
    
    return frame


def create_tracked_object(track_id, bbox):
    """Create a tracked object."""
    x1, y1, x2, y2 = bbox
    return TrackedObject(
        track_id=track_id,
        bbox=BoundingBox(x1, y1, x2, y2),
        confidence=0.9,
        class_id=0,
        class_name="person"
    )


@pytest.fixture
def red_player_frame():
    """Frame with red jersey player."""
    bbox = (100, 50, 200, 350)
    frame = create_player_frame((0, 0, 255), bbox)  # Red in BGR
    return frame, bbox


@pytest.fixture
def blue_player_frame():
    """Frame with blue jersey player."""
    bbox = (300, 50, 400, 350)
    frame = create_player_frame((255, 0, 0), bbox)  # Blue in BGR
    return frame, bbox


@pytest.fixture
def multi_player_frame():
    """Frame with multiple players of different teams."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[:] = (34, 139, 34)  # Green court
    
    # Red team players
    for x in [50, 150]:
        bbox = (x, 50, x + 80, 300)
        x1, y1, x2, y2 = bbox
        height = y2 - y1
        head_y2 = int(y1 + height * 0.15)
        jersey_y2 = int(y1 + height * 0.55)
        cv2.rectangle(frame, (x1, y1), (x2, head_y2), (180, 200, 230), -1)
        cv2.rectangle(frame, (x1, head_y2), (x2, jersey_y2), (0, 0, 255), -1)
        cv2.rectangle(frame, (x1, jersey_y2), (x2, y2), (50, 50, 50), -1)
    
    # Blue team players
    for x in [350, 450]:
        bbox = (x, 50, x + 80, 300)
        x1, y1, x2, y2 = bbox
        height = y2 - y1
        head_y2 = int(y1 + height * 0.15)
        jersey_y2 = int(y1 + height * 0.55)
        cv2.rectangle(frame, (x1, y1), (x2, head_y2), (180, 200, 230), -1)
        cv2.rectangle(frame, (x1, head_y2), (x2, jersey_y2), (255, 0, 0), -1)
        cv2.rectangle(frame, (x1, jersey_y2), (x2, y2), (50, 50, 50), -1)
    
    bboxes = [(50, 50, 130, 300), (150, 50, 230, 300), 
              (350, 50, 430, 300), (450, 50, 530, 300)]
    
    return frame, bboxes


class TestTeamClassifier:
    """Tests for TeamClassifier class."""
    
    def test_initialization(self, team_classifier):
        """Test classifier initializes correctly."""
        assert team_classifier.calibration_frames == 5
        assert not team_classifier.is_calibrated
    
    def test_not_calibrated_initially(self, team_classifier):
        """Test classifier is not calibrated initially."""
        assert not team_classifier.is_calibrated
    
    def test_classify_frame_during_calibration(self, team_classifier, red_player_frame):
        """Test classify_frame during calibration phase."""
        frame, bbox = red_player_frame
        tracked_obj = create_tracked_object(1, bbox)
        
        result = team_classifier.classify_frame(frame, [tracked_obj])
        
        # During calibration, team should still be UNKNOWN
        assert result[0].team == Team.UNKNOWN
    
    def test_calibration_completes(self, team_classifier, multi_player_frame):
        """Test calibration completes after enough frames."""
        frame, bboxes = multi_player_frame
        tracked_objects = [create_tracked_object(i, bbox) for i, bbox in enumerate(bboxes)]
        
        # Process enough frames for calibration
        for _ in range(10):
            team_classifier.classify_frame(frame, tracked_objects)
        
        assert team_classifier.is_calibrated
    
    def test_classification_after_calibration(self, team_classifier, multi_player_frame):
        """Test classification works after calibration."""
        frame, bboxes = multi_player_frame
        tracked_objects = [create_tracked_object(i, bbox) for i, bbox in enumerate(bboxes)]
        
        # Calibrate
        for _ in range(10):
            result = team_classifier.classify_frame(frame, tracked_objects)
        
        # After calibration, should have team assignments
        teams_assigned = [obj.team for obj in result if obj.team != Team.UNKNOWN]
        assert len(teams_assigned) > 0
    
    def test_team_colors_after_calibration(self, team_classifier, multi_player_frame):
        """Test team colors are available after calibration."""
        frame, bboxes = multi_player_frame
        tracked_objects = [create_tracked_object(i, bbox) for i, bbox in enumerate(bboxes)]
        
        # Calibrate
        for _ in range(10):
            team_classifier.classify_frame(frame, tracked_objects)
        
        colors = team_classifier.team_colors
        assert len(colors) > 0
    
    def test_reset_clears_calibration(self, team_classifier, multi_player_frame):
        """Test reset clears calibration state."""
        frame, bboxes = multi_player_frame
        tracked_objects = [create_tracked_object(i, bbox) for i, bbox in enumerate(bboxes)]
        
        # Calibrate
        for _ in range(10):
            team_classifier.classify_frame(frame, tracked_objects)
        
        assert team_classifier.is_calibrated
        
        # Reset
        team_classifier.reset()
        
        assert not team_classifier.is_calibrated
    
    def test_get_team_stats(self, team_classifier, multi_player_frame):
        """Test team statistics calculation."""
        frame, bboxes = multi_player_frame
        tracked_objects = [create_tracked_object(i, bbox) for i, bbox in enumerate(bboxes)]
        
        # Calibrate
        for _ in range(10):
            result = team_classifier.classify_frame(frame, tracked_objects)
        
        stats = team_classifier.get_team_stats(result)
        
        assert Team.UNKNOWN in stats
        total = sum(stats.values())
        assert total == len(tracked_objects)
    
    def test_jersey_color_assigned(self, team_classifier, multi_player_frame):
        """Test jersey color RGB is assigned to tracked objects."""
        frame, bboxes = multi_player_frame
        tracked_objects = [create_tracked_object(i, bbox) for i, bbox in enumerate(bboxes)]
        
        # Calibrate and classify
        for _ in range(10):
            result = team_classifier.classify_frame(frame, tracked_objects)
        
        # Check some objects have jersey colors
        colors_assigned = [obj.jersey_color_rgb for obj in result if obj.jersey_color_rgb is not None]
        assert len(colors_assigned) > 0


class TestTeamClassifierWithSceneDetection:
    """Tests for TeamClassifier with scene detection enabled."""
    
    def test_scene_change_triggers_recalibration(self, team_classifier_with_scene, multi_player_frame):
        """Test scene change triggers recalibration."""
        frame, bboxes = multi_player_frame
        tracked_objects = [create_tracked_object(i, bbox) for i, bbox in enumerate(bboxes)]
        
        # Calibrate
        for _ in range(10):
            team_classifier_with_scene.classify_frame(frame, tracked_objects)
        
        assert team_classifier_with_scene.is_calibrated
        
        # Create a very different frame (scene change)
        different_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        different_frame[:] = (200, 200, 200)  # Gray
        
        # Process several frames to pass cooldown, then the different frame
        for _ in range(20):
            team_classifier_with_scene.classify_frame(frame, tracked_objects)
        
        # Now trigger scene change
        team_classifier_with_scene.classify_frame(different_frame, [])
        
        # Should start recalibration
        assert not team_classifier_with_scene.is_calibrated