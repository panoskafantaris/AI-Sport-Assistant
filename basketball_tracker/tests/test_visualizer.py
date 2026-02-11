"""
Tests for visualizer.
"""
import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.visualizer import Visualizer
from src.models import Detection, TrackedObject, BoundingBox, Team


class TestVisualizer:
    """Tests for Visualizer class."""
    
    @pytest.fixture
    def visualizer(self):
        return Visualizer()
    
    def test_initialization(self, visualizer):
        assert visualizer.box_thickness == 2
        assert visualizer.font_scale == 0.6
        assert visualizer.draw_trajectory is True
    
    def test_get_color_consistent(self, visualizer):
        """Same track ID should always get same color."""
        color1 = visualizer._get_color(5)
        color2 = visualizer._get_color(5)
        
        assert color1 == color2
    
    def test_get_color_different_ids(self, visualizer):
        """Different IDs should get different colors (within palette)."""
        color1 = visualizer._get_color(0)
        color2 = visualizer._get_color(1)
        
        assert color1 != color2
    
    def test_draw_detections_output_shape(self, visualizer, sample_frame, sample_detection):
        annotated = visualizer.draw_detections(sample_frame, [sample_detection])
        
        assert annotated.shape == sample_frame.shape
        assert annotated.dtype == sample_frame.dtype
    
    def test_draw_detections_modifies_frame(self, visualizer, sample_frame, sample_detection):
        original = sample_frame.copy()
        annotated = visualizer.draw_detections(sample_frame, [sample_detection])
        
        # Original should be unchanged
        assert np.array_equal(original, sample_frame)
        
        # Annotated should be different
        assert not np.array_equal(annotated, sample_frame)
    
    def test_draw_detections_empty_list(self, visualizer, sample_frame):
        annotated = visualizer.draw_detections(sample_frame, [])
        
        # Should return unchanged copy
        assert annotated.shape == sample_frame.shape
    
    def test_draw_tracks_output_shape(self, visualizer, sample_frame, sample_tracked_object):
        annotated = visualizer.draw_tracks(sample_frame, [sample_tracked_object])
        
        assert annotated.shape == sample_frame.shape
    
    def test_draw_tracks_with_history(self, visualizer, sample_frame):
        """Test trajectory drawing with history."""
        obj = TrackedObject(
            track_id=1,
            bbox=BoundingBox(100, 100, 200, 300),
            confidence=0.9,
            class_id=0,
            history=[(150, 200), (160, 210), (170, 220), (180, 230)]
        )
        
        annotated = visualizer.draw_tracks(sample_frame, [obj])
        
        assert annotated.shape == sample_frame.shape
        # Frame should be modified (trajectory drawn)
        assert not np.array_equal(annotated, sample_frame)
    
    def test_draw_multiple_tracks(self, visualizer, sample_frame, multiple_tracked_objects):
        annotated = visualizer.draw_tracks(sample_frame, multiple_tracked_objects)
        
        assert annotated.shape == sample_frame.shape
    
    def test_draw_frame_info(self, visualizer, sample_frame):
        annotated = visualizer.draw_frame_info(
            sample_frame,
            frame_number=42,
            num_tracks=5,
            fps=30.0
        )
        
        assert annotated.shape == sample_frame.shape
        # Should have text drawn
        assert not np.array_equal(annotated, sample_frame)
    
    def test_draw_frame_info_no_fps(self, visualizer, sample_frame):
        annotated = visualizer.draw_frame_info(
            sample_frame,
            frame_number=42,
            num_tracks=5
        )
        
        assert annotated.shape == sample_frame.shape


class TestVisualizerConfiguration:
    """Test visualizer with different configurations."""
    
    def test_no_trajectory(self, sample_frame, sample_tracked_object):
        visualizer = Visualizer(draw_trajectory=False)
        
        annotated = visualizer.draw_tracks(sample_frame, [sample_tracked_object])
        
        assert annotated.shape == sample_frame.shape
    
    def test_custom_thickness(self, sample_frame, sample_tracked_object):
        visualizer = Visualizer(box_thickness=5)
        
        annotated = visualizer.draw_tracks(sample_frame, [sample_tracked_object])
        
        assert annotated.shape == sample_frame.shape
    
    def test_custom_trajectory_length(self, sample_frame):
        visualizer = Visualizer(trajectory_length=5)
        
        obj = TrackedObject(
            track_id=1,
            bbox=BoundingBox(100, 100, 200, 300),
            confidence=0.9,
            class_id=0,
            history=[(100 + i*10, 200 + i*5) for i in range(20)]
        )
        
        annotated = visualizer.draw_tracks(sample_frame, [obj])
        
        assert annotated.shape == sample_frame.shape


class TestVisualizerTeamColors:
    """Test visualizer team color functionality."""
    
    @pytest.fixture
    def visualizer(self):
        return Visualizer(use_team_colors=True)
    
    def test_set_team_colors(self, visualizer):
        """Test setting custom team colors."""
        colors = {
            Team.TEAM_A: (255, 0, 0),
            Team.TEAM_B: (0, 0, 255),
        }
        
        visualizer.set_team_colors(colors)
        
        # Colors should be stored (converted to BGR)
        assert Team.TEAM_A in visualizer._team_colors
        assert Team.TEAM_B in visualizer._team_colors
    
    def test_get_team_color(self, visualizer):
        """Test getting team color."""
        color = visualizer._get_team_color(Team.TEAM_A)
        
        assert color is not None
        assert len(color) == 3
    
    def test_draw_tracks_with_team_colors(self, visualizer, sample_frame, tracked_objects_with_teams):
        """Test drawing tracks with team assignments."""
        annotated = visualizer.draw_tracks(sample_frame, tracked_objects_with_teams)
        
        assert annotated.shape == sample_frame.shape
        assert not np.array_equal(annotated, sample_frame)
    
    def test_draw_tracks_unknown_team_uses_fallback(self, visualizer, sample_frame):
        """Test that UNKNOWN team uses fallback color."""
        obj = TrackedObject(
            track_id=1,
            bbox=BoundingBox(100, 100, 200, 300),
            confidence=0.9,
            class_id=0,
            team=Team.UNKNOWN
        )
        
        annotated = visualizer.draw_tracks(sample_frame, [obj])
        
        assert annotated.shape == sample_frame.shape
    
    def test_draw_frame_info_with_team_counts(self, visualizer, sample_frame):
        """Test drawing frame info with team counts."""
        team_counts = {
            Team.TEAM_A: 5,
            Team.TEAM_B: 4,
            Team.REFEREE: 3,
            Team.UNKNOWN: 0
        }
        
        annotated = visualizer.draw_frame_info(
            sample_frame,
            frame_number=10,
            num_tracks=12,
            team_counts=team_counts
        )
        
        assert annotated.shape == sample_frame.shape
        assert not np.array_equal(annotated, sample_frame)
    
    def test_disable_team_colors(self, sample_frame):
        """Test visualizer with team colors disabled."""
        visualizer = Visualizer(use_team_colors=False)
        
        obj = TrackedObject(
            track_id=1,
            bbox=BoundingBox(100, 100, 200, 300),
            confidence=0.9,
            class_id=0,
            team=Team.TEAM_A
        )
        
        annotated = visualizer.draw_tracks(sample_frame, [obj])
        
        assert annotated.shape == sample_frame.shape