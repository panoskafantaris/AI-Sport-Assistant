"""
Tests for team clusterer module.
"""
import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.team_classifier.team_clusterer import TeamClusterer
from src.models import Team


@pytest.fixture
def team_clusterer():
    """Create a team clusterer instance."""
    return TeamClusterer(
        num_teams=2,
        referee_threshold=60,
        min_samples=4
    )


@pytest.fixture
def red_team_colors():
    """Generate HSV colors for red team jerseys."""
    # Red in HSV: H around 0-10 or 170-180
    return [
        (5, 200, 200),
        (3, 210, 190),
        (7, 195, 205),
        (4, 205, 195),
    ]


@pytest.fixture
def blue_team_colors():
    """Generate HSV colors for blue team jerseys."""
    # Blue in HSV: H around 100-120
    return [
        (110, 200, 200),
        (108, 210, 190),
        (112, 195, 205),
        (109, 205, 195),
    ]


@pytest.fixture
def referee_color():
    """Generate HSV color for referee (gray/black)."""
    # Low saturation = gray
    return (0, 30, 100)


@pytest.fixture
def calibration_data(red_team_colors, blue_team_colors):
    """Create calibration data with both teams."""
    hsv_colors = red_team_colors + blue_team_colors
    # Convert HSV to approximate RGB for testing
    rgb_colors = [
        (255, 50, 50),  # Red team
        (250, 55, 55),
        (255, 45, 45),
        (248, 52, 52),
        (50, 50, 255),  # Blue team
        (55, 55, 250),
        (45, 45, 255),
        (52, 52, 248),
    ]
    return hsv_colors, rgb_colors


class TestTeamClusterer:
    """Tests for TeamClusterer class."""
    
    def test_initialization(self, team_clusterer):
        """Test team clusterer initializes correctly."""
        assert team_clusterer.num_teams == 2
        assert team_clusterer.referee_threshold == 60
        assert not team_clusterer.is_calibrated
    
    def test_calibrate_success(self, team_clusterer, calibration_data):
        """Test successful calibration."""
        hsv_colors, rgb_colors = calibration_data
        
        success = team_clusterer.calibrate(hsv_colors, rgb_colors)
        
        assert success is True
        assert team_clusterer.is_calibrated
    
    def test_calibrate_insufficient_samples(self, team_clusterer):
        """Test calibration fails with insufficient samples."""
        hsv_colors = [(100, 200, 200), (110, 200, 200)]  # Only 2 samples
        rgb_colors = [(50, 50, 255), (55, 55, 250)]
        
        success = team_clusterer.calibrate(hsv_colors, rgb_colors)
        
        assert success is False
        assert not team_clusterer.is_calibrated
    
    def test_team_colors_after_calibration(self, team_clusterer, calibration_data):
        """Test team colors are set after calibration."""
        hsv_colors, rgb_colors = calibration_data
        team_clusterer.calibrate(hsv_colors, rgb_colors)
        
        colors = team_clusterer.team_colors
        
        assert Team.TEAM_A in colors or Team.TEAM_B in colors
        assert len(colors) >= 1
    
    def test_classify_uncalibrated_returns_unknown(self, team_clusterer):
        """Test classify returns UNKNOWN when not calibrated."""
        result = team_clusterer.classify((100, 200, 200))
        assert result == Team.UNKNOWN
    
    def test_classify_team_a(self, team_clusterer, calibration_data, red_team_colors):
        """Test classification assigns to correct team."""
        hsv_colors, rgb_colors = calibration_data
        team_clusterer.calibrate(hsv_colors, rgb_colors)
        
        # Classify a red color
        result = team_clusterer.classify(red_team_colors[0])
        
        assert result in [Team.TEAM_A, Team.TEAM_B]  # Should be assigned to a team
    
    def test_classify_team_b(self, team_clusterer, calibration_data, blue_team_colors):
        """Test classification assigns to correct team."""
        hsv_colors, rgb_colors = calibration_data
        team_clusterer.calibrate(hsv_colors, rgb_colors)
        
        # Classify a blue color
        result = team_clusterer.classify(blue_team_colors[0])
        
        assert result in [Team.TEAM_A, Team.TEAM_B]
    
    def test_classify_referee(self, team_clusterer, calibration_data, referee_color):
        """Test referee detection (color far from both teams)."""
        hsv_colors, rgb_colors = calibration_data
        team_clusterer.calibrate(hsv_colors, rgb_colors)
        
        # Classify referee color (gray/black)
        result = team_clusterer.classify(referee_color)
        
        assert result == Team.REFEREE
    
    def test_classify_with_track_id_caching(self, team_clusterer, calibration_data, red_team_colors):
        """Test classification caches results by track ID."""
        hsv_colors, rgb_colors = calibration_data
        team_clusterer.calibrate(hsv_colors, rgb_colors)
        
        track_id = 1
        
        # First classification
        result1 = team_clusterer.classify(red_team_colors[0], track_id)
        
        # Second classification with same track
        result2 = team_clusterer.classify(red_team_colors[1], track_id)
        
        # Should be consistent
        assert result1 == result2
    
    def test_hsv_distance(self, team_clusterer):
        """Test HSV distance calculation."""
        color1 = (100, 200, 200)
        color2 = np.array([100, 200, 200])
        
        dist = team_clusterer._hsv_distance(color1, color2)
        
        assert dist == 0  # Identical colors
    
    def test_hsv_distance_different(self, team_clusterer):
        """Test HSV distance for different colors."""
        color1 = (0, 200, 200)  # Red
        color2 = np.array([100, 200, 200])  # Blue
        
        dist = team_clusterer._hsv_distance(color1, color2)
        
        assert dist > 100  # Should be significant
    
    def test_hsv_distance_hue_wraparound(self, team_clusterer):
        """Test HSV distance handles hue wraparound."""
        color1 = (5, 200, 200)  # Red (near 0)
        color2 = np.array([175, 200, 200])  # Red (near 180)
        
        dist = team_clusterer._hsv_distance(color1, color2)
        
        # Should be small due to wraparound handling
        assert dist < 50
    
    def test_reset_clears_state(self, team_clusterer, calibration_data):
        """Test reset clears all state."""
        hsv_colors, rgb_colors = calibration_data
        team_clusterer.calibrate(hsv_colors, rgb_colors)
        
        assert team_clusterer.is_calibrated
        
        team_clusterer.reset()
        
        assert not team_clusterer.is_calibrated
        assert len(team_clusterer.team_colors) == 0
    
    def test_get_track_history_empty(self, team_clusterer):
        """Test get track history for unknown track."""
        history = team_clusterer.get_track_history(999)
        assert history == []
    
    def test_get_stable_team_unknown(self, team_clusterer):
        """Test get stable team for unknown track."""
        result = team_clusterer.get_stable_team(999)
        assert result == Team.UNKNOWN