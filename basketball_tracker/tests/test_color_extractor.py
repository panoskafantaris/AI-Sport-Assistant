"""
Tests for color extractor module.
"""
import cv2
import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.team_classifier.color_extractor import ColorExtractor


@pytest.fixture
def color_extractor():
    """Create a color extractor instance."""
    return ColorExtractor()


@pytest.fixture
def red_jersey_frame():
    """Create a frame with a red jersey region."""
    frame = np.zeros((400, 300, 3), dtype=np.uint8)
    frame[:] = (34, 139, 34)  # Green background (court)
    
    # Draw a "player" with red jersey
    # Head (skin color)
    cv2.rectangle(frame, (100, 50), (200, 100), (180, 200, 230), -1)
    # Jersey (red)
    cv2.rectangle(frame, (100, 100), (200, 220), (0, 0, 255), -1)
    # Shorts/legs (dark)
    cv2.rectangle(frame, (100, 220), (200, 350), (50, 50, 50), -1)
    
    return frame


@pytest.fixture
def blue_jersey_frame():
    """Create a frame with a blue jersey region."""
    frame = np.zeros((400, 300, 3), dtype=np.uint8)
    frame[:] = (34, 139, 34)  # Green background
    
    # Draw a "player" with blue jersey
    cv2.rectangle(frame, (100, 50), (200, 100), (180, 200, 230), -1)  # Head
    cv2.rectangle(frame, (100, 100), (200, 220), (255, 0, 0), -1)  # Jersey (blue)
    cv2.rectangle(frame, (100, 220), (200, 350), (50, 50, 50), -1)  # Legs
    
    return frame


class TestColorExtractor:
    """Tests for ColorExtractor class."""
    
    def test_initialization(self, color_extractor):
        """Test color extractor initializes correctly."""
        assert color_extractor.crop_top_ratio == 0.15
        assert color_extractor.crop_bottom_ratio == 0.55
        assert color_extractor.n_clusters == 3
    
    def test_extract_jersey_region_valid(self, color_extractor, red_jersey_frame):
        """Test jersey region extraction with valid bbox."""
        bbox = (100, 50, 200, 350)  # x1, y1, x2, y2
        region = color_extractor.extract_jersey_region(red_jersey_frame, bbox)
        
        assert region is not None
        assert region.shape[0] > 0
        assert region.shape[1] > 0
    
    def test_extract_jersey_region_too_small(self, color_extractor):
        """Test jersey region extraction with too small bbox."""
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        bbox = (0, 0, 10, 15)  # Too small
        
        region = color_extractor.extract_jersey_region(frame, bbox)
        assert region is None
    
    def test_get_dominant_color_red(self, color_extractor, red_jersey_frame):
        """Test dominant color extraction for red jersey."""
        bbox = (100, 50, 200, 350)
        region = color_extractor.extract_jersey_region(red_jersey_frame, bbox)
        
        hsv_color = color_extractor.get_dominant_color_hsv(region)
        
        assert hsv_color is not None
        # Red in HSV has hue near 0 or 180
        h, s, v = hsv_color
        assert (h < 10 or h > 170) or s > 100  # Red hue range
    
    def test_get_dominant_color_blue(self, color_extractor, blue_jersey_frame):
        """Test dominant color extraction for blue jersey."""
        bbox = (100, 50, 200, 350)
        region = color_extractor.extract_jersey_region(blue_jersey_frame, bbox)
        
        hsv_color = color_extractor.get_dominant_color_hsv(region)
        
        assert hsv_color is not None
        # Blue in HSV has hue around 100-130
        h, s, v = hsv_color
        assert 90 < h < 140 or s > 100  # Blue hue range
    
    def test_hsv_to_rgb_conversion(self, color_extractor):
        """Test HSV to RGB conversion."""
        # Pure red in HSV
        hsv = (0, 255, 255)
        rgb = color_extractor.hsv_to_rgb(hsv)
        
        assert rgb is not None
        assert len(rgb) == 3
        # Should be close to pure red (255, 0, 0)
        assert rgb[0] > 200  # R
    
    def test_extract_color_full_pipeline(self, color_extractor, red_jersey_frame):
        """Test full color extraction pipeline."""
        bbox = (100, 50, 200, 350)
        result = color_extractor.extract_color(red_jersey_frame, bbox)
        
        assert result is not None
        hsv_color, rgb_color = result
        
        assert len(hsv_color) == 3
        assert len(rgb_color) == 3
    
    def test_extract_color_empty_region(self, color_extractor):
        """Test color extraction with invalid region."""
        frame = np.zeros((50, 50, 3), dtype=np.uint8)
        bbox = (0, 0, 5, 5)  # Too small
        
        result = color_extractor.extract_color(frame, bbox)
        assert result is None
    
    def test_get_dominant_color_none_input(self, color_extractor):
        """Test dominant color with None input."""
        result = color_extractor.get_dominant_color_hsv(None)
        assert result is None
    
    def test_get_dominant_color_empty_array(self, color_extractor):
        """Test dominant color with empty array."""
        empty = np.array([]).reshape(0, 0, 3).astype(np.uint8)
        result = color_extractor.get_dominant_color_hsv(empty)
        assert result is None