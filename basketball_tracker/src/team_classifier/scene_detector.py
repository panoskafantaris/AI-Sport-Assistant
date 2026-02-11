"""
Scene change detection for handling video highlights with multiple clips.
"""
import cv2
import numpy as np
from typing import Optional

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import config


class SceneDetector:
    """Detects scene changes in video using histogram comparison."""
    
    def __init__(
        self,
        threshold: float = config.SCENE_CHANGE_THRESHOLD,
        cooldown_frames: int = config.SCENE_CHANGE_COOLDOWN
    ):
        """
        Initialize scene detector.
        
        Args:
            threshold: Histogram difference threshold (0-1)
            cooldown_frames: Minimum frames between scene changes
        """
        self.threshold = threshold
        self.cooldown_frames = cooldown_frames
        
        self._previous_histogram: Optional[np.ndarray] = None
        self._frames_since_change: int = 0
    
    def _compute_histogram(self, frame: np.ndarray) -> np.ndarray:
        """
        Compute normalized color histogram for frame.
        
        Args:
            frame: BGR image
        
        Returns:
            Normalized histogram
        """
        # Convert to HSV for better scene comparison
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Compute histogram on H and S channels
        hist = cv2.calcHist(
            [hsv], 
            [0, 1],  # H and S channels
            None, 
            [50, 60],  # bins
            [0, 180, 0, 256]  # ranges
        )
        
        # Normalize
        cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
        
        return hist.flatten()
    
    def _histogram_difference(
        self, 
        hist1: np.ndarray, 
        hist2: np.ndarray
    ) -> float:
        """
        Compute difference between two histograms.
        
        Args:
            hist1: First histogram
            hist2: Second histogram
        
        Returns:
            Difference score (0 = identical, 1 = completely different)
        """
        # Use correlation method (1 = identical, -1 = opposite)
        correlation = cv2.compareHist(
            hist1.reshape(-1, 1).astype(np.float32),
            hist2.reshape(-1, 1).astype(np.float32),
            cv2.HISTCMP_CORREL
        )
        
        # Convert to difference (0 = identical, 1 = different)
        return 1.0 - max(0.0, correlation)
    
    def is_scene_change(self, frame: np.ndarray) -> bool:
        """
        Check if current frame represents a scene change.
        
        Args:
            frame: BGR image
        
        Returns:
            True if scene change detected
        """
        current_hist = self._compute_histogram(frame)
        
        # First frame
        if self._previous_histogram is None:
            self._previous_histogram = current_hist
            self._frames_since_change = 0
            return True  # First frame is a "scene change"
        
        # Check cooldown
        self._frames_since_change += 1
        if self._frames_since_change < self.cooldown_frames:
            return False
        
        # Compare histograms
        diff = self._histogram_difference(self._previous_histogram, current_hist)
        
        # Update previous histogram
        self._previous_histogram = current_hist
        
        # Check threshold
        if diff > self.threshold:
            self._frames_since_change = 0
            return True
        
        return False
    
    def reset(self) -> None:
        """Reset detector state for new video."""
        self._previous_histogram = None
        self._frames_since_change = 0
    
    def get_frames_since_change(self) -> int:
        """Get number of frames since last scene change."""
        return self._frames_since_change