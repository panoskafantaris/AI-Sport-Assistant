"""
Main team classifier that orchestrates color extraction, 
scene detection, and team clustering.
"""
import numpy as np
from typing import Dict, List, Optional, Tuple

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import config
from src.models import Team, TrackedObject, BoundingBox
from .color_extractor import ColorExtractor
from .scene_detector import SceneDetector
from .team_clusterer import TeamClusterer


class TeamClassifier:
    """
    Classifies tracked players into teams based on jersey colors.
    
    Handles scene changes in highlight videos by re-calibrating
    when a new scene is detected.
    """
    
    def __init__(
        self,
        calibration_frames: int = config.CALIBRATION_FRAMES,
        enable_scene_detection: bool = True
    ):
        """
        Initialize team classifier.
        
        Args:
            calibration_frames: Frames to collect before calibrating
            enable_scene_detection: Whether to detect scene changes
        """
        self.calibration_frames = calibration_frames
        self.enable_scene_detection = enable_scene_detection
        
        # Components
        self.color_extractor = ColorExtractor()
        self.scene_detector = SceneDetector()
        self.team_clusterer = TeamClusterer()
        
        # Calibration state
        self._calibration_colors_hsv: List[Tuple[int, int, int]] = []
        self._calibration_colors_rgb: List[Tuple[int, int, int]] = []
        self._frames_collected: int = 0
        self._is_calibrating: bool = True
    
    @property
    def is_calibrated(self) -> bool:
        """Check if classifier is calibrated."""
        return self.team_clusterer.is_calibrated
    
    @property
    def team_colors(self) -> Dict[Team, Tuple[int, int, int]]:
        """Get RGB colors for each team."""
        return self.team_clusterer.team_colors
    
    def _start_calibration(self) -> None:
        """Start or restart calibration phase."""
        self._calibration_colors_hsv.clear()
        self._calibration_colors_rgb.clear()
        self._frames_collected = 0
        self._is_calibrating = True
        self.team_clusterer.reset()
    
    def _collect_calibration_sample(
        self, 
        frame: np.ndarray,
        tracked_objects: List[TrackedObject]
    ) -> None:
        """
        Collect jersey colors for calibration.
        
        Args:
            frame: BGR image
            tracked_objects: List of tracked players
        """
        for obj in tracked_objects:
            bbox = obj.bbox.to_int_tuple()
            result = self.color_extractor.extract_color(frame, bbox)
            
            if result is not None:
                hsv_color, rgb_color = result
                self._calibration_colors_hsv.append(hsv_color)
                self._calibration_colors_rgb.append(rgb_color)
        
        self._frames_collected += 1
    
    def _try_calibrate(self) -> bool:
        """
        Attempt to calibrate if enough samples collected.
        
        Returns:
            True if calibration successful
        """
        if self._frames_collected < self.calibration_frames:
            return False
        
        success = self.team_clusterer.calibrate(
            self._calibration_colors_hsv,
            self._calibration_colors_rgb
        )
        
        if success:
            self._is_calibrating = False
            print(f"Team calibration complete with {len(self._calibration_colors_hsv)} samples")
            for team, color in self.team_clusterer.team_colors.items():
                print(f"  {team.name}: RGB{color}")
        
        return success
    
    def classify_frame(
        self, 
        frame: np.ndarray,
        tracked_objects: List[TrackedObject]
    ) -> List[TrackedObject]:
        """
        Classify all tracked objects in a frame.
        
        Args:
            frame: BGR image
            tracked_objects: List of tracked players
        
        Returns:
            Updated tracked objects with team assignments
        """
        # Check for scene change
        if self.enable_scene_detection:
            if self.scene_detector.is_scene_change(frame):
                if self.is_calibrated:
                    print("Scene change detected - re-calibrating...")
                self._start_calibration()
        
        # Calibration phase
        if self._is_calibrating:
            self._collect_calibration_sample(frame, tracked_objects)
            self._try_calibrate()
            return tracked_objects  # Return without team assignments
        
        # Classification phase
        for obj in tracked_objects:
            bbox = obj.bbox.to_int_tuple()
            result = self.color_extractor.extract_color(frame, bbox)
            
            if result is not None:
                hsv_color, rgb_color = result
                obj.team = self.team_clusterer.classify(hsv_color, obj.track_id)
                obj.jersey_color_rgb = rgb_color
            else:
                # Use cached team if color extraction failed
                obj.team = self.team_clusterer.get_stable_team(obj.track_id)
        
        return tracked_objects
    
    def reset(self) -> None:
        """Reset classifier for new video."""
        self._start_calibration()
        self.scene_detector.reset()
    
    def get_team_stats(
        self, 
        tracked_objects: List[TrackedObject]
    ) -> Dict[Team, int]:
        """
        Get count of players per team in current frame.
        
        Args:
            tracked_objects: List of tracked players
        
        Returns:
            Dictionary of team -> player count
        """
        stats = {team: 0 for team in Team}
        for obj in tracked_objects:
            stats[obj.team] += 1
        return stats