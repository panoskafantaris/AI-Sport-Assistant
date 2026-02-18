"""
Main team classifier using reference colors.

Every player is classified as Team A, Team B, or Referee.
No UNKNOWN - closest match always wins.
"""
import numpy as np
from typing import Dict, List, Optional, Tuple

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import config
from src.models import Team, TrackedObject
from .color_extractor import ColorExtractor
from .color_reference import ColorReferenceStore
from .scene_detector import SceneDetector


class TeamClassifier:
    """
    Classifies players into teams using reference colors.
    
    Every player gets assigned to Team A, Team B, or Referee.
    The closest color match wins - no UNKNOWN results.
    """
    
    def __init__(
        self,
        color_store: Optional[ColorReferenceStore] = None,
        enable_scene_detection: bool = True
    ):
        """
        Initialize team classifier.
        
        Args:
            color_store: Pre-configured color references (from calibration)
            enable_scene_detection: Whether to detect scene changes
        """
        self.color_extractor = ColorExtractor()
        self.scene_detector = SceneDetector() if enable_scene_detection else None
        self.color_store = color_store or ColorReferenceStore()
        
        # Cache team assignments for stability
        self._track_team_cache: Dict[int, Team] = {}
    
    @property
    def is_calibrated(self) -> bool:
        """Check if classifier has all color references."""
        return self.color_store.is_complete
    
    @property
    def team_colors(self) -> Dict[Team, Tuple[int, int, int]]:
        """Get RGB colors for each team."""
        colors = {}
        for team in [Team.TEAM_A, Team.TEAM_B, Team.REFEREE]:
            rgb = self.color_store.get_team_color_rgb(team)
            if rgb:
                colors[team] = rgb
        return colors
    
    def set_color_store(self, color_store: ColorReferenceStore) -> None:
        """Set color references after calibration."""
        self.color_store = color_store
        self._track_team_cache.clear()
    
    def classify_frame(
        self,
        frame: np.ndarray,
        tracked_objects: List[TrackedObject]
    ) -> List[TrackedObject]:
        """
        Classify all players in a frame.
        
        Every player gets assigned to Team A, B, or Referee.
        
        Args:
            frame: BGR image
            tracked_objects: Detected players
        
        Returns:
            Players with team assignments
        """
        # Scene change detection (informational only)
        if self.scene_detector:
            if self.scene_detector.is_scene_change(frame):
                pass  # Continue using same references
        
        if not self.is_calibrated:
            return tracked_objects
        
        for obj in tracked_objects:
            bbox = obj.bbox.to_int_tuple()
            result = self.color_extractor.extract_color(frame, bbox)
            
            if result is not None:
                hsv_color, rgb_color = result
                team, distance = self.color_store.classify(hsv_color)
                
                obj.team = team
                obj.jersey_color_rgb = rgb_color
                self._track_team_cache[obj.track_id] = team
            else:
                # Use cached team if extraction failed
                obj.team = self._track_team_cache.get(obj.track_id, Team.TEAM_A)
        
        return tracked_objects
    
    def get_team_stats(
        self,
        tracked_objects: List[TrackedObject]
    ) -> Dict[Team, int]:
        """Get player count per team."""
        stats = {Team.TEAM_A: 0, Team.TEAM_B: 0, Team.REFEREE: 0}
        for obj in tracked_objects:
            if obj.team in stats:
                stats[obj.team] += 1
        return stats
    
    def save_calibration(self, filepath: str = None) -> Path:
        """Save color references."""
        return self.color_store.save(filepath)
    
    def load_calibration(self, filepath: str = None) -> bool:
        """Load color references."""
        success = self.color_store.load(filepath)
        if success:
            self._track_team_cache.clear()
        return success
    
    def reset(self) -> None:
        """Reset state but keep color references."""
        self._track_team_cache.clear()
        if self.scene_detector:
            self.scene_detector.reset()