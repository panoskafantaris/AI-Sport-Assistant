"""
Team clustering using K-means on jersey colors.
"""
import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.cluster import KMeans
from collections import defaultdict

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import config
from src.models import Team


class TeamClusterer:
    """Clusters players into teams based on jersey colors."""
    
    def __init__(
        self,
        num_teams: int = config.NUM_TEAMS,
        referee_threshold: float = config.REFEREE_DISTANCE_THRESHOLD,
        min_samples: int = config.MIN_SAMPLES_FOR_CLUSTERING
    ):
        """
        Initialize team clusterer.
        
        Args:
            num_teams: Number of teams to detect
            referee_threshold: HSV distance threshold for referee detection
            min_samples: Minimum samples needed to perform clustering
        """
        self.num_teams = num_teams
        self.referee_threshold = referee_threshold
        self.min_samples = min_samples
        
        # Team centroids in HSV space
        self._centroids: Optional[np.ndarray] = None
        self._team_colors_rgb: Dict[Team, Tuple[int, int, int]] = {}
        self._is_calibrated: bool = False
        
        # Track color history per player ID for stability
        self._player_color_history: Dict[int, List[Tuple[int, int, int]]] = defaultdict(list)
        self._player_team_cache: Dict[int, Team] = {}
    
    @property
    def is_calibrated(self) -> bool:
        """Check if clusterer has been calibrated."""
        return self._is_calibrated
    
    @property
    def team_colors(self) -> Dict[Team, Tuple[int, int, int]]:
        """Get RGB colors for each team."""
        return self._team_colors_rgb.copy()
    
    def calibrate(
        self, 
        colors_hsv: List[Tuple[int, int, int]],
        colors_rgb: List[Tuple[int, int, int]]
    ) -> bool:
        """
        Calibrate team centroids using collected jersey colors.
        
        Args:
            colors_hsv: List of HSV colors from players
            colors_rgb: Corresponding RGB colors
        
        Returns:
            True if calibration successful
        """
        if len(colors_hsv) < self.min_samples:
            return False
        
        # Convert to numpy array
        colors_array = np.array(colors_hsv)
        rgb_array = np.array(colors_rgb)
        
        # K-means clustering
        kmeans = KMeans(n_clusters=self.num_teams, random_state=42, n_init=10)
        labels = kmeans.fit_predict(colors_array)
        
        self._centroids = kmeans.cluster_centers_
        
        # Compute average RGB color for each cluster
        for i, team in enumerate([Team.TEAM_A, Team.TEAM_B][:self.num_teams]):
            cluster_mask = labels == i
            if np.any(cluster_mask):
                avg_rgb = rgb_array[cluster_mask].mean(axis=0)
                self._team_colors_rgb[team] = tuple(int(c) for c in avg_rgb)
        
        self._is_calibrated = True
        return True
    
    def _hsv_distance(
        self, 
        color1: Tuple[int, int, int], 
        color2: np.ndarray
    ) -> float:
        """
        Compute weighted distance between HSV colors.
        
        Hue is circular (0-180), so we handle wraparound.
        """
        h1, s1, v1 = color1
        h2, s2, v2 = color2
        
        # Hue difference (circular)
        h_diff = min(abs(h1 - h2), 180 - abs(h1 - h2))
        
        # Weighted distance (hue matters most for jersey color)
        distance = np.sqrt(
            (h_diff * 2.0) ** 2 +  # Hue weight
            (s1 - s2) ** 2 +        # Saturation
            ((v1 - v2) * 0.5) ** 2  # Value weight (less important)
        )
        
        return distance
    
    def classify(
        self, 
        color_hsv: Tuple[int, int, int],
        track_id: Optional[int] = None
    ) -> Team:
        """
        Classify a player into a team based on jersey color.
        
        Args:
            color_hsv: Player's jersey color in HSV
            track_id: Optional track ID for caching
        
        Returns:
            Team classification
        """
        if not self._is_calibrated or self._centroids is None:
            return Team.UNKNOWN
        
        # Check cache for this track
        if track_id is not None and track_id in self._player_team_cache:
            # Update history and check if classification is stable
            self._player_color_history[track_id].append(color_hsv)
            if len(self._player_color_history[track_id]) > 10:
                self._player_color_history[track_id].pop(0)
            
            # Use cached team if history is short
            if len(self._player_color_history[track_id]) < 5:
                return self._player_team_cache[track_id]
        
        # Compute distance to each centroid
        distances = []
        for centroid in self._centroids:
            dist = self._hsv_distance(color_hsv, centroid)
            distances.append(dist)
        
        min_distance = min(distances)
        
        # Check if too far from all centroids (likely referee)
        if min_distance > self.referee_threshold:
            team = Team.REFEREE
        else:
            # Assign to nearest team
            nearest_idx = np.argmin(distances)
            team = [Team.TEAM_A, Team.TEAM_B][nearest_idx]
        
        # Cache result
        if track_id is not None:
            self._player_team_cache[track_id] = team
        
        return team
    
    def get_stable_team(self, track_id: int) -> Team:
        """
        Get stable team assignment for a track based on history.
        
        Args:
            track_id: Track ID
        
        Returns:
            Most common team assignment
        """
        if track_id not in self._player_color_history:
            return self._player_team_cache.get(track_id, Team.UNKNOWN)
        
        history = self._player_color_history[track_id]
        if len(history) < 3:
            return self._player_team_cache.get(track_id, Team.UNKNOWN)
        
        # Reclassify based on average color
        avg_color = tuple(int(c) for c in np.mean(history, axis=0))
        return self.classify(avg_color)
    
    def reset(self) -> None:
        """Reset clusterer state for new scene."""
        self._centroids = None
        self._team_colors_rgb.clear()
        self._is_calibrated = False
        self._player_color_history.clear()
        self._player_team_cache.clear()