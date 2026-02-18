"""
Store and compare reference colors for team classification.

Collects multiple samples per team to compute average dominant color.
Every player is assigned to the closest reference (Team A, B, or Referee).
"""
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import config
from src.models import Team


@dataclass
class ColorSample:
    """A single color sample from a player."""
    hsv_color: Tuple[int, int, int]
    rgb_color: Tuple[int, int, int]


@dataclass
class TeamColorReference:
    """
    Color reference for a team, computed from multiple samples.
    """
    team: Team
    samples: List[ColorSample] = field(default_factory=list)
    
    # Computed dominant color (average of samples)
    _dominant_hsv: Optional[Tuple[int, int, int]] = None
    _dominant_rgb: Optional[Tuple[int, int, int]] = None
    
    def add_sample(self, hsv: Tuple[int, int, int], rgb: Tuple[int, int, int]) -> None:
        """Add a color sample."""
        self.samples.append(ColorSample(hsv_color=hsv, rgb_color=rgb))
        self._compute_dominant()
    
    def _compute_dominant(self) -> None:
        """Compute dominant color by averaging samples."""
        if not self.samples:
            self._dominant_hsv = None
            self._dominant_rgb = None
            return
        
        # Average HSV (handle hue wraparound)
        hsv_array = np.array([s.hsv_color for s in self.samples])
        rgb_array = np.array([s.rgb_color for s in self.samples])
        
        # For hue, use circular mean
        hues = hsv_array[:, 0]
        # Convert to radians (hue is 0-180 in OpenCV)
        hue_rad = hues * np.pi / 90  # 180 degrees = pi radians
        mean_sin = np.mean(np.sin(hue_rad))
        mean_cos = np.mean(np.cos(hue_rad))
        mean_hue = np.arctan2(mean_sin, mean_cos) * 90 / np.pi
        if mean_hue < 0:
            mean_hue += 180
        
        # Simple average for S and V
        mean_s = np.mean(hsv_array[:, 1])
        mean_v = np.mean(hsv_array[:, 2])
        
        self._dominant_hsv = (int(mean_hue), int(mean_s), int(mean_v))
        self._dominant_rgb = tuple(int(c) for c in np.mean(rgb_array, axis=0))
    
    @property
    def dominant_hsv(self) -> Optional[Tuple[int, int, int]]:
        """Get dominant HSV color."""
        return self._dominant_hsv
    
    @property
    def dominant_rgb(self) -> Optional[Tuple[int, int, int]]:
        """Get dominant RGB color."""
        return self._dominant_rgb
    
    @property
    def sample_count(self) -> int:
        """Number of samples collected."""
        return len(self.samples)
    
    def describe_color(self) -> str:
        """Human-readable color description."""
        if self._dominant_hsv is None:
            return "unknown"
        
        h, s, v = self._dominant_hsv
        
        if s < 30:
            if v < 80:
                return "black/dark"
            elif v > 200:
                return "white"
            else:
                return "gray"
        
        if h < 10 or h > 170:
            return "red"
        elif h < 25:
            return "orange"
        elif h < 35:
            return "yellow"
        elif h < 85:
            return "green"
        elif h < 130:
            return "blue"
        elif h < 150:
            return "purple"
        else:
            return "pink"
    
    def to_dict(self) -> dict:
        return {
            "team": self.team.name,
            "sample_count": self.sample_count,
            "samples": [
                {"hsv": list(s.hsv_color), "rgb": list(s.rgb_color)}
                for s in self.samples
            ],
            "dominant_hsv": list(self._dominant_hsv) if self._dominant_hsv else None,
            "dominant_rgb": list(self._dominant_rgb) if self._dominant_rgb else None,
            "color_description": self.describe_color()
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "TeamColorReference":
        ref = cls(team=Team[data["team"]])
        for sample_data in data.get("samples", []):
            ref.samples.append(ColorSample(
                hsv_color=tuple(sample_data["hsv"]),
                rgb_color=tuple(sample_data["rgb"])
            ))
        ref._compute_dominant()
        return ref


class ColorReferenceStore:
    """
    Stores color references for teams.
    
    Collects 3 samples per team, averages to find dominant color.
    Classifies players to closest team - no UNKNOWN.
    """
    
    SAMPLES_REQUIRED = 3  # Samples needed per team
    
    def __init__(self):
        """Initialize color reference store."""
        self._references: Dict[Team, TeamColorReference] = {
            Team.TEAM_A: TeamColorReference(team=Team.TEAM_A),
            Team.TEAM_B: TeamColorReference(team=Team.TEAM_B),
            Team.REFEREE: TeamColorReference(team=Team.REFEREE),
        }
    
    @property
    def is_complete(self) -> bool:
        """Check if all teams have enough samples."""
        for team in [Team.TEAM_A, Team.TEAM_B, Team.REFEREE]:
            if self._references[team].sample_count < self.SAMPLES_REQUIRED:
                return False
        return True
    
    def get_sample_count(self, team: Team) -> int:
        """Get number of samples for a team."""
        return self._references[team].sample_count
    
    def needs_samples(self, team: Team) -> int:
        """Get how many more samples needed for a team."""
        return max(0, self.SAMPLES_REQUIRED - self.get_sample_count(team))
    
    def add_sample(
        self,
        team: Team,
        hsv_color: Tuple[int, int, int],
        rgb_color: Tuple[int, int, int]
    ) -> bool:
        """
        Add a color sample for a team.
        
        Returns:
            True if sample was added (not at limit)
        """
        if self.get_sample_count(team) >= self.SAMPLES_REQUIRED:
            return False
        
        self._references[team].add_sample(hsv_color, rgb_color)
        return True
    
    def get_team_color_rgb(self, team: Team) -> Optional[Tuple[int, int, int]]:
        """Get dominant RGB color for a team."""
        return self._references[team].dominant_rgb
    
    def get_team_color_hsv(self, team: Team) -> Optional[Tuple[int, int, int]]:
        """Get dominant HSV color for a team."""
        return self._references[team].dominant_hsv
    
    def _hsv_distance(
        self,
        color1: Tuple[int, int, int],
        color2: Tuple[int, int, int]
    ) -> float:
        """Compute weighted distance between HSV colors."""
        h1, s1, v1 = color1
        h2, s2, v2 = color2
        
        # Hue difference (circular)
        h_diff = min(abs(h1 - h2), 180 - abs(h1 - h2))
        
        # Weighted distance
        distance = np.sqrt(
            (h_diff * 2.0) ** 2 +
            (s1 - s2) ** 2 +
            ((v1 - v2) * 0.5) ** 2
        )
        
        return distance
    
    def classify(
        self,
        hsv_color: Tuple[int, int, int]
    ) -> Tuple[Team, float]:
        """
        Classify a color to the closest team.
        
        Always returns Team A, B, or Referee - no UNKNOWN.
        """
        best_team = Team.TEAM_A
        best_distance = float('inf')
        
        for team in [Team.TEAM_A, Team.TEAM_B, Team.REFEREE]:
            ref_hsv = self._references[team].dominant_hsv
            if ref_hsv is None:
                continue
            
            dist = self._hsv_distance(hsv_color, ref_hsv)
            if dist < best_distance:
                best_distance = dist
                best_team = team
        
        return best_team, best_distance
    
    def save(self, filepath: str = None) -> Path:
        """Save references to JSON."""
        if filepath is None:
            filepath = config.RESULTS_DIR / config.REFERENCE_COLORS_FILE
        else:
            filepath = Path(filepath)
        
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "description": "Team color references (3 samples each)",
            "samples_required": self.SAMPLES_REQUIRED,
            "references": {
                team.name: self._references[team].to_dict()
                for team in [Team.TEAM_A, Team.TEAM_B, Team.REFEREE]
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        return filepath
    
    def load(self, filepath: str = None) -> bool:
        """Load references from JSON."""
        if filepath is None:
            filepath = config.RESULTS_DIR / config.REFERENCE_COLORS_FILE
        else:
            filepath = Path(filepath)
        
        if not filepath.exists():
            return False
        
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            refs_data = data.get("references", {})
            for team_name, ref_data in refs_data.items():
                team = Team[team_name]
                self._references[team] = TeamColorReference.from_dict(ref_data)
            
            return True
        except Exception as e:
            print(f"Error loading color references: {e}")
            return False
    
    def clear(self) -> None:
        """Clear all samples."""
        for team in self._references:
            self._references[team] = TeamColorReference(team=team)
    
    def __str__(self) -> str:
        lines = ["Color References (3 samples each):"]
        for team in [Team.TEAM_A, Team.TEAM_B, Team.REFEREE]:
            ref = self._references[team]
            count = ref.sample_count
            if ref.dominant_rgb:
                color_desc = ref.describe_color()
                lines.append(f"  {team.name}: {color_desc} RGB{ref.dominant_rgb} ({count}/{self.SAMPLES_REQUIRED} samples)")
            else:
                lines.append(f"  {team.name}: ({count}/{self.SAMPLES_REQUIRED} samples)")
        return "\n".join(lines)