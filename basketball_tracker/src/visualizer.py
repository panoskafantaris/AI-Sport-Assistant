"""
Visualization utilities for drawing annotations on frames.
"""
import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional

from .models import Detection, TrackedObject, Team
import config


# Default team colors (BGR format for OpenCV)
DEFAULT_TEAM_COLORS = {
    Team.TEAM_A: (255, 100, 100),    # Light blue
    Team.TEAM_B: (100, 100, 255),    # Light red
    Team.REFEREE: (128, 128, 128),   # Gray
    Team.UNKNOWN: (200, 200, 200),   # Light gray
}


class Visualizer:
    """Draws bounding boxes with team colors on frames."""
    
    def __init__(
        self,
        box_thickness: int = config.BOX_THICKNESS,
        font_scale: float = config.FONT_SCALE,
        font_thickness: int = config.FONT_THICKNESS,
        draw_trajectory: bool = True,
        trajectory_length: int = 30
    ):
        self.box_thickness = box_thickness
        self.font_scale = font_scale
        self.font_thickness = font_thickness
        self.draw_trajectory = draw_trajectory
        self.trajectory_length = trajectory_length
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Team colors (BGR)
        self._team_colors: Dict[Team, Tuple[int, int, int]] = DEFAULT_TEAM_COLORS.copy()
    
    def set_team_colors(self, colors: Dict[Team, Tuple[int, int, int]]) -> None:
        """Set team colors from calibration (RGB → BGR)."""
        for team, rgb in colors.items():
            self._team_colors[team] = (rgb[2], rgb[1], rgb[0])
    
    def _get_team_color(self, team: Team) -> Tuple[int, int, int]:
        """Get BGR color for team."""
        return self._team_colors.get(team, DEFAULT_TEAM_COLORS[Team.UNKNOWN])
    
    def draw_detections(
        self, 
        frame: np.ndarray, 
        detections: List[Detection],
        color: Tuple[int, int, int] = (0, 255, 0)
    ) -> np.ndarray:
        """Draw detection boxes."""
        annotated = frame.copy()
        for det in detections:
            x1, y1, x2, y2 = det.bbox.to_int_tuple()
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, self.box_thickness)
        return annotated
    
    def draw_tracks(
        self, 
        frame: np.ndarray, 
        tracked_objects: List[TrackedObject]
    ) -> np.ndarray:
        """Draw tracked objects with team-colored boxes."""
        annotated = frame.copy()
        
        for obj in tracked_objects:
            color = self._get_team_color(obj.team)
            x1, y1, x2, y2 = obj.bbox.to_int_tuple()
            
            # Box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, self.box_thickness)
            
            # Label
            label = obj.team.name
            self._draw_label(annotated, label, (x1, y1), color)
            
            # Trajectory
            if self.draw_trajectory and obj.history:
                self._draw_trajectory(annotated, obj.history, color)
        
        return annotated
    
    def _draw_label(
        self, 
        frame: np.ndarray, 
        label: str,
        position: Tuple[int, int], 
        color: Tuple[int, int, int]
    ) -> None:
        """Draw label with colored background."""
        x, y = position
        (tw, th), baseline = cv2.getTextSize(
            label, self.font, self.font_scale, self.font_thickness
        )
        
        cv2.rectangle(frame, (x, y - th - baseline - 5), 
                     (x + tw + 4, y), color, -1)
        cv2.putText(frame, label, (x + 2, y - baseline - 2), 
                   self.font, self.font_scale, (255, 255, 255), self.font_thickness)
    
    def _draw_trajectory(
        self, 
        frame: np.ndarray, 
        history: List[Tuple[float, float]],
        color: Tuple[int, int, int]
    ) -> None:
        """Draw movement trajectory."""
        points = history[-self.trajectory_length:]
        if len(points) < 2:
            return
        
        for i in range(1, len(points)):
            pt1 = (int(points[i - 1][0]), int(points[i - 1][1]))
            pt2 = (int(points[i][0]), int(points[i][1]))
            thick = max(1, int(self.box_thickness * (i / len(points))))
            cv2.line(frame, pt1, pt2, color, thick)
    
    def draw_frame_info(
        self, 
        frame: np.ndarray, 
        frame_number: int,
        num_tracks: int, 
        fps: Optional[float] = None,
        team_counts: Optional[Dict[Team, int]] = None
    ) -> np.ndarray:
        """Draw frame info with team colors."""
        annotated = frame.copy()
        y = 30
        
        cv2.putText(annotated, f"Frame: {frame_number}", (10, y), 
                   self.font, self.font_scale, (0, 255, 0), self.font_thickness)
        y += 25
        
        if team_counts:
            for team in [Team.TEAM_A, Team.TEAM_B, Team.REFEREE]:
                count = team_counts.get(team, 0)
                if count > 0:
                    color = self._get_team_color(team)
                    cv2.rectangle(annotated, (10, y - 15), (30, y + 3), color, -1)
                    cv2.putText(annotated, f"{team.name}: {count}", (40, y), 
                               self.font, self.font_scale, color, self.font_thickness)
                    y += 25
        
        return annotated