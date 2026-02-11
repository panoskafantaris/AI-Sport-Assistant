"""
Visualization utilities for drawing annotations on frames.
"""
import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional

from .models import Detection, TrackedObject, Team
import config


# Color palette for different track IDs (fallback)
COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255),
    (0, 255, 255), (128, 0, 255), (255, 128, 0), (0, 128, 255), (128, 255, 0),
]

# Default team colors (BGR format for OpenCV)
DEFAULT_TEAM_COLORS = {
    Team.TEAM_A: (255, 100, 100),    # Light blue
    Team.TEAM_B: (100, 100, 255),    # Light red
    Team.REFEREE: (128, 128, 128),   # Gray
    Team.UNKNOWN: (200, 200, 200),   # Light gray
}


class Visualizer:
    """Draws bounding boxes, IDs, and trajectories on frames."""
    
    def __init__(
        self,
        box_thickness: int = config.BOX_THICKNESS,
        font_scale: float = config.FONT_SCALE,
        font_thickness: int = config.FONT_THICKNESS,
        draw_trajectory: bool = True,
        trajectory_length: int = 30,
        use_team_colors: bool = True
    ):
        self.box_thickness = box_thickness
        self.font_scale = font_scale
        self.font_thickness = font_thickness
        self.draw_trajectory = draw_trajectory
        self.trajectory_length = trajectory_length
        self.use_team_colors = use_team_colors
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Team colors (can be updated with actual jersey colors)
        self._team_colors: Dict[Team, Tuple[int, int, int]] = DEFAULT_TEAM_COLORS.copy()
    
    def set_team_colors(self, colors: Dict[Team, Tuple[int, int, int]]) -> None:
        """
        Set custom team colors (e.g., from jersey color extraction).
        
        Args:
            colors: Dictionary mapping Team to RGB color tuple
        """
        for team, rgb_color in colors.items():
            # Convert RGB to BGR for OpenCV
            bgr_color = (rgb_color[2], rgb_color[1], rgb_color[0])
            self._team_colors[team] = bgr_color
    
    def _get_color(self, track_id: int) -> Tuple[int, int, int]:
        """Get consistent color for a track ID (fallback)."""
        return COLORS[track_id % len(COLORS)]
    
    def _get_team_color(self, team: Team) -> Tuple[int, int, int]:
        """Get color for a team."""
        return self._team_colors.get(team, DEFAULT_TEAM_COLORS[Team.UNKNOWN])
    
    def draw_detections(
        self, frame: np.ndarray, detections: List[Detection],
        color: Tuple[int, int, int] = (0, 255, 0)
    ) -> np.ndarray:
        """Draw detection boxes on frame."""
        annotated = frame.copy()
        for det in detections:
            x1, y1, x2, y2 = det.bbox.to_int_tuple()
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, self.box_thickness)
            self._draw_label(annotated, f"{det.class_name}: {det.confidence:.2f}", (x1, y1), color)
        return annotated
    
    def draw_tracks(self, frame: np.ndarray, tracked_objects: List[TrackedObject]) -> np.ndarray:
        """Draw tracked objects with IDs, team colors, and trajectories."""
        annotated = frame.copy()
        for obj in tracked_objects:
            # Use team color if available and enabled, otherwise fall back to track ID color
            if self.use_team_colors and obj.team != Team.UNKNOWN:
                color = self._get_team_color(obj.team)
                label = f"ID:{obj.track_id} {obj.team.name}"
            else:
                color = self._get_color(obj.track_id)
                label = f"ID:{obj.track_id}"
            
            x1, y1, x2, y2 = obj.bbox.to_int_tuple()
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, self.box_thickness)
            self._draw_label(annotated, label, (x1, y1), color)
            
            if self.draw_trajectory and obj.history:
                self._draw_trajectory(annotated, obj.history, color)
        return annotated
    
    def _draw_label(self, frame: np.ndarray, label: str, 
                    position: Tuple[int, int], color: Tuple[int, int, int]) -> None:
        """Draw text label with background."""
        x, y = position
        (text_w, text_h), baseline = cv2.getTextSize(label, self.font, self.font_scale, self.font_thickness)
        cv2.rectangle(frame, (x, y - text_h - baseline - 5), (x + text_w, y), color, -1)
        cv2.putText(frame, label, (x, y - baseline - 2), self.font, self.font_scale, (255, 255, 255), self.font_thickness)
    
    def _draw_trajectory(self, frame: np.ndarray, history: List[Tuple[float, float]],
                         color: Tuple[int, int, int]) -> None:
        """Draw movement trajectory as connected line."""
        points = history[-self.trajectory_length:]
        if len(points) < 2:
            return
        for i in range(1, len(points)):
            pt1 = (int(points[i - 1][0]), int(points[i - 1][1]))
            pt2 = (int(points[i][0]), int(points[i][1]))
            thickness = max(1, int(self.box_thickness * (i / len(points))))
            cv2.line(frame, pt1, pt2, color, thickness)
    
    def draw_frame_info(self, frame: np.ndarray, frame_number: int,
                        num_tracks: int, fps: Optional[float] = None,
                        team_counts: Optional[Dict[Team, int]] = None) -> np.ndarray:
        """Draw frame information overlay."""
        annotated = frame.copy()
        info_lines = [f"Frame: {frame_number}", f"Tracks: {num_tracks}"]
        
        if fps is not None:
            info_lines.append(f"FPS: {fps:.1f}")
        
        # Add team counts if available
        if team_counts:
            team_a = team_counts.get(Team.TEAM_A, 0)
            team_b = team_counts.get(Team.TEAM_B, 0)
            refs = team_counts.get(Team.REFEREE, 0)
            if team_a > 0 or team_b > 0:
                info_lines.append(f"Teams: A={team_a} B={team_b}")
            if refs > 0:
                info_lines.append(f"Refs: {refs}")
        
        y_offset = 30
        for line in info_lines:
            cv2.putText(annotated, line, (10, y_offset), self.font, self.font_scale, (0, 255, 0), self.font_thickness)
            y_offset += 25
        return annotated