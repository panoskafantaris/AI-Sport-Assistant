"""
Two-phase interactive calibration with separate video passes.

PHASE 1: COURT CALIBRATION (full video pass)
    - Left-click: Add COURT color sample
    - Right-click: Add OUT-OF-BOUNDS color sample
    - Video plays through entirely (or press ENTER to finish early)
    - Review results, continue clicking if not satisfied
    - Press SPACE to accept and move to Phase 2

PHASE 2: TEAM ASSIGNMENT (restart video)
    - Video restarts from beginning
    - Court boundary is shown
    - Click players → assign to teams (A/B/R)
    - Press ENTER when complete
"""
import cv2
import numpy as np
from enum import Enum
from typing import List, Optional, Tuple

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import config
from src.models import Team, TrackedObject
from .color_reference import ColorReferenceStore
from .color_extractor import ColorExtractor
from .court_detector import CourtDetector


class CalibrationPhase(Enum):
    COURT_CALIBRATION = 1
    COURT_REVIEW = 2
    TEAM_ASSIGNMENT = 3
    COMPLETE = 4


class ClickType(Enum):
    COURT = 1
    OUT_OF_BOUNDS = 2


class InteractiveCalibrator:
    """
    Two-phase calibration with separate video passes.
    
    Phase 1: Court calibration (court colors + OOB colors)
    Phase 2: Team assignment (restart video, click players)
    """
    
    SAMPLES_PER_TEAM = 3
    
    def __init__(
        self,
        window_name: str = config.INTERACTIVE_WINDOW_NAME,
        playback_fps: int = 30
    ):
        self.window_name = window_name
        self.frame_delay = max(1, int(1000 / playback_fps))
        
        # Components
        self.court_detector = CourtDetector()
        self.color_store = ColorReferenceStore()
        self.color_extractor = ColorExtractor()
        
        # Current phase
        self._phase = CalibrationPhase.COURT_CALIBRATION
        
        # Click handling
        self._left_click: Optional[Tuple[int, int]] = None
        self._right_click: Optional[Tuple[int, int]] = None
        
        # State
        self._frame: Optional[np.ndarray] = None
        self._objects: List[TrackedObject] = []
        self._selected: Optional[TrackedObject] = None
        self._paused: bool = False
        self._window_ready: bool = False
        
        # Video control
        self._video_ended: bool = False
        self._request_restart: bool = False
        self._request_phase2: bool = False
    
    def _on_mouse(self, event, x, y, flags, param):
        """Handle mouse clicks."""
        if event == cv2.EVENT_LBUTTONDOWN:
            self._left_click = (x, y)
        elif event == cv2.EVENT_RBUTTONDOWN:
            self._right_click = (x, y)
    
    def _handle_court_click(self, x: int, y: int, click_type: ClickType) -> None:
        """Handle click during court calibration phase."""
        if click_type == ClickType.COURT:
            rgb = self.court_detector.add_court_sample(self._frame, x, y)
            print(f"COURT sample {self.court_detector.court_sample_count}: RGB{rgb}")
        else:
            rgb = self.court_detector.add_oob_sample(self._frame, x, y)
            print(f"OUT-OF-BOUNDS sample {self.court_detector.oob_sample_count}: RGB{rgb}")
    
    def _find_object_at(self, x: int, y: int) -> Optional[TrackedObject]:
        """Find tracked object at click position."""
        for obj in self._objects:
            if obj.bbox.contains_point(x, y):
                return obj
        return None
    
    def _handle_team_click(self, x: int, y: int) -> None:
        """Handle click during team assignment phase."""
        clicked_obj = self._find_object_at(x, y)
        if clicked_obj:
            bbox = clicked_obj.bbox.to_int_tuple()
            if self.court_detector.is_on_court(self._frame, bbox):
                self._selected = clicked_obj
                self._paused = True
            else:
                print("Player outside court boundary")
    
    def _assign_team(self, team: Team) -> bool:
        """Assign selected player to team."""
        if not self._selected or self._frame is None:
            return False
        
        if self.color_store.needs_samples(team) <= 0:
            return False
        
        bbox = self._selected.bbox.to_int_tuple()
        result = self.color_extractor.extract_color(self._frame, bbox)
        
        if result is None:
            print("Could not extract color")
            return False
        
        hsv, rgb = result
        self.color_store.add_sample(team, hsv, rgb)
        
        remaining = self.color_store.needs_samples(team)
        print(f"Added {team.name}: RGB{rgb} ({remaining} more needed)")
        
        self._selected = None
        self._paused = False
        return True
    
    def _draw(self) -> np.ndarray:
        """Draw frame with overlays based on current phase."""
        if self._frame is None:
            return np.zeros((480, 640, 3), dtype=np.uint8)
        
        frame = self._frame.copy()
        
        # Draw court boundary if we have samples
        if self.court_detector.is_color_set:
            frame = self.court_detector.draw_boundary(frame, (0, 255, 255), 3)
        
        # Draw player boxes in team assignment phase
        if self._phase == CalibrationPhase.TEAM_ASSIGNMENT:
            for obj in self._objects:
                x1, y1, x2, y2 = obj.bbox.to_int_tuple()
                on_court = self.court_detector.is_on_court(self._frame, (x1, y1, x2, y2))
                
                if self._selected and obj.track_id == self._selected.track_id:
                    color, thick = (0, 255, 255), 3
                elif on_court:
                    color, thick = (255, 255, 255), 2
                else:
                    color, thick = (100, 100, 100), 1
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, thick)
        
        self._draw_status(frame)
        self._draw_instructions(frame)
        
        return frame
    
    def _draw_status(self, frame: np.ndarray) -> None:
        """Draw status bar at top."""
        h, w = frame.shape[:2]
        
        # Background
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 90), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
        
        y = 20
        
        # Phase indicator
        if self._phase in [CalibrationPhase.COURT_CALIBRATION, CalibrationPhase.COURT_REVIEW]:
            phase_text = "PHASE 1: COURT CALIBRATION"
            phase_color = (0, 255, 255)
        else:
            phase_text = "PHASE 2: TEAM ASSIGNMENT"
            phase_color = (0, 255, 0)
        
        cv2.putText(frame, phase_text, (10, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.55, phase_color, 2)
        
        y += 25
        
        # Court samples
        court_count = self.court_detector.court_sample_count
        oob_count = self.court_detector.oob_sample_count
        
        cv2.putText(frame, f"Court: {court_count}", (10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Draw court color swatches
        x = 80
        for rgb in self.court_detector.get_court_colors()[:8]:
            bgr = (rgb[2], rgb[1], rgb[0])
            cv2.rectangle(frame, (x, y - 10), (x + 12, y + 2), bgr, -1)
            cv2.rectangle(frame, (x, y - 10), (x + 12, y + 2), (255, 255, 255), 1)
            x += 15
        
        # OOB samples
        x = max(x + 20, 220)
        cv2.putText(frame, f"OOB: {oob_count}", (x, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        x += 50
        for rgb in self.court_detector.get_oob_colors()[:6]:
            bgr = (rgb[2], rgb[1], rgb[0])
            cv2.rectangle(frame, (x, y - 10), (x + 12, y + 2), bgr, -1)
            cv2.rectangle(frame, (x, y - 10), (x + 12, y + 2), (150, 150, 150), 1)
            x += 15
        
        y += 22
        
        # Team samples (Phase 2)
        if self._phase == CalibrationPhase.TEAM_ASSIGNMENT:
            x = 10
            cv2.putText(frame, "Teams:", (x, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            x = 60
            
            for team in [Team.TEAM_A, Team.TEAM_B, Team.REFEREE]:
                count = self.color_store.get_sample_count(team)
                rgb = self.color_store.get_team_color_rgb(team)
                
                if rgb:
                    bgr = (rgb[2], rgb[1], rgb[0])
                    cv2.rectangle(frame, (x, y - 10), (x + 12, y + 2), bgr, -1)
                    x += 15
                
                label = team.name.replace("TEAM_", "")
                text = f"{label}:{count}/{self.SAMPLES_PER_TEAM}"
                color = (0, 255, 0) if count >= self.SAMPLES_PER_TEAM else (255, 255, 255)
                cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                x += 70
    
    def _draw_instructions(self, frame: np.ndarray) -> None:
        """Draw instructions at bottom."""
        h, w = frame.shape[:2]
        
        # Background
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h - 50), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
        
        if self._phase == CalibrationPhase.COURT_CALIBRATION:
            line1 = "LEFT-click: Court color | RIGHT-click: Out-of-bounds color"
            line2 = "[ENTER] Finish & review | [Q] Quit"
            color = (0, 255, 255)
        
        elif self._phase == CalibrationPhase.COURT_REVIEW:
            line1 = "Review court detection. Continue clicking to improve."
            line2 = "[SPACE] Accept & go to teams | [R] Restart video | [Q] Quit"
            color = (0, 255, 0)
        
        elif self._phase == CalibrationPhase.TEAM_ASSIGNMENT:
            if self._paused and self._selected:
                line1 = "Player selected!"
                line2 = "[A] Team A | [B] Team B | [R] Referee | [ESC] Cancel"
                color = (0, 255, 255)
            elif self.color_store.is_complete:
                line1 = "All teams assigned!"
                line2 = "[ENTER] Start processing | [Q] Quit"
                color = (0, 255, 0)
            else:
                line1 = "Click on players inside the yellow boundary"
                line2 = "[ENTER] Finish (if complete) | [Q] Quit"
                color = (200, 200, 200)
        else:
            line1 = "Calibration complete!"
            line2 = ""
            color = (0, 255, 0)
        
        cv2.putText(frame, line1, (10, h - 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
        cv2.putText(frame, line2, (10, h - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    
    def update(
        self,
        frame: np.ndarray,
        tracked_objects: List[TrackedObject],
        is_last_frame: bool = False
    ) -> Optional[str]:
        """
        Update with new frame.
        
        Returns:
            None: continue
            "restart": restart video for more court calibration
            "phase2": restart video for team assignment
            "complete": calibration complete
            "quit": user quit
        """
        self._frame = frame
        self._objects = tracked_objects
        
        # Update court detection
        if self.court_detector.is_color_set:
            self.court_detector.update_frame(frame)
        
        # Initialize window
        if not self._window_ready:
            cv2.namedWindow(self.window_name)
            cv2.setMouseCallback(self.window_name, self._on_mouse)
            self._window_ready = True
        
        # Handle clicks based on phase
        if self._phase in [CalibrationPhase.COURT_CALIBRATION, CalibrationPhase.COURT_REVIEW]:
            if self._left_click:
                self._handle_court_click(*self._left_click, ClickType.COURT)
                self._left_click = None
            if self._right_click:
                self._handle_court_click(*self._right_click, ClickType.OUT_OF_BOUNDS)
                self._right_click = None
        
        elif self._phase == CalibrationPhase.TEAM_ASSIGNMENT:
            if self._left_click and not self._paused:
                self._handle_team_click(*self._left_click)
                self._left_click = None
        
        # Clear unused clicks
        self._left_click = None
        self._right_click = None
        
        # Draw
        display = self._draw()
        cv2.imshow(self.window_name, display)
        
        # Handle video end during court calibration
        if is_last_frame and self._phase == CalibrationPhase.COURT_CALIBRATION:
            print("\nVideo ended. Entering review mode...")
            self._phase = CalibrationPhase.COURT_REVIEW
        
        # Key handling
        wait_time = 0 if (self._paused or self._phase == CalibrationPhase.COURT_REVIEW) else self.frame_delay
        key = cv2.waitKey(wait_time) & 0xFF
        
        return self._handle_key(key)
    
    def _handle_key(self, key: int) -> Optional[str]:
        """Handle keyboard input."""
        
        # Quit
        if key == ord('q'):
            self.cleanup()
            return "quit"
        
        # Phase-specific handling
        if self._phase == CalibrationPhase.COURT_CALIBRATION:
            if key == 13:  # Enter - finish early, go to review
                if self.court_detector.court_sample_count >= 1:
                    print("\nEntering review mode...")
                    self._phase = CalibrationPhase.COURT_REVIEW
                else:
                    print("Add at least 1 court color sample first!")
        
        elif self._phase == CalibrationPhase.COURT_REVIEW:
            if key == ord(' '):  # Space - accept and go to phase 2
                if self.court_detector.court_sample_count >= 1:
                    print("\nCourt calibration accepted. Restarting for team assignment...")
                    self._phase = CalibrationPhase.TEAM_ASSIGNMENT
                    self.court_detector.reset_temporal()
                    return "phase2"
            elif key == ord('r'):  # R - restart video to add more samples
                print("\nRestarting video for more samples...")
                self.court_detector.reset_temporal()
                self._phase = CalibrationPhase.COURT_CALIBRATION
                return "restart"
        
        elif self._phase == CalibrationPhase.TEAM_ASSIGNMENT:
            if key == 27:  # Escape - cancel selection
                self._selected = None
                self._paused = False
            
            elif self._paused and self._selected:
                if key == ord('a'):
                    self._assign_team(Team.TEAM_A)
                elif key == ord('b'):
                    self._assign_team(Team.TEAM_B)
                elif key == ord('r'):
                    self._assign_team(Team.REFEREE)
            
            elif key == 13:  # Enter - finish if complete
                if self.color_store.is_complete:
                    self._phase = CalibrationPhase.COMPLETE
                    self.cleanup()
                    return "complete"
        
        return None
    
    def cleanup(self) -> None:
        """Close window."""
        if self._window_ready:
            cv2.destroyWindow(self.window_name)
            self._window_ready = False
    
    @property
    def phase(self) -> CalibrationPhase:
        return self._phase
    
    def is_complete(self) -> bool:
        return self._phase == CalibrationPhase.COMPLETE
    
    def is_court_ready(self) -> bool:
        return self.court_detector.court_sample_count >= 1
    
    def get_color_store(self) -> ColorReferenceStore:
        return self.color_store
    
    def get_court_detector(self) -> CourtDetector:
        return self.court_detector