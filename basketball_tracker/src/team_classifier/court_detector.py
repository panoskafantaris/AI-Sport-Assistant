"""
Advanced court detector with court colors + out-of-bounds colors.

Features:
1. Court colors (OR logic) - any match = court
2. Out-of-bounds colors (OR logic) - any match = NOT court
3. Line detection for edge refinement
4. Geometric estimation for robust boundaries
5. Temporal smoothing for stability
"""
import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List
import json

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import config

from .court_colors import CourtColorManager
from .geometry_estimator import GeometryEstimator


class CourtDetector:
    """
    Advanced court detection using court + out-of-bounds colors.
    
    The boundary is where court colors meet out-of-bounds colors.
    Geometric estimation infers edges even when partially occluded.
    """
    
    def __init__(
        self,
        min_court_area_ratio: float = 0.08,
        temporal_smoothing: float = 0.6
    ):
        self.min_court_area_ratio = min_court_area_ratio
        self.temporal_smoothing = temporal_smoothing
        
        # Color manager (court + OOB samples)
        self._color_manager = CourtColorManager()
        
        # Geometry estimator
        self._geometry = GeometryEstimator()
        
        # Current detection
        self._current_mask: Optional[np.ndarray] = None
        self._current_polygon: Optional[np.ndarray] = None
        self._prev_polygon: Optional[np.ndarray] = None
    
    def add_court_sample(
        self,
        frame: np.ndarray,
        x: int, y: int,
        label: str = ""
    ) -> Tuple[int, int, int]:
        """Add a court color sample."""
        return self._color_manager.add_court_sample(frame, x, y, label=label)
    
    def add_oob_sample(
        self,
        frame: np.ndarray,
        x: int, y: int,
        label: str = ""
    ) -> Tuple[int, int, int]:
        """Add an out-of-bounds color sample."""
        return self._color_manager.add_oob_sample(frame, x, y, label=label)
    
    def detect_court(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Detect court boundary polygon.
        
        Uses both court colors and OOB colors to find the boundary.
        """
        if not self._color_manager.has_court_samples:
            return None
        
        # Create court mask (includes OOB subtraction)
        mask = self._color_manager.create_court_mask(frame)
        
        # Clean and fill
        mask = self._clean_mask(mask, frame)
        self._current_mask = mask
        
        # Detect lines
        lines = self._geometry.detect_lines(frame, mask)
        
        # Estimate boundary
        polygon = self._geometry.estimate_boundary_from_mask(mask, lines)
        
        # Temporal smoothing
        if polygon is not None:
            polygon = self._apply_temporal_smoothing(polygon)
        
        self._current_polygon = polygon
        return polygon
    
    def _clean_mask(self, mask: np.ndarray, frame: np.ndarray) -> np.ndarray:
        """Clean mask with morphological operations."""
        # Remove noise
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small)
        
        # Fill gaps from players/objects
        kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 35))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_large)
        
        # Connect regions
        kernel_connect = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
        mask = cv2.dilate(mask, kernel_connect, iterations=1)
        
        # If we have OOB samples, use them to refine edges
        if self._color_manager.has_oob_samples:
            oob_mask = self._color_manager.create_oob_mask(frame)
            # Dilate OOB slightly to create a clear boundary
            oob_dilated = cv2.dilate(oob_mask, kernel_small, iterations=2)
            # Subtract OOB from court mask
            mask = cv2.bitwise_and(mask, cv2.bitwise_not(oob_dilated))
        
        # Keep largest component
        mask = self._keep_largest_component(mask)
        
        # Fill holes
        mask = self._fill_holes(mask)
        
        # Erode back
        kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (12, 12))
        mask = cv2.erode(mask, kernel_erode, iterations=1)
        
        return mask
    
    def _keep_largest_component(self, mask: np.ndarray) -> np.ndarray:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        if num_labels <= 1:
            return mask
        
        areas = stats[1:, cv2.CC_STAT_AREA]
        largest_idx = 1 + np.argmax(areas)
        
        result = np.zeros_like(mask)
        result[labels == largest_idx] = 255
        return result
    
    def _fill_holes(self, mask: np.ndarray) -> np.ndarray:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return mask
        
        filled = np.zeros_like(mask)
        cv2.drawContours(filled, contours, -1, 255, -1)
        return filled
    
    def _apply_temporal_smoothing(self, polygon: np.ndarray) -> np.ndarray:
        if self._prev_polygon is None or self.temporal_smoothing <= 0:
            self._prev_polygon = polygon.copy()
            return polygon
        
        if len(polygon) != len(self._prev_polygon):
            self._prev_polygon = polygon.copy()
            return polygon
        
        alpha = self.temporal_smoothing
        current = polygon.reshape(-1, 2).astype(float)
        previous = self._prev_polygon.reshape(-1, 2).astype(float)
        
        matched_prev = self._match_vertices(current, previous)
        smoothed = (1 - alpha) * current + alpha * matched_prev
        
        self._prev_polygon = smoothed.reshape(-1, 1, 2).astype(np.int32)
        return self._prev_polygon.copy()
    
    def _match_vertices(self, current: np.ndarray, previous: np.ndarray) -> np.ndarray:
        n = len(current)
        best_shift = 0
        best_dist = float('inf')
        
        for shift in range(n):
            rotated = np.roll(previous, shift, axis=0)
            dist = np.sum(np.linalg.norm(current - rotated, axis=1))
            if dist < best_dist:
                best_dist = dist
                best_shift = shift
        
        return np.roll(previous, best_shift, axis=0)
    
    def is_on_court(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> bool:
        if not self._color_manager.has_court_samples:
            return True
        
        if self._current_polygon is None:
            return True
        
        x1, y1, x2, y2 = [int(v) for v in bbox]
        feet_x = (x1 + x2) // 2
        feet_y = y2
        
        result = cv2.pointPolygonTest(
            self._current_polygon,
            (float(feet_x), float(feet_y)),
            False
        )
        return result >= 0
    
    def draw_boundary(
        self,
        frame: np.ndarray,
        color: Tuple[int, int, int] = (0, 255, 255),
        thickness: int = 3,
        show_lines: bool = False
    ) -> np.ndarray:
        result = frame.copy()
        
        if show_lines:
            for line in self._geometry.detected_lines:
                cv2.line(result, (line.x1, line.y1), (line.x2, line.y2), (0, 0, 255), 1)
        
        if self._current_polygon is not None and len(self._current_polygon) >= 3:
            overlay = result.copy()
            cv2.fillPoly(overlay, [self._current_polygon], color)
            cv2.addWeighted(overlay, 0.12, result, 0.88, 0, result)
            
            cv2.polylines(result, [self._current_polygon], True, color, thickness)
            
            pts = self._current_polygon.reshape(-1, 2)
            for pt in pts:
                cv2.circle(result, tuple(pt.astype(int)), 5, color, -1)
        
        return result
    
    def update_frame(self, frame: np.ndarray) -> None:
        self._current_mask = None
        self._current_polygon = None
        self.detect_court(frame)
    
    def reset_temporal(self) -> None:
        """Reset temporal smoothing (call when restarting video)."""
        self._prev_polygon = None
    
    # Properties
    @property
    def is_color_set(self) -> bool:
        return self._color_manager.has_court_samples
    
    @property
    def court_sample_count(self) -> int:
        return self._color_manager.court_count
    
    @property
    def oob_sample_count(self) -> int:
        return self._color_manager.oob_count
    
    @property
    def sample_count(self) -> int:
        return self._color_manager.total_count
    
    @property
    def court_color_rgb(self) -> Optional[Tuple[int, int, int]]:
        colors = self._color_manager.get_court_colors_rgb()
        if colors:
            return colors[0]
        return None
    
    def get_all_colors(self) -> List[Tuple[int, int, int]]:
        return self._color_manager.get_court_colors_rgb()
    
    def get_court_colors(self) -> List[Tuple[int, int, int]]:
        return self._color_manager.get_court_colors_rgb()
    
    def get_oob_colors(self) -> List[Tuple[int, int, int]]:
        return self._color_manager.get_oob_colors_rgb()
    
    # Persistence
    def save(self, filepath: str = None) -> Path:
        if filepath is None:
            filepath = config.RESULTS_DIR / "court_color.json"
        else:
            filepath = Path(filepath)
        
        filepath.parent.mkdir(parents=True, exist_ok=True)
        self._color_manager.save(filepath)
        return filepath
    
    def load(self, filepath: str = None) -> bool:
        if filepath is None:
            filepath = config.RESULTS_DIR / "court_color.json"
        else:
            filepath = Path(filepath)
        
        return self._color_manager.load(filepath)
    
    def reset(self) -> None:
        self._color_manager.clear()
        self._current_mask = None
        self._current_polygon = None
        self._prev_polygon = None