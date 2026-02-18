"""
Geometric estimation of court boundaries using perspective analysis.

Uses:
- Line detection to find court edges
- Vanishing point detection for perspective understanding
- RANSAC-like fitting for robust estimation
- Geometric constraints (parallel sides, straight edges)
"""
import cv2
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class Line:
    """Represents a line segment."""
    x1: int
    y1: int
    x2: int
    y2: int
    
    @property
    def length(self) -> float:
        return np.sqrt((self.x2 - self.x1)**2 + (self.y2 - self.y1)**2)
    
    @property
    def angle(self) -> float:
        """Angle in radians."""
        return np.arctan2(self.y2 - self.y1, self.x2 - self.x1)
    
    @property
    def midpoint(self) -> Tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)
    
    def to_homogeneous(self) -> np.ndarray:
        """Convert to homogeneous line representation (a, b, c) where ax + by + c = 0."""
        p1 = np.array([self.x1, self.y1, 1])
        p2 = np.array([self.x2, self.y2, 1])
        return np.cross(p1, p2)


class GeometryEstimator:
    """
    Estimates court boundary geometry using perspective analysis.
    
    Key insight: Basketball court is a rectangle in 3D space.
    In 2D image, it appears as a quadrilateral with:
    - Opposite edges converging to vanishing points
    - Straight edges (even if partially occluded)
    """
    
    def __init__(self):
        self._detected_lines: List[Line] = []
        self._vanishing_points: List[np.ndarray] = []
    
    def detect_lines(
        self,
        frame: np.ndarray,
        mask: np.ndarray,
        min_length: int = 80
    ) -> List[Line]:
        """
        Detect straight lines in the court area.
        
        Args:
            frame: BGR image
            mask: Binary mask of court area
            min_length: Minimum line length
        
        Returns:
            List of detected lines
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply mask - focus on court region + edges
        dilated_mask = cv2.dilate(mask, np.ones((20, 20), np.uint8), iterations=2)
        masked = cv2.bitwise_and(gray, gray, mask=dilated_mask)
        
        # Edge detection with adaptive thresholds
        blurred = cv2.GaussianBlur(masked, (5, 5), 0)
        edges = cv2.Canny(blurred, 30, 100, apertureSize=3)
        
        # Dilate edges to connect broken lines
        edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
        
        # Probabilistic Hough transform
        hough_lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=50,
            minLineLength=min_length,
            maxLineGap=20
        )
        
        if hough_lines is None:
            self._detected_lines = []
            return []
        
        lines = [Line(l[0][0], l[0][1], l[0][2], l[0][3]) for l in hough_lines]
        
        # Filter out very short lines
        lines = [l for l in lines if l.length >= min_length]
        
        self._detected_lines = lines
        return lines
    
    def find_vanishing_points(self, lines: List[Line], frame_shape: Tuple[int, int]) -> List[np.ndarray]:
        """
        Find vanishing points from detected lines.
        
        Groups lines by angle and finds intersection of parallel groups.
        """
        if len(lines) < 4:
            return []
        
        h, w = frame_shape
        
        # Group lines by angle (horizontal-ish vs vertical-ish)
        horizontal = []  # roughly horizontal lines
        vertical = []    # roughly vertical lines
        
        for line in lines:
            angle = abs(line.angle)
            if angle < np.pi / 4 or angle > 3 * np.pi / 4:
                horizontal.append(line)
            else:
                vertical.append(line)
        
        vanishing_points = []
        
        # Find vanishing point for each group
        for group in [horizontal, vertical]:
            if len(group) < 2:
                continue
            
            vp = self._find_vanishing_point_ransac(group, (h, w))
            if vp is not None:
                vanishing_points.append(vp)
        
        self._vanishing_points = vanishing_points
        return vanishing_points
    
    def _find_vanishing_point_ransac(
        self,
        lines: List[Line],
        frame_shape: Tuple[int, int],
        iterations: int = 100
    ) -> Optional[np.ndarray]:
        """
        Find vanishing point using RANSAC-like approach.
        """
        if len(lines) < 2:
            return None
        
        h, w = frame_shape
        best_vp = None
        best_inliers = 0
        
        for _ in range(iterations):
            # Random sample 2 lines
            idx = np.random.choice(len(lines), 2, replace=False)
            l1 = lines[idx[0]].to_homogeneous()
            l2 = lines[idx[1]].to_homogeneous()
            
            # Find intersection
            vp = np.cross(l1, l2)
            
            if abs(vp[2]) < 1e-6:
                continue  # Parallel lines
            
            vp = vp / vp[2]  # Normalize
            
            # Count inliers (lines passing near this vanishing point)
            inliers = 0
            for line in lines:
                l_hom = line.to_homogeneous()
                dist = abs(np.dot(l_hom, vp)) / np.linalg.norm(l_hom[:2])
                if dist < 20:
                    inliers += 1
            
            if inliers > best_inliers:
                best_inliers = inliers
                best_vp = vp[:2]
        
        return best_vp
    
    def estimate_boundary_from_mask(
        self,
        mask: np.ndarray,
        lines: List[Line]
    ) -> Optional[np.ndarray]:
        """
        Estimate court boundary polygon from mask and detected lines.
        
        Strategy:
        1. Get convex hull of mask (basic shape)
        2. Simplify to polygon
        3. Refine edges using detected lines
        4. Apply geometric constraints
        """
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # Get largest contour
        largest = max(contours, key=cv2.contourArea)
        
        # Check minimum area (at least 8% of frame)
        frame_area = mask.shape[0] * mask.shape[1]
        if cv2.contourArea(largest) < frame_area * 0.08:
            return None
        
        # Convex hull
        hull = cv2.convexHull(largest)
        
        # Simplify to polygon
        polygon = self._simplify_to_polygon(hull)
        
        # Refine using detected lines
        if lines:
            polygon = self._refine_polygon_with_lines(polygon, lines)
        
        return polygon
    
    def _simplify_to_polygon(self, hull: np.ndarray) -> np.ndarray:
        """Simplify convex hull to clean polygon (4-8 vertices)."""
        perimeter = cv2.arcLength(hull, True)
        
        # Try different simplification levels
        for eps_factor in [0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05]:
            epsilon = eps_factor * perimeter
            approx = cv2.approxPolyDP(hull, epsilon, True)
            
            if 4 <= len(approx) <= 8:
                return approx
        
        # If still too many points, force to 4-8
        if len(approx) > 8:
            return self._reduce_to_corners(hull, 6)
        
        return approx if len(approx) >= 3 else hull
    
    def _reduce_to_corners(self, contour: np.ndarray, n_points: int) -> np.ndarray:
        """Reduce contour to n most significant corner points."""
        pts = contour.reshape(-1, 2).astype(float)
        
        if len(pts) <= n_points:
            return contour
        
        # Find points with maximum curvature
        angles = []
        for i in range(len(pts)):
            p1 = pts[i - 1]
            p2 = pts[i]
            p3 = pts[(i + 1) % len(pts)]
            
            v1 = p1 - p2
            v2 = p3 - p2
            
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
            angle = np.arccos(np.clip(cos_angle, -1, 1))
            angles.append((i, angle))
        
        # Sort by angle (sharper corners first)
        angles.sort(key=lambda x: x[1])
        
        # Take n sharpest corners
        indices = sorted([a[0] for a in angles[:n_points]])
        corners = pts[indices]
        
        return corners.reshape(-1, 1, 2).astype(np.int32)
    
    def _refine_polygon_with_lines(
        self,
        polygon: np.ndarray,
        lines: List[Line]
    ) -> np.ndarray:
        """
        Refine polygon edges to align with detected court lines.
        """
        if len(lines) < 2:
            return polygon
        
        refined = polygon.copy()
        pts = refined.reshape(-1, 2).astype(float)
        n = len(pts)
        
        # For each edge of polygon
        for i in range(n):
            p1 = pts[i]
            p2 = pts[(i + 1) % n]
            
            edge_vec = p2 - p1
            edge_len = np.linalg.norm(edge_vec)
            if edge_len < 20:
                continue
            
            edge_dir = edge_vec / edge_len
            
            # Find best matching detected line
            best_line = None
            best_score = float('inf')
            
            for line in lines:
                line_vec = np.array([line.x2 - line.x1, line.y2 - line.y1])
                line_len = np.linalg.norm(line_vec)
                if line_len < 50:
                    continue
                
                line_dir = line_vec / line_len
                
                # Check if roughly parallel
                dot = abs(np.dot(edge_dir, line_dir))
                if dot < 0.8:
                    continue
                
                # Check distance
                mid = (p1 + p2) / 2
                line_mid = np.array(line.midpoint)
                dist = np.linalg.norm(mid - line_mid)
                
                # Score: prefer close and parallel
                score = dist * (2 - dot)
                
                if score < best_score and dist < 80:
                    best_score = score
                    best_line = line
            
            # Adjust edge toward detected line
            if best_line is not None:
                self._adjust_edge_to_line(pts, i, best_line)
        
        return pts.reshape(-1, 1, 2).astype(np.int32)
    
    def _adjust_edge_to_line(
        self,
        pts: np.ndarray,
        edge_idx: int,
        line: Line
    ) -> None:
        """Adjust polygon edge to align with detected line."""
        n = len(pts)
        i = edge_idx
        j = (i + 1) % n
        
        line_p1 = np.array([line.x1, line.y1])
        line_p2 = np.array([line.x2, line.y2])
        line_vec = line_p2 - line_p1
        line_len = np.linalg.norm(line_vec)
        
        if line_len < 1:
            return
        
        line_dir = line_vec / line_len
        
        # Project polygon vertices onto line and move partially
        for idx in [i, j]:
            pt = pts[idx]
            t = np.dot(pt - line_p1, line_dir)
            proj = line_p1 + t * line_dir
            
            # Move 40% toward the line
            pts[idx] = pt + 0.4 * (proj - pt)
    
    @property
    def detected_lines(self) -> List[Line]:
        return self._detected_lines
    
    @property
    def vanishing_points(self) -> List[np.ndarray]:
        return self._vanishing_points