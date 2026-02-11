"""
Extract dominant jersey color from player bounding boxes.
"""
import cv2
import numpy as np
from typing import Optional, Tuple
from sklearn.cluster import KMeans

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import config


class ColorExtractor:
    """Extracts dominant jersey color from player bounding boxes."""
    
    def __init__(
        self,
        crop_top_ratio: float = config.JERSEY_CROP_TOP_RATIO,
        crop_bottom_ratio: float = config.JERSEY_CROP_BOTTOM_RATIO,
        crop_side_ratio: float = config.JERSEY_CROP_SIDE_RATIO,
        n_clusters: int = config.DOMINANT_COLOR_CLUSTERS
    ):
        """
        Initialize color extractor.
        
        Args:
            crop_top_ratio: Ratio to crop from top (skip head)
            crop_bottom_ratio: Ratio where jersey ends (skip legs)
            crop_side_ratio: Ratio to crop from sides (skip arms)
            n_clusters: Number of K-means clusters for dominant color
        """
        self.crop_top_ratio = crop_top_ratio
        self.crop_bottom_ratio = crop_bottom_ratio
        self.crop_side_ratio = crop_side_ratio
        self.n_clusters = n_clusters
    
    def extract_jersey_region(
        self, 
        frame: np.ndarray, 
        bbox: Tuple[int, int, int, int]
    ) -> Optional[np.ndarray]:
        """
        Crop the jersey region from a bounding box.
        
        Args:
            frame: BGR image
            bbox: Bounding box (x1, y1, x2, y2)
        
        Returns:
            Cropped jersey region or None if too small
        """
        x1, y1, x2, y2 = bbox
        height = y2 - y1
        width = x2 - x1
        
        # Skip if bbox is too small
        if height < 20 or width < 15:
            return None
        
        # Calculate jersey region (upper body)
        jersey_y1 = int(y1 + height * self.crop_top_ratio)
        jersey_y2 = int(y1 + height * self.crop_bottom_ratio)
        jersey_x1 = int(x1 + width * self.crop_side_ratio)
        jersey_x2 = int(x2 - width * self.crop_side_ratio)
        
        # Ensure valid bounds
        jersey_y1 = max(0, jersey_y1)
        jersey_y2 = min(frame.shape[0], jersey_y2)
        jersey_x1 = max(0, jersey_x1)
        jersey_x2 = min(frame.shape[1], jersey_x2)
        
        if jersey_y2 <= jersey_y1 or jersey_x2 <= jersey_x1:
            return None
        
        return frame[jersey_y1:jersey_y2, jersey_x1:jersey_x2]
    
    def get_dominant_color_hsv(
        self, 
        region: np.ndarray
    ) -> Optional[Tuple[int, int, int]]:
        """
        Extract dominant color from region using K-means in HSV space.
        
        Args:
            region: BGR image region
        
        Returns:
            Dominant color as (H, S, V) tuple or None
        """
        if region is None or region.size == 0:
            return None
        
        # Convert to HSV
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        
        # Reshape to list of pixels
        pixels = hsv.reshape(-1, 3)
        
        # Filter out very dark or very bright pixels (likely not jersey)
        mask = (pixels[:, 2] > 30) & (pixels[:, 2] < 250)  # V channel
        mask &= (pixels[:, 1] > 20)  # S channel (skip grays)
        filtered_pixels = pixels[mask]
        
        if len(filtered_pixels) < 10:
            # Not enough valid pixels, use all
            filtered_pixels = pixels
        
        if len(filtered_pixels) < 5:
            return None
        
        # K-means clustering
        n_clusters = min(self.n_clusters, len(filtered_pixels))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        kmeans.fit(filtered_pixels)
        
        # Find largest cluster
        labels, counts = np.unique(kmeans.labels_, return_counts=True)
        dominant_idx = labels[np.argmax(counts)]
        dominant_color = kmeans.cluster_centers_[dominant_idx]
        
        return tuple(int(c) for c in dominant_color)
    
    def hsv_to_rgb(self, hsv: Tuple[int, int, int]) -> Tuple[int, int, int]:
        """Convert HSV tuple to RGB tuple."""
        hsv_array = np.uint8([[list(hsv)]])
        rgb_array = cv2.cvtColor(hsv_array, cv2.COLOR_HSV2RGB)
        return tuple(int(c) for c in rgb_array[0, 0])
    
    def extract_color(
        self, 
        frame: np.ndarray, 
        bbox: Tuple[int, int, int, int]
    ) -> Optional[Tuple[Tuple[int, int, int], Tuple[int, int, int]]]:
        """
        Extract jersey color from bounding box.
        
        Args:
            frame: BGR image
            bbox: Bounding box (x1, y1, x2, y2)
        
        Returns:
            Tuple of (HSV color, RGB color) or None
        """
        jersey_region = self.extract_jersey_region(frame, bbox)
        if jersey_region is None:
            return None
        
        hsv_color = self.get_dominant_color_hsv(jersey_region)
        if hsv_color is None:
            return None
        
        rgb_color = self.hsv_to_rgb(hsv_color)
        return hsv_color, rgb_color