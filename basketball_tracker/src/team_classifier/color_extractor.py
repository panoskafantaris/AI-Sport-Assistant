"""
Extract the single most dominant jersey color from player bounding boxes.
"""
import cv2
import numpy as np
from typing import Optional, Tuple
from sklearn.cluster import KMeans
from collections import Counter

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import config


class ColorExtractor:
    """Extracts the single most dominant jersey color from player bounding boxes."""
    
    def __init__(
        self,
        crop_top_ratio: float = config.JERSEY_CROP_TOP_RATIO,
        crop_bottom_ratio: float = config.JERSEY_CROP_BOTTOM_RATIO,
        crop_side_ratio: float = config.JERSEY_CROP_SIDE_RATIO
    ):
        """
        Initialize color extractor.
        
        Args:
            crop_top_ratio: Ratio to crop from top (skip head)
            crop_bottom_ratio: Ratio where jersey ends (skip legs)
            crop_side_ratio: Ratio to crop from sides (skip arms/background)
        """
        self.crop_top_ratio = crop_top_ratio
        self.crop_bottom_ratio = crop_bottom_ratio
        self.crop_side_ratio = crop_side_ratio
    
    def extract_jersey_region(
        self, 
        frame: np.ndarray, 
        bbox: Tuple[int, int, int, int]
    ) -> Optional[np.ndarray]:
        """
        Crop the jersey region (upper body, center) from a bounding box.
        """
        x1, y1, x2, y2 = bbox
        height = y2 - y1
        width = x2 - x1
        
        if height < 20 or width < 15:
            return None
        
        # Focus on upper body center (jersey area)
        jersey_y1 = int(y1 + height * self.crop_top_ratio)
        jersey_y2 = int(y1 + height * self.crop_bottom_ratio)
        jersey_x1 = int(x1 + width * self.crop_side_ratio)
        jersey_x2 = int(x2 - width * self.crop_side_ratio)
        
        # Clamp to frame bounds
        jersey_y1 = max(0, jersey_y1)
        jersey_y2 = min(frame.shape[0], jersey_y2)
        jersey_x1 = max(0, jersey_x1)
        jersey_x2 = min(frame.shape[1], jersey_x2)
        
        if jersey_y2 <= jersey_y1 or jersey_x2 <= jersey_x1:
            return None
        
        return frame[jersey_y1:jersey_y2, jersey_x1:jersey_x2]
    
    def get_dominant_color(
        self, 
        region: np.ndarray
    ) -> Optional[Tuple[Tuple[int, int, int], Tuple[int, int, int]]]:
        """
        Extract THE single most dominant color from jersey region.
        
        Filters out skin tones and very dark/bright pixels.
        Returns the color that appears most frequently.
        
        Args:
            region: BGR image region
        
        Returns:
            Tuple of (HSV color, RGB color) or None
        """
        if region is None or region.size == 0:
            return None
        
        # Convert to HSV
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        pixels = hsv.reshape(-1, 3)
        
        # Filter out unwanted pixels:
        # - Very dark (shadows, black): V < 40
        # - Very bright (white, highlights): V > 240
        # - Skin tones: H in [0-25], S in [30-170], V > 80
        mask = np.ones(len(pixels), dtype=bool)
        
        # Remove dark pixels
        mask &= pixels[:, 2] > 40
        
        # Remove very bright pixels
        mask &= pixels[:, 2] < 240
        
        # Remove skin tones (approximate range)
        skin_mask = (
            (pixels[:, 0] >= 0) & (pixels[:, 0] <= 25) &
            (pixels[:, 1] >= 30) & (pixels[:, 1] <= 170) &
            (pixels[:, 2] > 80)
        )
        mask &= ~skin_mask
        
        # Remove very low saturation (grays that aren't referee uniform)
        # Keep some low saturation for referee detection
        # mask &= pixels[:, 1] > 15
        
        filtered_pixels = pixels[mask]
        
        if len(filtered_pixels) < 10:
            # Not enough valid pixels, use all non-dark pixels
            mask = pixels[:, 2] > 30
            filtered_pixels = pixels[mask]
        
        if len(filtered_pixels) < 5:
            return None
        
        # Use K-means with K=3 to find color clusters
        n_clusters = min(3, len(filtered_pixels))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        kmeans.fit(filtered_pixels)
        
        # Find the largest cluster (most dominant color)
        labels = kmeans.labels_
        label_counts = Counter(labels)
        dominant_label = label_counts.most_common(1)[0][0]
        
        # Get the centroid of the dominant cluster
        dominant_hsv = kmeans.cluster_centers_[dominant_label]
        dominant_hsv = tuple(int(c) for c in dominant_hsv)
        
        # Convert to RGB
        hsv_array = np.uint8([[list(dominant_hsv)]])
        rgb_array = cv2.cvtColor(hsv_array, cv2.COLOR_HSV2RGB)
        dominant_rgb = tuple(int(c) for c in rgb_array[0, 0])
        
        return dominant_hsv, dominant_rgb
    
    def extract_color(
        self, 
        frame: np.ndarray, 
        bbox: Tuple[int, int, int, int]
    ) -> Optional[Tuple[Tuple[int, int, int], Tuple[int, int, int]]]:
        """
        Extract the dominant jersey color from bounding box.
        
        Args:
            frame: BGR image
            bbox: Bounding box (x1, y1, x2, y2)
        
        Returns:
            Tuple of (HSV color, RGB color) or None
        """
        jersey_region = self.extract_jersey_region(frame, bbox)
        if jersey_region is None:
            return None
        
        return self.get_dominant_color(jersey_region)