"""
Court color management - court colors AND out-of-bounds colors.

Court colors: Combined with OR logic (any match = court)
Out-of-bounds colors: Combined with OR logic (any match = NOT court)

The boundary is where court colors meet out-of-bounds colors.
"""
import cv2
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass
import json
from pathlib import Path


@dataclass
class ColorSample:
    """Single color sample."""
    hsv: Tuple[int, int, int]
    rgb: Tuple[int, int, int]
    label: str = ""
    
    # Tolerances
    hue_tol: int = 15
    sat_tol: int = 50
    val_tol: int = 60
    
    def create_mask(self, hsv_frame: np.ndarray) -> np.ndarray:
        """Create binary mask for this color."""
        h, s, v = self.hsv
        
        lower = np.array([
            max(0, h - self.hue_tol),
            max(0, s - self.sat_tol),
            max(0, v - self.val_tol)
        ])
        upper = np.array([
            min(179, h + self.hue_tol),
            min(255, s + self.sat_tol),
            min(255, v + self.val_tol)
        ])
        
        # Handle hue wraparound
        if h - self.hue_tol < 0:
            mask1 = cv2.inRange(hsv_frame, 
                np.array([0, lower[1], lower[2]]), 
                np.array([h + self.hue_tol, upper[1], upper[2]]))
            mask2 = cv2.inRange(hsv_frame, 
                np.array([180 + (h - self.hue_tol), lower[1], lower[2]]),
                np.array([179, upper[1], upper[2]]))
            return cv2.bitwise_or(mask1, mask2)
        elif h + self.hue_tol > 179:
            mask1 = cv2.inRange(hsv_frame, 
                np.array([h - self.hue_tol, lower[1], lower[2]]),
                np.array([179, upper[1], upper[2]]))
            mask2 = cv2.inRange(hsv_frame, 
                np.array([0, lower[1], lower[2]]),
                np.array([(h + self.hue_tol) - 180, upper[1], upper[2]]))
            return cv2.bitwise_or(mask1, mask2)
        else:
            return cv2.inRange(hsv_frame, lower, upper)
    
    def to_dict(self) -> dict:
        return {
            "hsv": list(self.hsv),
            "rgb": list(self.rgb),
            "label": self.label,
            "hue_tol": self.hue_tol,
            "sat_tol": self.sat_tol,
            "val_tol": self.val_tol
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'ColorSample':
        return cls(
            hsv=tuple(data["hsv"]),
            rgb=tuple(data["rgb"]),
            label=data.get("label", ""),
            hue_tol=data.get("hue_tol", 15),
            sat_tol=data.get("sat_tol", 50),
            val_tol=data.get("val_tol", 60)
        )


class CourtColorManager:
    """
    Manages court AND out-of-bounds color samples.
    
    Court colors: OR'd together (any match = court)
    Out-of-bounds colors: OR'd together (any match = NOT court)
    
    The boundary line is where these two regions meet.
    """
    
    def __init__(self):
        self._court_samples: List[ColorSample] = []
        self._oob_samples: List[ColorSample] = []  # Out of bounds
    
    def add_court_sample(
        self,
        frame: np.ndarray,
        x: int, y: int,
        radius: int = 25,
        label: str = ""
    ) -> Tuple[int, int, int]:
        """Add a court color sample."""
        sample = self._create_sample(frame, x, y, radius, label)
        self._court_samples.append(sample)
        return sample.rgb
    
    def add_oob_sample(
        self,
        frame: np.ndarray,
        x: int, y: int,
        radius: int = 25,
        label: str = ""
    ) -> Tuple[int, int, int]:
        """Add an out-of-bounds color sample."""
        sample = self._create_sample(frame, x, y, radius, label)
        self._oob_samples.append(sample)
        return sample.rgb
    
    def _create_sample(
        self,
        frame: np.ndarray,
        x: int, y: int,
        radius: int,
        label: str
    ) -> ColorSample:
        """Create a color sample from click position."""
        h, w = frame.shape[:2]
        
        x1 = max(0, x - radius)
        y1 = max(0, y - radius)
        x2 = min(w, x + radius)
        y2 = min(h, y + radius)
        
        region = frame[y1:y2, x1:x2]
        
        hsv_region = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        hsv = (
            int(np.median(hsv_region[:, :, 0])),
            int(np.median(hsv_region[:, :, 1])),
            int(np.median(hsv_region[:, :, 2]))
        )
        
        rgb_region = cv2.cvtColor(region, cv2.COLOR_BGR2RGB)
        rgb = (
            int(np.median(rgb_region[:, :, 0])),
            int(np.median(rgb_region[:, :, 1])),
            int(np.median(rgb_region[:, :, 2]))
        )
        
        return ColorSample(hsv=hsv, rgb=rgb, label=label)
    
    def create_court_mask(self, frame: np.ndarray) -> np.ndarray:
        """
        Create mask: court pixels = 255, others = 0.
        Uses both court colors (include) and OOB colors (exclude).
        """
        if not self._court_samples:
            return np.ones(frame.shape[:2], dtype=np.uint8) * 255
        
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Start with court color mask (OR all court samples)
        court_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        for sample in self._court_samples:
            mask = sample.create_mask(hsv)
            court_mask = cv2.bitwise_or(court_mask, mask)
        
        # Create OOB mask if we have samples
        if self._oob_samples:
            oob_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            for sample in self._oob_samples:
                mask = sample.create_mask(hsv)
                oob_mask = cv2.bitwise_or(oob_mask, mask)
            
            # Subtract OOB from court
            court_mask = cv2.bitwise_and(court_mask, cv2.bitwise_not(oob_mask))
        
        return court_mask
    
    def create_oob_mask(self, frame: np.ndarray) -> np.ndarray:
        """Create mask of out-of-bounds areas."""
        if not self._oob_samples:
            return np.zeros(frame.shape[:2], dtype=np.uint8)
        
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        oob_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        for sample in self._oob_samples:
            mask = sample.create_mask(hsv)
            oob_mask = cv2.bitwise_or(oob_mask, mask)
        
        return oob_mask
    
    @property
    def court_count(self) -> int:
        return len(self._court_samples)
    
    @property
    def oob_count(self) -> int:
        return len(self._oob_samples)
    
    @property
    def total_count(self) -> int:
        return len(self._court_samples) + len(self._oob_samples)
    
    @property
    def has_court_samples(self) -> bool:
        return len(self._court_samples) > 0
    
    @property
    def has_oob_samples(self) -> bool:
        return len(self._oob_samples) > 0
    
    def get_court_colors_rgb(self) -> List[Tuple[int, int, int]]:
        return [s.rgb for s in self._court_samples]
    
    def get_oob_colors_rgb(self) -> List[Tuple[int, int, int]]:
        return [s.rgb for s in self._oob_samples]
    
    def clear(self) -> None:
        self._court_samples.clear()
        self._oob_samples.clear()
    
    def save(self, filepath: Path) -> None:
        data = {
            "court_samples": [s.to_dict() for s in self._court_samples],
            "oob_samples": [s.to_dict() for s in self._oob_samples]
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load(self, filepath: Path) -> bool:
        if not filepath.exists():
            return False
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            self._court_samples = [ColorSample.from_dict(s) for s in data.get("court_samples", [])]
            self._oob_samples = [ColorSample.from_dict(s) for s in data.get("oob_samples", [])]
            return True
        except Exception as e:
            print(f"Error loading colors: {e}")
            return False