"""
Video loading and frame extraction.
"""
import cv2
import numpy as np
from pathlib import Path
from typing import Generator, Optional, Tuple

from .models import VideoMetadata


class VideoLoader:
    """Handles video file loading and frame extraction."""
    
    def __init__(self, video_path: str):
        """
        Initialize video loader.
        
        Args:
            video_path: Path to the video file
        """
        self.video_path = Path(video_path)
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        self._cap: Optional[cv2.VideoCapture] = None
        self._metadata: Optional[VideoMetadata] = None
    
    @property
    def metadata(self) -> VideoMetadata:
        """Get video metadata, loading if necessary."""
        if self._metadata is None:
            self._metadata = self._load_metadata()
        return self._metadata
    
    def _load_metadata(self) -> VideoMetadata:
        """Extract metadata from video file."""
        cap = cv2.VideoCapture(str(self.video_path))
        
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {self.video_path}")
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        
        cap.release()
        
        return VideoMetadata(
            filepath=str(self.video_path),
            width=width,
            height=height,
            fps=fps,
            total_frames=total_frames,
            duration_seconds=duration
        )
    
    def open(self) -> None:
        """Open video capture."""
        if self._cap is not None:
            self._cap.release()
        self._cap = cv2.VideoCapture(str(self.video_path))
        if not self._cap.isOpened():
            raise IOError(f"Cannot open video: {self.video_path}")
    
    def close(self) -> None:
        """Release video capture."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
    
    def __enter__(self) -> "VideoLoader":
        self.open()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
    
    def frames(
        self, 
        skip: int = 0,
        max_frames: Optional[int] = None
    ) -> Generator[Tuple[int, float, np.ndarray], None, None]:
        """
        Iterate over video frames.
        
        Args:
            skip: Number of frames to skip between yields (0 = no skip)
            max_frames: Maximum number of frames to yield (None = all)
        
        Yields:
            Tuple of (frame_number, timestamp_ms, frame_array)
        """
        if self._cap is None:
            self.open()
        
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        frame_count = 0
        yielded_count = 0
        
        while True:
            ret, frame = self._cap.read()
            if not ret:
                break
            
            if frame_count % (skip + 1) == 0:
                timestamp_ms = self._cap.get(cv2.CAP_PROP_POS_MSEC)
                yield frame_count, timestamp_ms, frame
                yielded_count += 1
                
                if max_frames and yielded_count >= max_frames:
                    break
            
            frame_count += 1
    
    def get_frame(self, frame_number: int) -> Optional[np.ndarray]:
        """
        Get a specific frame by number.
        
        Args:
            frame_number: Frame index (0-based)
        
        Returns:
            Frame array or None if frame doesn't exist
        """
        if self._cap is None:
            self.open()
        
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = self._cap.read()
        
        return frame if ret else None