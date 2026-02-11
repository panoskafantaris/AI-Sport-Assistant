"""
Object tracking using ByteTrack via ultralytics.
"""
import numpy as np
from typing import List, Optional
from ultralytics import YOLO

from .models import TrackedObject, BoundingBox
import config


class Tracker:
    """ByteTrack-based multi-object tracker using ultralytics."""
    
    def __init__(
        self,
        model_name: str = config.DETECTION_MODEL,
        tracker_type: str = config.TRACKER_TYPE,
        confidence_threshold: float = config.DETECTION_CONFIDENCE,
        iou_threshold: float = config.DETECTION_IOU_THRESHOLD,
        device: Optional[str] = None
    ):
        """
        Initialize tracker.
        
        Args:
            model_name: YOLO model name for detection
            tracker_type: Tracker type ('bytetrack' or 'botsort')
            confidence_threshold: Minimum confidence for detections
            iou_threshold: IOU threshold for NMS
            device: Device to run on ('cpu', 'cuda', or None for auto)
        """
        self.model = YOLO(model_name)
        self.tracker_type = tracker_type
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        self.target_class_id = config.PERSON_CLASS_ID
        
        # Track history for trajectory analysis
        self._track_history: dict = {}
    
    def track(self, frame: np.ndarray, persist: bool = True) -> List[TrackedObject]:
        """
        Detect and track objects in a frame.
        
        Args:
            frame: BGR image array
            persist: Whether to persist tracks across frames
        
        Returns:
            List of TrackedObject with persistent IDs
        """
        results = self.model.track(
            frame,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False,
            classes=[self.target_class_id],
            tracker=f"{self.tracker_type}.yaml",
            persist=persist
        )
        
        return self._parse_tracking_results(results[0])
    
    def _parse_tracking_results(self, result) -> List[TrackedObject]:
        """
        Parse tracking results into TrackedObject list.
        
        Args:
            result: Single YOLO tracking result
        
        Returns:
            List of TrackedObject
        """
        tracked_objects = []
        
        if result.boxes is None or len(result.boxes) == 0:
            return tracked_objects
        
        # Check if tracking IDs are available
        if result.boxes.id is None:
            return tracked_objects
        
        boxes = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy().astype(int)
        track_ids = result.boxes.id.cpu().numpy().astype(int)
        
        for bbox, conf, cls_id, track_id in zip(boxes, confidences, class_ids, track_ids):
            center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
            
            # Update track history
            if track_id not in self._track_history:
                self._track_history[track_id] = []
            self._track_history[track_id].append(center)
            
            # Keep history limited
            if len(self._track_history[track_id]) > 50:
                self._track_history[track_id] = self._track_history[track_id][-50:]
            
            tracked_obj = TrackedObject(
                track_id=int(track_id),
                bbox=BoundingBox(
                    x1=float(bbox[0]),
                    y1=float(bbox[1]),
                    x2=float(bbox[2]),
                    y2=float(bbox[3])
                ),
                confidence=float(conf),
                class_id=int(cls_id),
                class_name=self.model.names[cls_id],
                history=list(self._track_history[track_id])
            )
            tracked_objects.append(tracked_obj)
        
        return tracked_objects
    
    def reset(self) -> None:
        """Reset tracker state for new video."""
        self._track_history.clear()
        # Reset the YOLO model's tracker
        self.model = YOLO(self.model.model_name if hasattr(self.model, 'model_name') else config.DETECTION_MODEL)
    
    def get_track_history(self, track_id: int) -> List[tuple]:
        """
        Get position history for a specific track.
        
        Args:
            track_id: Track ID to retrieve history for
        
        Returns:
            List of (x, y) center positions
        """
        return self._track_history.get(track_id, [])