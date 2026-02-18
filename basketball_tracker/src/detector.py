"""
YOLO-based object detection for person detection.
"""
import numpy as np
from typing import List, Optional
from ultralytics import YOLO

from .models import Detection, BoundingBox
import config


class Detector:
    """YOLOv8-based person detector."""
    
    def __init__(
        self,
        model_name: str = config.DETECTION_MODEL,
        confidence_threshold: float = config.DETECTION_CONFIDENCE,
        iou_threshold: float = config.DETECTION_IOU_THRESHOLD,
        device: Optional[str] = None
    ):
        """
        Initialize detector.
        
        Args:
            model_name: YOLO model name (e.g., 'yolov8n.pt')
            confidence_threshold: Minimum confidence for detections
            iou_threshold: IOU threshold for NMS
            device: Device to run on ('cpu', 'cuda', or None for auto)
        """
        self.model = YOLO(model_name)
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        self.target_class_id = config.PERSON_CLASS_ID
    
    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Detect persons in a frame.
        
        Args:
            frame: BGR image array
        
        Returns:
            List of Detection objects for persons only
        """
        results = self.model(
            frame,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False,
            classes=[self.target_class_id]
        )
        
        return self._parse_results(results[0])
    
    def _parse_results(self, result) -> List[Detection]:
        """
        Parse YOLO results into Detection objects.
        
        Args:
            result: Single YOLO result object
        
        Returns:
            List of Detection objects
        """
        detections = []
        
        if result.boxes is None or len(result.boxes) == 0:
            return detections
        
        boxes = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy().astype(int)
        
        for bbox, conf, cls_id in zip(boxes, confidences, class_ids):
            detection = Detection(
                bbox=BoundingBox(
                    x1=float(bbox[0]),
                    y1=float(bbox[1]),
                    x2=float(bbox[2]),
                    y2=float(bbox[3])
                ),
                confidence=float(conf),
                class_id=int(cls_id),
                class_name=self.model.names[cls_id]
            )
            detections.append(detection)
        
        return detections