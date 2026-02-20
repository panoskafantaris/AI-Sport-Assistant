"""Court Detection V3 — calibration-seeded + multi-modal alignment."""
from .court_detector_v3 import CourtDetectorV3
from .calibrator import (
    CourtCalibrator, save_calibration, load_calibration)
from .detection_masks import DetectionMasks
from . import template

__all__ = [
    "CourtDetectorV3",
    "CourtCalibrator", "save_calibration", "load_calibration",
    "DetectionMasks", "template",
]