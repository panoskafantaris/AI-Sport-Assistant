"""
Team classification module for basketball tracker.

Two-phase calibration:
1. Court boundary definition (click corners)
2. Team color calibration (3 samples per team)
"""
from .color_extractor import ColorExtractor
from .color_reference import ColorReferenceStore, TeamColorReference
from .scene_detector import SceneDetector
from .classifier import TeamClassifier
from .interactive_calibrator import InteractiveCalibrator
from .court_detector import CourtDetector

__all__ = [
    "ColorExtractor",
    "ColorReferenceStore",
    "TeamColorReference",
    "SceneDetector",
    "TeamClassifier",
    "InteractiveCalibrator",
    "CourtDetector",
]