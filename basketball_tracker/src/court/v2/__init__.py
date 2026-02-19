"""Court Detection V2 — spatial ROI + net-aware line classification."""
from .court_detector_v2 import CourtDetectorV2, AUTO_THRESHOLD
from .line_filter       import LineDetector, DetectedLine, cluster_lines
from .line_classifier   import classify_lines, ClassifiedLines
from .net_detector      import detect_net, score_net_likelihood
from .intersection      import (generate_candidates,
                                corners_from_classified,
                                corners_from_baselines)
from .model_fitter      import ModelFitter
from .refiner           import HomographyRefiner
from .baseline_refiner  import refine_baseline
from . import template

__all__ = [
    "CourtDetectorV2", "AUTO_THRESHOLD",
    "LineDetector", "DetectedLine", "cluster_lines",
    "classify_lines", "ClassifiedLines",
    "detect_net", "score_net_likelihood",
    "generate_candidates",
    "ModelFitter", "HomographyRefiner", "template",
]