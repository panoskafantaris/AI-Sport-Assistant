"""
Tennis Tracker – source package.

Public API:  all major components are importable directly from `src`.

    from src import Pipeline
    from src import CourtLineDetector, CourtCalibrator, HomographyCalc
    from src import PlayerTracker
    from src import BallDetector, TrajectoryTracker, LandingPredictor
    from src import MovementAnalyser, BallSpeedEstimator, RallyDetector, KinesiologyAnalyser
    from src import Visualizer, Exporter
    from src.models import CourtBoundary, Player, Ball, FrameData, TrackingResult
"""

# ── Pipeline (top-level entry point) ─────────────────────────────────────────
from .pipeline import Pipeline

# ── Phase 1: Court ────────────────────────────────────────────────────────────
from .court.detector   import CourtLineDetector
from .court.calibrator import CourtCalibrator
from .court.homography import HomographyCalc

# ── Phase 2: Players ──────────────────────────────────────────────────────────
from .players.tracker  import PlayerTracker
from .players.detector import PlayerDetector

# ── Phase 3: Ball ─────────────────────────────────────────────────────────────
from .ball.detector    import BallDetector
from .ball.trajectory  import TrajectoryTracker, LandingPredictor

# ── Phase 4: Stats ────────────────────────────────────────────────────────────
from .stats.movement    import MovementAnalyser, HeatmapAccumulator
from .stats.rally       import BallSpeedEstimator, RallyDetector
from .stats.kinesiology import KinesiologyAnalyser

# ── Utilities ─────────────────────────────────────────────────────────────────
from .visualizer import Visualizer
from .exporter   import Exporter
from .video.loader import VideoLoader

# ── Models (data classes) ─────────────────────────────────────────────────────
from .models import (
    CourtBoundary, CourtLine, CourtHomography,
    Player, PlayerRole, BoundingBox,
    Ball, BallTrajectory, LandingEstimate, BallStatus,
    FrameData, VideoMetadata, TrackingResult,
)

__all__ = [
    # Pipeline
    "Pipeline",
    # Court
    "CourtLineDetector", "CourtCalibrator", "HomographyCalc",
    # Players
    "PlayerTracker", "PlayerDetector",
    # Ball
    "BallDetector", "TrajectoryTracker", "LandingPredictor",
    # Stats
    "MovementAnalyser", "HeatmapAccumulator",
    "BallSpeedEstimator", "RallyDetector",
    "KinesiologyAnalyser",
    # Utilities
    "Visualizer", "Exporter", "VideoLoader",
    # Models
    "CourtBoundary", "CourtLine", "CourtHomography",
    "Player", "PlayerRole", "BoundingBox",
    "Ball", "BallTrajectory", "LandingEstimate", "BallStatus",
    "FrameData", "VideoMetadata", "TrackingResult",
]