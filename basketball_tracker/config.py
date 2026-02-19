"""
Configuration for Tennis Tracker.
"""
from pathlib import Path

# ── GPU / Device ──────────────────────────────────────────────────────────────
# Auto-selects CUDA if available, falls back to CPU.
# With an RTX 4050 6 GB you should see 5-10× speedup over CPU.
def _resolve_device() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"[Config] GPU detected: {name}  ({vram:.1f} GB VRAM) → using CUDA")
            return "cuda"
    except ImportError:
        pass
    print("[Config] No GPU / torch not found → using CPU")
    return "cpu"

DEVICE = _resolve_device()

# Image size tuned for RTX 4050 6 GB:
#   1280 saturates VRAM with pose ON; drop to 960 if you run out.
DETECTION_IMG_SIZE = 1280 if DEVICE == "cuda" else 960

# Half-precision (FP16) speeds up inference ~2× on Ampere/Ada GPUs
USE_HALF_PRECISION = DEVICE == "cuda"

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent
SAMPLES_DIR  = PROJECT_ROOT / "samples"
RESULTS_DIR  = PROJECT_ROOT / "results"
MODELS_DIR   = PROJECT_ROOT / "models"

for d in (SAMPLES_DIR, RESULTS_DIR, MODELS_DIR):
    d.mkdir(exist_ok=True)

# ── Detection models ──────────────────────────────────────────────────────────
PERSON_MODEL       = "yolov8m.pt"       # Person detection (medium = better accuracy)
BALL_MODEL         = "yolov8n.pt"       # Fallback; TrackNet preferred when available
PERSON_CLASS_ID    = 0
SPORTS_BALL_CLASS  = 32                 # COCO class for sports ball
DETECTION_CONF     = 0.35
DETECTION_IOU      = 0.4
# DETECTION_IMG_SIZE is set above based on DEVICE

# ── Tracking ──────────────────────────────────────────────────────────────────
TRACKER_TYPE       = "bytetrack"
TRACK_HIGH_THRESH  = 0.5
TRACK_LOW_THRESH   = 0.1
TRACK_BUFFER       = 60
TRACK_MATCH_THRESH = 0.8

# ── Court detection ───────────────────────────────────────────────────────────
# Hough line parameters
HOUGH_RHO          = 1
HOUGH_THETA        = 1            # degrees (converted to radians in code)
HOUGH_THRESHOLD    = 20           # very low: far baseline & sidelines have few edge pixels
HOUGH_MIN_LENGTH   = 20           # short: frame-relative filter in auto_detector handles real cutoff
HOUGH_MAX_GAP      = 60           # wide gap: far end lines are broken by court markings
LINE_CLUSTER_ANGLE = 5            # degrees tolerance for merging parallel lines
LINE_CLUSTER_DIST  = 30           # pixel tolerance — wider to handle 1080p footage

# Court homography: standard tennis court dimensions (meters)
COURT_LENGTH       = 23.77        # baseline to baseline
COURT_WIDTH_SINGLE = 8.23         # singles
COURT_WIDTH_DOUBLE = 10.97        # doubles
SERVICE_BOX_DEPTH  = 6.40
NET_OFFSET         = 11.885       # from baseline to net

# ── Player classification ─────────────────────────────────────────────────────
# Tennis singles: 2 players only. Doubles: 4.
MAX_PLAYERS_SINGLES   = 2
MAX_PLAYERS_DOUBLES   = 4
BALL_BOY_HEIGHT_RATIO = 0.75      # Ball boys tend to crouch/be smaller
UMPIRE_ZONE_TOP_RATIO = 0.1       # Umpire chair typically in top region of frame
COURT_MARGIN_PX       = 80        # Pixels outside court still counted as "on court"

# ── Ball detection ────────────────────────────────────────────────────────────
BALL_MAX_RADIUS_PX     = 18
BALL_MIN_RADIUS_PX     = 3
BALL_TRAJECTORY_WINDOW = 8        # frames to use for trajectory fitting
BALL_LANDING_HORIZON   = 15       # frames ahead to predict landing
BALL_OUT_MARGIN_PX     = 5        # pixel tolerance for in/out calls

# ── Statistics ────────────────────────────────────────────────────────────────
SPEED_SMOOTHING_WINDOW = 5        # frames to smooth speed readings
POSE_MODEL             = "yolov8m-pose.pt"
RALLY_GAP_FRAMES       = 30       # frames without ball = rally ended
PLAYER_SPEED_SCALE     = 1.0      # multiplier after homography conversion

# ── Video output ──────────────────────────────────────────────────────────────
OUTPUT_CODEC    = "mp4v"
OUTPUT_FPS      = None            # None = match input
DEFAULT_SKIP    = 0
BOX_THICKNESS   = 2
FONT_SCALE      = 0.55
FONT_THICKNESS  = 1

# ── Interactive calibration window ───────────────────────────────────────────
CAL_WINDOW_NAME = "Tennis Tracker - Court Calibration"
CAL_WIDTH       = 1280
CAL_HEIGHT      = 720