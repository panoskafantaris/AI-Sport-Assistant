"""
Configuration settings for basketball tracker.
"""
from pathlib import Path


# Project paths
PROJECT_ROOT = Path(__file__).parent
SAMPLES_DIR = PROJECT_ROOT / "samples"
RESULTS_DIR = PROJECT_ROOT / "results"

# Ensure directories exist
SAMPLES_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)


# Detection settings
DETECTION_MODEL = "yolov8n.pt"  # nano model, fast. Options: yolov8n/s/m/l/x
DETECTION_CONFIDENCE = 0.5
DETECTION_IOU_THRESHOLD = 0.45
PERSON_CLASS_ID = 0  # COCO class ID for 'person'


# Tracking settings
TRACKER_TYPE = "bytetrack"  # Options: bytetrack, botsort
TRACK_HIGH_THRESH = 0.5
TRACK_LOW_THRESH = 0.1
TRACK_BUFFER = 30  # Frames to keep lost tracks


# Video processing settings
DEFAULT_FRAME_SKIP = 0  # Process every frame (0 = no skip)
OUTPUT_VIDEO_CODEC = "mp4v"  # MPEG-4 codec (most compatible)
OUTPUT_VIDEO_FPS = None  # None = same as input


# Visualization settings
BOX_THICKNESS = 2
FONT_SCALE = 0.6
FONT_THICKNESS = 2


# Download settings
DEFAULT_SAMPLE_URL = "https://www.youtube.com/watch?v=6OBFpgXJLXk"  # Basketball highlights
DOWNLOAD_FORMAT = "mp4"
DOWNLOAD_RESOLUTION = "720"


# Team classification settings
CALIBRATION_FRAMES = 30              # Frames to sample for initial calibration
JERSEY_CROP_TOP_RATIO = 0.15         # Skip head (15% from top of bbox)
JERSEY_CROP_BOTTOM_RATIO = 0.55      # Skip legs (stop at 55% from top)
JERSEY_CROP_SIDE_RATIO = 0.15        # Crop sides to avoid arms

# Color extraction settings
DOMINANT_COLOR_CLUSTERS = 3          # K-means clusters for dominant color
COLOR_SPACE = "HSV"                  # HSV is more robust to lighting

# Team clustering settings
NUM_TEAMS = 2                        # Number of teams to detect
REFEREE_DISTANCE_THRESHOLD = 60      # HSV distance to be considered referee
MIN_SAMPLES_FOR_CLUSTERING = 6       # Minimum players needed to cluster

# Scene change detection
SCENE_CHANGE_THRESHOLD = 0.4         # Histogram difference threshold (0-1)
SCENE_CHANGE_COOLDOWN = 15           # Frames to wait after scene change