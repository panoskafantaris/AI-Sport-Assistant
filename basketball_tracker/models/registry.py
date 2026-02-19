"""
Model registry.

Single source of truth for every model the app uses.
Add new entries here; the downloader/validator read from this dict.
"""
from pathlib import Path
import config

# ── Registry ──────────────────────────────────────────────────────────────────
# Each entry:
#   url      – direct download URL (Ultralytics CDN or HuggingFace)
#   size_mb  – approximate size (for display only)
#   phase    – which pipeline phase uses it
#   required – if False the app still works without it (graceful fallback)

MODELS: dict = {

    # Phase 2 – Player detection & tracking
    "yolov8m.pt": {
        "url":      "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8m.pt",
        "size_mb":  52,
        "phase":    "Phase 2 – Players",
        "required": True,
    },

    # Phase 3 – Ball detection (YOLO fallback; Hough is code-only)
    "yolov8n.pt": {
        "url":      "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt",
        "size_mb":  6,
        "phase":    "Phase 3 – Ball (YOLO fallback)",
        "required": True,
    },

    # Phase 4 – Kinesiology / pose estimation (optional, --pose flag)
    "yolov8m-pose.pt": {
        "url":      "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8m-pose.pt",
        "size_mb":  52,
        "phase":    "Phase 4 – Kinesiology",
        "required": False,
    },
}


def model_path(name: str) -> Path:
    """Return the absolute path where a model should live on disk."""
    return config.MODELS_DIR / name


def is_downloaded(name: str) -> bool:
    """Check whether a model file exists and is non-empty."""
    p = model_path(name)
    return p.exists() and p.stat().st_size > 10_000   # >10 KB = real file


def required_models() -> list[str]:
    return [n for n, m in MODELS.items() if m["required"]]


def optional_models() -> list[str]:
    return [n for n, m in MODELS.items() if not m["required"]]


def missing_required() -> list[str]:
    return [n for n in required_models() if not is_downloaded(n)]