"""
Exporter – saves JSON tracking data and annotated video.
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Optional

import cv2

from .models.frame import TrackingResult, VideoMetadata
import config


class Exporter:
    def __init__(self, output_dir: str = str(config.RESULTS_DIR)):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ── JSON ──────────────────────────────────────────────────────────────────

    def export_json(self, result: TrackingResult, filename: str = "tennis_tracking.json") -> Path:
        out = self.output_dir / filename
        with open(out, "w") as f:
            json.dump(result.to_dict(), f, indent=2)
        print(f"[Exporter] JSON saved → {out}")
        return out

    # ── Video writer ──────────────────────────────────────────────────────────

    def create_video_writer(
        self,
        meta: VideoMetadata,
        filename: str = "tennis_annotated.mp4",
        fps: Optional[float] = None,
    ) -> cv2.VideoWriter:
        out = self.output_dir / filename
        output_fps = fps or meta.fps

        for codec in (config.OUTPUT_CODEC, "XVID", "MJPG"):
            fourcc = cv2.VideoWriter_fourcc(*codec)
            writer = cv2.VideoWriter(
                str(out), fourcc, output_fps, (meta.width, meta.height)
            )
            if writer.isOpened():
                print(f"[Exporter] Video writer → {out}  (codec={codec})")
                return writer
            writer.release()

        raise RuntimeError(f"Failed to create video writer for {out}")