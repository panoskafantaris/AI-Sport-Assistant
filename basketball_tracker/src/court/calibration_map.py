"""
Calibration map.

Stores a mapping of  frame_number → CourtBoundary  so that during
video processing the correct court boundary is always looked up for
the current scene, even after multiple scene changes.

Boundaries are stored as JSON and can be reloaded for reprocessing.
"""
from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import numpy as np

from ..models.court import CourtBoundary
from .surface_classifier import SurfaceType


@dataclass
class SceneRecord:
    start_frame:  int
    surface:      str           # SurfaceType.value
    confidence:   float
    auto_detected: bool         # True = auto, False = manual calibration
    boundary:     Optional[CourtBoundary]
    is_court:     bool = True   # False = user marked as non-court (skip)


class CalibrationMap:
    """
    Ordered list of SceneRecords. At any frame, the active boundary is
    the one from the most recent scene that started at or before that frame.
    """

    def __init__(self):
        self._scenes: list[SceneRecord] = []

    # ── Write ──────────────────────────────────────────────────────────────────

    def add_scene(
        self,
        start_frame:   int,
        boundary:      Optional[CourtBoundary],
        surface:       SurfaceType = SurfaceType.UNKNOWN,
        confidence:    float = 0.0,
        auto_detected: bool  = False,
        is_court:      bool  = True,
    ) -> None:
        record = SceneRecord(
            start_frame   = start_frame,
            surface       = surface.value,
            confidence    = round(confidence, 3),
            auto_detected = auto_detected,
            boundary      = boundary,
            is_court      = is_court,
        )
        # Replace existing record for the same start frame if present
        for i, s in enumerate(self._scenes):
            if s.start_frame == start_frame:
                self._scenes[i] = record
                return
        self._scenes.append(record)
        self._scenes.sort(key=lambda s: s.start_frame)

    # ── Read ───────────────────────────────────────────────────────────────────

    def get_boundary(self, frame_number: int) -> Optional[CourtBoundary]:
        """Return the active CourtBoundary for the given frame, or None."""
        record = self._get_record(frame_number)
        if record is None or not record.is_court:
            return None
        return record.boundary

    def get_surface(self, frame_number: int) -> SurfaceType:
        record = self._get_record(frame_number)
        if record is None:
            return SurfaceType.UNKNOWN
        try:
            return SurfaceType(record.surface)
        except ValueError:
            return SurfaceType.UNKNOWN

    def is_court_scene(self, frame_number: int) -> bool:
        record = self._get_record(frame_number)
        return record is not None and record.is_court

    def _get_record(self, frame_number: int) -> Optional[SceneRecord]:
        active = None
        for scene in self._scenes:
            if scene.start_frame <= frame_number:
                active = scene
            else:
                break
        return active

    # ── Persistence ────────────────────────────────────────────────────────────

    def save(self, path: Path) -> None:
        data = {"scenes": [self._record_to_dict(s) for s in self._scenes]}
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"[CalibrationMap] Saved {len(self._scenes)} scenes → {path}")

    def load(self, path: Path) -> bool:
        if not path.exists():
            return False
        try:
            with open(path) as f:
                data = json.load(f)
            self._scenes = [self._dict_to_record(d) for d in data["scenes"]]
            self._scenes.sort(key=lambda s: s.start_frame)
            print(f"[CalibrationMap] Loaded {len(self._scenes)} scenes ← {path}")
            return True
        except Exception as e:
            print(f"[CalibrationMap] Load failed: {e}")
            return False

    def summary(self) -> str:
        lines = [f"CalibrationMap — {len(self._scenes)} scene(s):"]
        for s in self._scenes:
            tag = "AUTO" if s.auto_detected else "MANUAL"
            nc  = "" if s.is_court else " [NON-COURT]"
            lines.append(
                f"  frame {s.start_frame:>5}  {s.surface:<8}  "
                f"conf={s.confidence:.2f}  [{tag}]{nc}"
            )
        return "\n".join(lines)

    # ── Serialisation helpers ──────────────────────────────────────────────────

    @staticmethod
    def _record_to_dict(r: SceneRecord) -> dict:
        corners = None
        if r.boundary is not None and r.boundary.is_valid():
            corners = r.boundary.corners.tolist()
        return {
            "start_frame":   r.start_frame,
            "surface":       r.surface,
            "confidence":    r.confidence,
            "auto_detected": r.auto_detected,
            "is_court":      r.is_court,
            "corners":       corners,
        }

    @staticmethod
    def _dict_to_record(d: dict) -> SceneRecord:
        boundary = None
        if d.get("corners") is not None:
            corners  = np.array(d["corners"], dtype=np.float32)
            boundary = CourtBoundary(corners=corners, confidence=d.get("confidence", 0.0))
        return SceneRecord(
            start_frame   = d["start_frame"],
            surface       = d.get("surface", "unknown"),
            confidence    = d.get("confidence", 0.0),
            auto_detected = d.get("auto_detected", False),
            boundary      = boundary,
            is_court      = d.get("is_court", True),
        )