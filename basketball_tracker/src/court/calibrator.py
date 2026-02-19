"""
Court calibrator – interactive corner-click UI.

Fix 5 applied: corners are clicked in window-scaled space and
back-projected to original frame coordinates before being stored.
This ensures the boundary is pixel-accurate regardless of window size.

Two modes:
  INITIAL – called once at startup on the first frame.
  SCENE   – called mid-video when auto-detection confidence is too low.

Controls:
  Left-click        – place next corner (NL → NR → FR → FL)
  Right-click       – undo last corner
  ENTER / SPACE     – confirm (4 corners required)
  R                 – reset all corners
  S                 – skip this scene (mark as non-court)
  Q                 – quit entire pipeline
"""
from __future__ import annotations
from enum import Enum
from typing import Optional, Tuple
import cv2
import numpy as np

from ..models.court import CourtBoundary, CourtHomography
from .homography import HomographyCalc
import config


_LABELS = ["Near-Left", "Near-Right", "Far-Right", "Far-Left"]
_COLORS = [(0, 255, 0), (0, 200, 255), (255, 100, 0), (255, 0, 200)]

# Guide lines drawn while placing corners so the user can see alignment
_GUIDE_COLOR  = (255, 255, 0)   # yellow
_GUIDE_ALPHA  = 0.35


class CalibrationMode(Enum):
    INITIAL = "initial"
    SCENE   = "scene"


class CalibrationResult(Enum):
    CONFIRMED = "confirmed"
    SKIPPED   = "skipped"
    QUIT      = "quit"


class CourtCalibrator:
    """Interactive court corner selection with scale-corrected output."""

    def __init__(self, doubles: bool = False):
        self._doubles = doubles
        self._corners:      list[Tuple[int, int]] = []
        self._display:      Optional[np.ndarray]  = None
        self._scale:        float                 = 1.0
        self._orig_size:    Tuple[int, int]        = (0, 0)   # (W, H) original

    # ── Public API ─────────────────────────────────────────────────────────────

    def run(
        self,
        frame:      np.ndarray,
        mode:       CalibrationMode = CalibrationMode.INITIAL,
        auto_conf:  float = 0.0,
        surface:    str   = "",
    ) -> Tuple[CalibrationResult, Optional[CourtBoundary], Optional[CourtHomography]]:
        """
        Show calibration UI and block until user confirms, skips, or quits.
        Returns (CalibrationResult, CourtBoundary | None, CourtHomography | None).
        """
        h, w              = frame.shape[:2]
        self._orig_size   = (w, h)
        self._display     = self._fit_to_window(frame)   # sets self._scale
        self._corners     = []
        result            = CalibrationResult.QUIT

        cv2.namedWindow(config.CAL_WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(config.CAL_WINDOW_NAME, config.CAL_WIDTH, config.CAL_HEIGHT)
        cv2.setMouseCallback(config.CAL_WINDOW_NAME, self._on_mouse)

        while True:
            self._render(mode, auto_conf, surface)
            key = cv2.waitKey(30) & 0xFF

            if key == ord('q'):
                result = CalibrationResult.QUIT
                break
            elif key == ord('s'):
                result = CalibrationResult.SKIPPED
                break
            elif key == ord('r'):
                self._corners = []
            elif key in (13, 32) and len(self._corners) == 4:
                result = CalibrationResult.CONFIRMED
                break

        cv2.destroyWindow(config.CAL_WINDOW_NAME)

        if result != CalibrationResult.CONFIRMED:
            return result, None, None

        return result, *self._build_output()

    # ── Mouse handler ──────────────────────────────────────────────────────────

    def _on_mouse(self, event, x, y, *_):
        if event == cv2.EVENT_LBUTTONDOWN and len(self._corners) < 4:
            self._corners.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN and self._corners:
            self._corners.pop()

    # ── Rendering ─────────────────────────────────────────────────────────────

    def _render(self, mode: CalibrationMode, auto_conf: float, surface: str):
        display = self._display.copy()
        n       = len(self._corners)

        # Draw guide crosshair at mouse position via overlay (informational)
        if n < 4:
            self._draw_guide_lines(display, n)

        # Draw placed corners
        for i, (cx, cy) in enumerate(self._corners):
            cv2.circle(display, (cx, cy), 9, _COLORS[i], -1)
            cv2.circle(display, (cx, cy), 9, (255, 255, 255), 1)
            cv2.putText(display, _LABELS[i], (cx + 12, cy - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, _COLORS[i], 2)

        # Draw polygon preview when all 4 are placed
        if n == 4:
            pts = np.array(self._corners, np.int32)
            overlay = display.copy()
            cv2.fillPoly(overlay, [pts], (0, 255, 255))
            cv2.addWeighted(overlay, 0.15, display, 0.85, 0, display)
            cv2.polylines(display, [pts], True, (0, 255, 255), 2)

        # Header bar
        bar_h = 72
        cv2.rectangle(display, (0, 0), (display.shape[1], bar_h), (15, 15, 15), -1)

        if mode == CalibrationMode.SCENE:
            reason = f"Low auto-detection confidence ({auto_conf:.0%})"
            if surface and surface != "unknown":
                reason += f"  |  Surface: {surface.upper()}"
            cv2.putText(display, reason, (10, 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 200, 255), 1)

        next_label = _LABELS[n] if n < 4 else "Press ENTER to confirm"
        cv2.putText(display, f"[{n}/4]  {next_label}", (10, 44),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1)
        cv2.putText(display,
                    "Left-click=place  Right-click=undo  R=reset  S=skip  Q=quit",
                    (10, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (160, 160, 160), 1)

        cv2.imshow(config.CAL_WINDOW_NAME, display)

    @staticmethod
    def _draw_guide_lines(display: np.ndarray, n_placed: int):
        """Faint horizontal/vertical guide grid to help click accurately."""
        h, w = display.shape[:2]
        overlay = display.copy()
        # Three horizontal guides at 1/3, 1/2, 2/3 height
        for ratio in (0.33, 0.50, 0.67):
            y = int(h * ratio)
            cv2.line(overlay, (0, y), (w, y), _GUIDE_COLOR, 1)
        # Two vertical guides at 1/4 and 3/4 width
        for ratio in (0.25, 0.75):
            x = int(w * ratio)
            cv2.line(overlay, (x, 0), (x, h), _GUIDE_COLOR, 1)
        cv2.addWeighted(overlay, _GUIDE_ALPHA, display, 1 - _GUIDE_ALPHA, 0, display)

    # ── Scale helpers (Fix 5) ──────────────────────────────────────────────────

    def _fit_to_window(self, frame: np.ndarray) -> np.ndarray:
        """
        Resize frame to fit inside the calibration window.
        Stores the scale factor so corners can be back-projected.
        """
        h, w  = frame.shape[:2]
        scale = min(config.CAL_WIDTH / w, config.CAL_HEIGHT / h, 1.0)
        self._scale = scale
        if scale < 1.0:
            new_w = int(w * scale)
            new_h = int(h * scale)
            return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return frame.copy()

    def _corners_to_original(self) -> np.ndarray:
        """
        Fix 5: back-project window-space clicks → original frame coordinates.
        """
        corners = np.array(self._corners, dtype=np.float32)
        return corners / self._scale   # divide by scale to undo the resize

    # ── Build output ──────────────────────────────────────────────────────────

    def _build_output(
        self,
    ) -> Tuple[Optional[CourtBoundary], Optional[CourtHomography]]:
        if len(self._corners) != 4:
            return None, None

        # Fix 5: use back-projected (original frame) coordinates
        corners  = self._corners_to_original()
        boundary = CourtBoundary(corners=corners, confidence=1.0)
        calc     = HomographyCalc(doubles=self._doubles)
        homogr   = calc.compute(boundary)
        return boundary, homogr