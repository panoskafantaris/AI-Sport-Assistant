"""
Scene change detector.

Two complementary signals:
  1. Histogram difference   – catches hard cuts instantly (< 1 frame lag)
  2. SSIM drop              – catches soft cuts, zooms, dissolves

A scene change is declared when EITHER signal exceeds its threshold,
subject to a cooldown so a single cut doesn't fire multiple times.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import cv2
import numpy as np


@dataclass
class SceneChangeEvent:
    frame_number: int
    timestamp_ms: float
    hist_diff: float        # 0-1
    ssim_score: float       # 0-1  (1 = identical)
    trigger: str            # "histogram" | "ssim" | "both"


class SceneDetector:
    """
    Detects scene changes between consecutive frames.

    Args:
        hist_threshold:  Histogram difference [0-1] that triggers a cut.
                         0.35 works well for broadcast tennis; lower = more sensitive.
        ssim_threshold:  SSIM drop below this triggers a soft-cut.
                         0.55 catches zooms without too many false positives.
        cooldown_frames: Minimum frames between two consecutive scene events.
    """

    def __init__(
        self,
        hist_threshold:  float = 0.35,
        ssim_threshold:  float = 0.55,
        cooldown_frames: int   = 15,
    ):
        self.hist_threshold  = hist_threshold
        self.ssim_threshold  = ssim_threshold
        self.cooldown_frames = cooldown_frames

        self._prev_frame:  Optional[np.ndarray] = None
        self._prev_hist:   Optional[np.ndarray] = None
        self._last_event_frame: int = -cooldown_frames

    # ── Public API ─────────────────────────────────────────────────────────────

    def update(
        self, frame: np.ndarray, frame_number: int, timestamp_ms: float
    ) -> Optional[SceneChangeEvent]:
        """
        Feed the next frame. Returns a SceneChangeEvent if a cut is detected,
        otherwise None.
        """
        small = self._downscale(frame)
        hist  = self._compute_hist(small)

        event = None
        in_cooldown = (frame_number - self._last_event_frame) < self.cooldown_frames

        if self._prev_hist is not None and not in_cooldown:
            hist_diff = self._histogram_diff(self._prev_hist, hist)
            ssim      = self._ssim(self._prev_frame, small) if self._prev_frame is not None else 1.0

            h_triggered = hist_diff > self.hist_threshold
            s_triggered = ssim      < self.ssim_threshold

            if h_triggered or s_triggered:
                trigger = (
                    "both"      if h_triggered and s_triggered else
                    "histogram" if h_triggered else
                    "ssim"
                )
                event = SceneChangeEvent(
                    frame_number = frame_number,
                    timestamp_ms = timestamp_ms,
                    hist_diff    = round(hist_diff, 3),
                    ssim_score   = round(ssim, 3),
                    trigger      = trigger,
                )
                self._last_event_frame = frame_number

        self._prev_frame = small
        self._prev_hist  = hist
        return event

    def reset(self) -> None:
        self._prev_frame = None
        self._prev_hist  = None
        self._last_event_frame = -self.cooldown_frames

    # ── Internals ──────────────────────────────────────────────────────────────

    @staticmethod
    def _downscale(frame: np.ndarray, width: int = 320) -> np.ndarray:
        h, w = frame.shape[:2]
        scale = width / w
        return cv2.resize(frame, (width, int(h * scale)))

    @staticmethod
    def _compute_hist(frame: np.ndarray) -> np.ndarray:
        hsv   = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h_hist = cv2.calcHist([hsv], [0], None, [50], [0, 180])
        s_hist = cv2.calcHist([hsv], [1], None, [50], [0, 256])
        hist   = np.concatenate([h_hist, s_hist]).flatten()
        cv2.normalize(hist, hist)
        return hist

    @staticmethod
    def _histogram_diff(a: np.ndarray, b: np.ndarray) -> float:
        # Bhattacharyya distance: 0 = identical, 1 = completely different
        return float(cv2.compareHist(a, b, cv2.HISTCMP_BHATTACHARYYA))

    @staticmethod
    def _ssim(a: np.ndarray, b: np.ndarray) -> float:
        """Simplified SSIM on luminance channel."""
        if a.shape != b.shape:
            b = cv2.resize(b, (a.shape[1], a.shape[0]))
        ga = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY).astype(np.float32)
        gb = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY).astype(np.float32)
        C1, C2 = 6.5025, 58.5225
        mu_a  = cv2.GaussianBlur(ga, (11, 11), 1.5)
        mu_b  = cv2.GaussianBlur(gb, (11, 11), 1.5)
        mu_ab = mu_a * mu_b
        sig_a = cv2.GaussianBlur(ga * ga, (11, 11), 1.5) - mu_a ** 2
        sig_b = cv2.GaussianBlur(gb * gb, (11, 11), 1.5) - mu_b ** 2
        sig_ab= cv2.GaussianBlur(ga * gb, (11, 11), 1.5) - mu_ab
        num   = (2 * mu_ab + C1) * (2 * sig_ab + C2)
        den   = (mu_a**2 + mu_b**2 + C1) * (sig_a + sig_b + C2)
        return float(np.mean(num / (den + 1e-8)))