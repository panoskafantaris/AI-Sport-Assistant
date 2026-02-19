"""
Phase 3 – Ball detector.

Two-stage approach:
  1. YOLO sports-ball class (fast, works well when ball is large enough).
  2. Circular Hough Transform fallback (works for small/blurred balls).

Both detections are merged and the most confident one is returned.
"""
from __future__ import annotations
from typing import List, Optional
import cv2
import numpy as np
from ultralytics import YOLO

from ..models.ball import Ball
import config


class BallDetector:
    """Detects the tennis ball in each frame."""

    def __init__(self):
        self._model = YOLO(config.BALL_MODEL)
        self._frame_count = 0

    # ── Public API ─────────────────────────────────────────────────────────────

    def detect(
        self,
        frame: np.ndarray,
        frame_number: int = 0,
        timestamp_ms: float = 0.0,
    ) -> Optional[Ball]:
        """
        Detect the ball in a single frame.

        Returns the best Ball detection or None.
        """
        self._frame_count = frame_number
        candidates: List[Ball] = []

        # --- YOLO detection ---
        yolo_ball = self._detect_yolo(frame, frame_number, timestamp_ms)
        if yolo_ball:
            candidates.append(yolo_ball)

        # --- Hough fallback (especially good for fast-moving ball) ---
        hough_ball = self._detect_hough(frame, frame_number, timestamp_ms)
        if hough_ball:
            candidates.append(hough_ball)

        if not candidates:
            return None

        # Return highest confidence
        return max(candidates, key=lambda b: b.confidence)

    # ── Internals ──────────────────────────────────────────────────────────────

    def _detect_yolo(
        self, frame: np.ndarray, fn: int, ts: float
    ) -> Optional[Ball]:
        results = self._model.predict(
            frame,
            conf=0.25,
            classes=[config.SPORTS_BALL_CLASS],
            imgsz=config.DETECTION_IMG_SIZE,
            verbose=False,
        )
        best: Optional[Ball] = None
        for res in results:
            for box in res.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                r  = ((x2 - x1) + (y2 - y1)) / 4
                conf = float(box.conf[0])

                if not (config.BALL_MIN_RADIUS_PX <= r <= config.BALL_MAX_RADIUS_PX):
                    continue
                if best is None or conf > best.confidence:
                    best = Ball(x=cx, y=cy, radius=r, confidence=conf,
                                frame_number=fn, timestamp_ms=ts)
        return best

    def _detect_hough(
        self, frame: np.ndarray, fn: int, ts: float
    ) -> Optional[Ball]:
        """
        Circular Hough on a pre-processed ROI.
        Works well for partially blurred/fast balls.
        """
        gray   = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur   = cv2.GaussianBlur(gray, (9, 9), 2)

        circles = cv2.HoughCircles(
            blur,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=30,
            param1=100,
            param2=20,
            minRadius=config.BALL_MIN_RADIUS_PX,
            maxRadius=config.BALL_MAX_RADIUS_PX,
        )
        if circles is None:
            return None

        circles = np.round(circles[0]).astype(int)
        # Pick the circle most likely to be the ball:
        # brightest interior (tennis balls are bright yellow/white)
        best_ball: Optional[Ball] = None
        best_brightness = -1.0

        for cx, cy, r in circles:
            mask = np.zeros(gray.shape, np.uint8)
            cv2.circle(mask, (cx, cy), r, 255, -1)
            mean_bright = float(cv2.mean(gray, mask=mask)[0])

            if mean_bright > best_brightness:
                best_brightness = mean_bright
                # Confidence proxy: brightness / 255
                conf = min(mean_bright / 255.0, 0.9)
                best_ball = Ball(
                    x=float(cx), y=float(cy), radius=float(r),
                    confidence=conf, frame_number=fn, timestamp_ms=ts,
                )

        return best_ball