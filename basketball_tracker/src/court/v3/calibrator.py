"""
Interactive court calibrator — 4-corner click + corner detection snap.

User clicks 4 boundary corners (approximate is fine):
  BL (near-left) → BR (near-right) → TR (far-right) → TL (far-left)

After each click, the algorithm finds the nearest white-paint
corner using Shi-Tomasi corner detection on the local patch.
Yellow boundary lines show the result in real-time.

These 4 corners = court limits. Ball outside = out of bounds.
"""
from __future__ import annotations
from pathlib import Path
from typing import List, Optional, Tuple
import cv2, json
import numpy as np

_NAMES = ["BL (near-left)", "BR (near-right)",
          "TR (far-right)", "TL (far-left)"]
_KEYS = ["BL", "BR", "TR", "TL"]
_COLS = [(0,255,0), (0,255,0), (0,200,255), (0,200,255)]


class CourtCalibrator:
    """4-corner calibrator with corner-detection snapping."""

    def __init__(self, max_w: int = 1280, snap_r: int = 50):
        self._max_w, self._snap_r = max_w, snap_r
        self._scale = 1.0
        self._raw: List[Optional[Tuple[float,float]]] = [None]*4
        self._snap: List[Optional[Tuple[float,float]]] = [None]*4
        self._idx = 0
        self._frame = self._wmask = None

    def calibrate(self, frame: np.ndarray,
                  white_mask: np.ndarray = None,
                  win: str = "Court Calibrator") -> Optional[np.ndarray]:
        """Returns corners [BL,BR,TR,TL] (4,2) float32, or None."""
        self._frame = frame.copy()
        h, w = frame.shape[:2]
        self._scale = min(self._max_w / w, 1.0)
        self._raw = [None]*4; self._snap = [None]*4; self._idx = 0
        self._wmask = white_mask if white_mask is not None \
            else _white_mask(frame)

        cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(win, self._click)
        while True:
            cv2.imshow(win, self._draw())
            k = cv2.waitKey(30) & 0xFF
            if k == 27:
                cv2.destroyWindow(win); return None
            if k in (13, 10):
                c = self._corners()
                cv2.destroyWindow(win); return c
            if k == ord('z') and self._idx > 0:
                self._idx -= 1
                self._raw[self._idx] = self._snap[self._idx] = None

    def _click(self, ev, x, y, flags, param):
        if ev != cv2.EVENT_LBUTTONDOWN or self._idx >= 4:
            return
        ox, oy = x / self._scale, y / self._scale
        self._raw[self._idx] = (ox, oy)
        self._snap[self._idx] = _snap_to_corner(
            self._wmask, ox, oy, self._snap_r)
        self._idx = min(self._idx + 1, 4)

    def _pt(self, i):
        return self._snap[i] or self._raw[i]

    def _corners(self) -> Optional[np.ndarray]:
        pts = [self._pt(i) for i in range(4)]
        if any(p is None for p in pts): return None
        return np.array(pts, dtype=np.float32)

    def _draw(self) -> np.ndarray:
        h, w = self._frame.shape[:2]
        dw, dh = int(w*self._scale), int(h*self._scale)
        vis = cv2.resize(self._frame, (dw, dh))
        s = self._scale

        if self._idx < 4:
            _put(vis, f"Click {self._idx+1}/4: {_NAMES[self._idx]}",
                 (10,25), 0.55, (0,255,255))
        else:
            _put(vis, "4/4 done. ENTER=confirm, z=undo",
                 (10,25), 0.55, (0,255,0))
        _put(vis, "z=undo  ESC=cancel", (10,50), 0.4)

        # Draw corners: hollow circle = raw click, filled = snapped
        for i in range(4):
            if self._raw[i] is None: continue
            rx, ry = int(self._raw[i][0]*s), int(self._raw[i][1]*s)
            cv2.circle(vis, (rx,ry), 4, (0,0,200), 1)  # raw click
            p = self._pt(i)
            sx, sy = int(p[0]*s), int(p[1]*s)
            cv2.circle(vis, (sx,sy), 8, _COLS[i], -1)
            cv2.circle(vis, (sx,sy), 10, (255,255,255), 2)
            _put(vis, _KEYS[i], (sx+12, sy+5), 0.5, _COLS[i])

        # Yellow boundary
        if all(self._raw[i] is not None for i in range(4)):
            for i in range(4):
                p1, p2 = self._pt(i), self._pt((i+1)%4)
                a = (int(p1[0]*s), int(p1[1]*s))
                b = (int(p2[0]*s), int(p2[1]*s))
                cv2.line(vis, a, b, (0,255,255), 2)

        n = sum(1 for c in self._raw if c is not None)
        _put(vis, f"{n}/4", (10, dh-15), 0.5,
             (0,255,0) if n==4 else (0,0,255))
        return vis


def _snap_to_corner(mask, cx, cy, r=50):
    """
    Find nearest white-paint corner near (cx, cy) with sub-pixel accuracy.

    Phase 1: Shi-Tomasi finds strong corners in local patch.
    Phase 2: cornerSubPix refines to sub-pixel precision.

    This auto-corrects user clicks that are off by up to `r` pixels.
    """
    fh, fw = mask.shape[:2]
    x1, x2 = max(0, int(cx-r)), min(fw, int(cx+r))
    y1, y2 = max(0, int(cy-r)), min(fh, int(cy+r))
    if x2-x1 < 10 or y2-y1 < 10:
        return None

    patch = mask[y1:y2, x1:x2]

    # Dilate to connect nearby line fragments
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    patch_d = cv2.dilate(patch, k, iterations=1)

    # Phase 1: Shi-Tomasi corner detection
    corners = cv2.goodFeaturesToTrack(
        patch_d, maxCorners=10, qualityLevel=0.1,
        minDistance=8, blockSize=7)

    if corners is None or len(corners) == 0:
        return None

    # Pick closest to click
    best_idx, best_dist = 0, float('inf')
    for i, c in enumerate(corners):
        px = float(c[0, 0]) + x1
        py = float(c[0, 1]) + y1
        d = (px - cx)**2 + (py - cy)**2
        if d < best_dist:
            best_dist = d
            best_idx = i

    # Phase 2: sub-pixel refinement
    best_corner = corners[best_idx:best_idx+1].copy()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                30, 0.01)
    refined = cv2.cornerSubPix(
        patch_d, best_corner, (5, 5), (-1, -1), criteria)

    px = float(refined[0, 0, 0]) + x1
    py = float(refined[0, 0, 1]) + y1
    return (px, py)


def _white_mask(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, m = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    return cv2.morphologyEx(m, cv2.MORPH_OPEN, k)


def _put(img, text, pos, scale, color=(255, 255, 255)):
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX,
                scale, (0, 0, 0), 3)
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, 1)


def save_calibration(corners: np.ndarray, path: str):
    data = {"corners_bl_br_tr_tl": corners.tolist()}
    Path(path).write_text(json.dumps(data, indent=2))
    print(f"Calibration saved: {path}")


def load_calibration(path: str) -> Optional[np.ndarray]:
    p = Path(path)
    if not p.exists(): return None
    data = json.loads(p.read_text())
    return np.array(data["corners_bl_br_tr_tl"], dtype=np.float32)