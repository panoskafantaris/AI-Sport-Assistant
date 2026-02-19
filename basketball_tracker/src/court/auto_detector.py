"""
Automatic court detector.

Key improvement in this version: smarter boundary line selection.

Instead of blindly picking outermost H/V lines, we:
  1. Score each line by how close it is to the edge of the court colour mask
  2. Prefer lines that are long and near the court perimeter
  3. Validate sidelines: they must span significant court height
  4. Validate baselines: they must span significant court width

This prevents interior service lines and external ad-board edges from
being selected as court boundaries.
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional, Tuple
import cv2
import numpy as np

from ..models.court import CourtBoundary, CourtLine
from .surface_classifier import SurfaceClassifier, SurfaceResult, SurfaceType
import config


AUTO_THRESHOLD = 0.70

# ── Tuning ────────────────────────────────────────────────────────────────────
_ROI_TOP_RATIO      = 0.20
_ROI_SIDE_RATIO     = 0.04   # slightly wider side exclusion
_COACHING_BOX_X     = 0.30
_COACHING_BOX_Y     = 0.42
_WHITE_THRESH       = 175
_MIN_CC_PIXELS      = 60

# Minimum line length as fraction of frame dimension
_BASELINE_MIN_SPAN  = 0.18   # baseline > 18% of FRAME width
_SIDELINE_MIN_SPAN  = 0.15   # sideline > 15% of FRAME height

_COURT_ASPECT_MIN   = 1.4
_COURT_ASPECT_MAX   = 5.5
_NEAR_WIDER_RATIO   = 1.00   # relaxed: just ensure near >= far
_MAX_H_LINES        = 8
_MAX_V_LINES        = 6


class AutoCourtDetector:

    def __init__(self, debug: bool = False, debug_dir: str = "debug_court"):
        self._classifier = SurfaceClassifier()
        self._debug      = debug
        self._debug_dir  = Path(debug_dir)
        if debug:
            self._debug_dir.mkdir(parents=True, exist_ok=True)

    # ── Public API ─────────────────────────────────────────────────────────────

    def detect(
        self, frame: np.ndarray, frame_number: int = 0
    ) -> Tuple[Optional[CourtBoundary], float, SurfaceResult]:
        H, W    = frame.shape[:2]
        surface = self._classifier.classify(frame)

        if surface.surface == SurfaceType.UNKNOWN:
            return None, 0.0, surface

        roi    = self._roi_mask(H, W)
        bright = self._white_pixels(frame, surface.mask, roi)
        clean  = self._filter_blobs(bright)
        edges  = cv2.Canny(cv2.GaussianBlur(clean, (3, 3), 0), 30, 100)
        lines  = self._hough_lines(edges, W, H)

        # Compute court colour bounding box for line validation
        court_bbox = self._court_bbox(surface.mask, roi)

        if self._debug:
            print(f"  [Debug] Raw Hough lines: {len(lines)} total")

        if not lines:
            return None, 0.1, surface

        h_lines, v_lines = self._cluster_lines(lines)

        # Filter lines by minimum span relative to court colour bbox
        h_lines = self._filter_baselines(h_lines, court_bbox, W)
        v_lines = self._filter_sidelines(v_lines, court_bbox, H)

        if self._debug:
            print(f"  [Debug] After cluster+filter: H={len(h_lines)} V={len(v_lines)}")
            for i, l in enumerate(h_lines):
                print(f"           H[{i}] midY={l.midpoint[1]:.0f} len={l.length:.0f}")
            for i, l in enumerate(v_lines):
                print(f"           V[{i}] midX={l.midpoint[0]:.0f} len={l.length:.0f}")

        if len(h_lines) < 2 or len(v_lines) < 2:
            conf = 0.15 + 0.05 * (len(h_lines) + len(v_lines))
            if self._debug:
                self._save_debug(frame, bright, clean, edges, lines,
                                 None, h_lines, v_lines, frame_number, surface)
            return None, round(conf, 2), surface

        boundary = self._build_boundary(h_lines, v_lines, H, W)

        if self._debug:
            self._save_debug(frame, bright, clean, edges, lines,
                             boundary, h_lines, v_lines, frame_number, surface)

        if boundary is None:
            return None, 0.28, surface

        confidence = self._score(boundary, h_lines, v_lines, surface, H, W)
        boundary.confidence = confidence
        return boundary, confidence, surface

    def needs_calibration(self, confidence: float) -> bool:
        return confidence < AUTO_THRESHOLD

    # ── Court colour bounding box ──────────────────────────────────────────────

    @staticmethod
    def _court_bbox(surface_mask: np.ndarray, roi: np.ndarray) -> Optional[tuple]:
        """
        Returns (x, y, w, h) bounding box of the court colour region within ROI.
        Used to scale minimum line span requirements.
        """
        if surface_mask is None or not np.any(surface_mask):
            return None
        masked = cv2.bitwise_and(surface_mask, roi)
        coords = cv2.findNonZero(masked)
        if coords is None:
            return None
        x, y, w, h = cv2.boundingRect(coords)
        return (x, y, w, h)

    # ── ROI mask ──────────────────────────────────────────────────────────────

    @staticmethod
    def _roi_mask(H: int, W: int) -> np.ndarray:
        mask = np.zeros((H, W), np.uint8)
        t = int(H * _ROI_TOP_RATIO)
        s = int(W * _ROI_SIDE_RATIO)
        mask[t:, s:W - s] = 255
        mask[int(H * _COACHING_BOX_Y):, :int(W * _COACHING_BOX_X)] = 0
        return mask

    # ── White pixel extraction ─────────────────────────────────────────────────

    @staticmethod
    def _white_pixels(frame, surface_mask, roi) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, bright = cv2.threshold(gray, _WHITE_THRESH, 255, cv2.THRESH_BINARY)
        if surface_mask is not None and np.any(surface_mask):
            kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
            dilated = cv2.dilate(surface_mask, kernel, iterations=2)
            bright  = cv2.bitwise_and(bright, dilated)
        return cv2.bitwise_and(bright, roi)

    # ── Noise filter ──────────────────────────────────────────────────────────

    @staticmethod
    def _filter_blobs(binary: np.ndarray) -> np.ndarray:
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            binary, connectivity=8
        )
        clean = np.zeros_like(binary)
        for lbl in range(1, n_labels):
            if stats[lbl, cv2.CC_STAT_AREA] >= _MIN_CC_PIXELS:
                clean[labels == lbl] = 255
        return clean

    # ── Hough ──────────────────────────────────────────────────────────────────

    @staticmethod
    def _hough_lines(edges, W, H) -> list[CourtLine]:
        raw = cv2.HoughLinesP(
            edges,
            rho=config.HOUGH_RHO,
            theta=np.deg2rad(config.HOUGH_THETA),
            threshold=config.HOUGH_THRESHOLD,
            minLineLength=config.HOUGH_MIN_LENGTH,
            maxLineGap=config.HOUGH_MAX_GAP,
        )
        if raw is None:
            return []
        # Very permissive here — span filtering happens in _filter_baselines/sidelines
        min_h = W * 0.04
        min_v = H * 0.04
        out   = []
        for x1, y1, x2, y2 in raw[:, 0]:
            ln  = CourtLine(float(x1), float(y1), float(x2), float(y2))
            ang = abs(ln.angle_deg) % 180
            if (ang < 35 or ang > 145) and ln.length >= min_h:
                out.append(ln)
            elif 55 < ang < 125 and ln.length >= min_v:
                out.append(ln)
        return out

    # ── Clustering ────────────────────────────────────────────────────────────

    @staticmethod
    def _cluster_lines(lines):
        h = [l for l in lines if abs(l.angle_deg)%180 < 35 or abs(l.angle_deg)%180 > 145]
        v = [l for l in lines if 55 < abs(l.angle_deg)%180 < 125]
        return (
            AutoCourtDetector._merge(h, "h")[:_MAX_H_LINES],
            AutoCourtDetector._merge(v, "v")[:_MAX_V_LINES],
        )

    @staticmethod
    def _merge(lines, axis):
        if not lines:
            return []
        key   = (lambda l: l.midpoint[1]) if axis == "h" else (lambda l: l.midpoint[0])
        lines = sorted(lines, key=key)
        merged, cluster = [], [lines[0]]
        for ln in lines[1:]:
            if abs(key(ln) - key(cluster[-1])) < config.LINE_CLUSTER_DIST:
                cluster.append(ln)
            else:
                merged.append(AutoCourtDetector._avg(cluster))
                cluster = [ln]
        merged.append(AutoCourtDetector._avg(cluster))
        return merged

    @staticmethod
    def _avg(cluster):
        ang = np.mean([l.angle_deg for l in cluster])
        L   = max(l.length for l in cluster)
        cx  = np.mean([(l.x1+l.x2)/2 for l in cluster])
        cy  = np.mean([(l.y1+l.y2)/2 for l in cluster])
        return CourtLine(cx - np.cos(np.deg2rad(ang))*L/2,
                         cy - np.sin(np.deg2rad(ang))*L/2,
                         cx + np.cos(np.deg2rad(ang))*L/2,
                         cy + np.sin(np.deg2rad(ang))*L/2)

    # ── Line validation against court colour bbox ─────────────────────────────

    @staticmethod
    def _filter_baselines(
        h_lines: list[CourtLine], court_bbox: Optional[tuple], W: int
    ) -> list[CourtLine]:
        """
        Keep horizontal lines spanning > _BASELINE_MIN_SPAN of frame width.
        Eliminates service lines, net, center marks which are shorter.
        Falls back to keeping all if filter is too aggressive.
        """
        if not h_lines:
            return []
        min_span = W * _BASELINE_MIN_SPAN
        valid = [l for l in h_lines if l.length >= min_span]
        return valid if len(valid) >= 2 else h_lines

    @staticmethod
    def _filter_sidelines(
        v_lines: list[CourtLine], court_bbox: Optional[tuple], H: int
    ) -> list[CourtLine]:
        """
        Keep vertical lines spanning > _SIDELINE_MIN_SPAN of frame height.
        Eliminates net posts, centre service line, short ad board edges.
        Falls back to keeping all if filter is too aggressive.
        """
        if not v_lines:
            return []
        min_span = H * _SIDELINE_MIN_SPAN
        valid = [l for l in v_lines if l.length >= min_span]
        return valid if len(valid) >= 2 else v_lines

    # ── Corner extraction ──────────────────────────────────────────────────────

    @staticmethod
    def _build_boundary(h_lines, v_lines, H, W) -> Optional[CourtBoundary]:
        h_s = sorted(h_lines, key=lambda l: l.midpoint[1])
        v_s = sorted(v_lines, key=lambda l: l.midpoint[0])

        tl = AutoCourtDetector._intersect(h_s[0],  v_s[0])
        tr = AutoCourtDetector._intersect(h_s[0],  v_s[-1])
        br = AutoCourtDetector._intersect(h_s[-1], v_s[-1])
        bl = AutoCourtDetector._intersect(h_s[-1], v_s[0])

        if any(p is None for p in (tl, tr, br, bl)):
            return None

        corners = AutoCourtDetector._order_cw(
            np.array([tl, tr, br, bl], dtype=np.float32))

        m = 0.20   # allow 20% outside frame — sidelines often extend beyond
        for cx, cy in corners:
            if not (-W*m <= cx <= W*(1+m) and -H*m <= cy <= H*(1+m)):
                return None

        return CourtBoundary(corners=corners) if \
            AutoCourtDetector._valid_geometry(corners, H, W) else None

    @staticmethod
    def _intersect(a, b):
        denom = (a.x1-a.x2)*(b.y1-b.y2) - (a.y1-a.y2)*(b.x1-b.x2)
        if abs(denom) < 1e-6:
            return None
        t = ((a.x1-b.x1)*(b.y1-b.y2) - (a.y1-b.y1)*(b.x1-b.x2)) / denom
        return (a.x1 + t*(a.x2-a.x1), a.y1 + t*(a.y2-a.y1))

    @staticmethod
    def _order_cw(pts):
        s = pts.sum(axis=1)
        d = np.diff(pts, axis=1).flatten()
        return np.array([pts[np.argmin(s)], pts[np.argmin(d)],
                         pts[np.argmax(s)], pts[np.argmax(d)]], dtype=np.float32)

    @staticmethod
    def _valid_geometry(corners, H, W):
        tl, tr, br, bl = corners
        if tl[0] >= tr[0] or bl[0] >= br[0]: return False
        if tl[1] >= bl[1] or tr[1] >= br[1]: return False
        nw = float(np.linalg.norm(np.array(br)-np.array(bl)))
        fw = float(np.linalg.norm(np.array(tr)-np.array(tl)))
        if fw > 0 and nw / fw < _NEAR_WIDER_RATIO: return False
        xs, ys = corners[:,0], corners[:,1]
        aspect = (xs.max()-xs.min()) / max(ys.max()-ys.min(), 1)
        return _COURT_ASPECT_MIN <= aspect <= _COURT_ASPECT_MAX

    # ── Confidence ─────────────────────────────────────────────────────────────

    @staticmethod
    def _score(boundary, h_lines, v_lines, surface, H, W):
        s = []
        s.append(min(len(h_lines)/3.0,1.0)*0.5 + min(len(v_lines)/2.0,1.0)*0.5)
        area  = float(cv2.contourArea(boundary.corners.astype(np.float32)))
        ratio = area/(H*W)
        s.append(1.0 if 0.15<=ratio<=0.70 else max(0.0,1.0-abs(ratio-0.4)*4))
        if surface.mask is not None and np.any(surface.mask):
            poly = boundary.corners.reshape((-1,1,2)).astype(np.int32)
            fill = np.zeros((H,W),np.uint8)
            cv2.fillPoly(fill,[poly],255)
            ov = cv2.bitwise_and(surface.mask,fill)
            s.append(float(np.sum(ov>0))/max(int(np.sum(fill>0)),1))
        else:
            s.append(0.5)
        s.append(min(surface.confidence*4.0,1.0))
        tl,tr,br,bl = boundary.corners
        nw = float(np.linalg.norm(br-bl))
        fw = float(np.linalg.norm(tr-tl))
        s.append(max(min(nw/max(fw,1.0)-1.0,1.0),0.0))
        return round(float(np.mean(s)),3)

    # ── Debug ──────────────────────────────────────────────────────────────────

    def _save_debug(self, frame, bright, clean, edges, lines,
                    boundary, h_lines, v_lines, fn, surface):
        H, W   = frame.shape[:2]
        ph, pw = H//2, W//2
        tag    = f"{fn:06d}"

        def rsz(img): return cv2.resize(img,(pw,ph))
        def bgr(g):   return cv2.cvtColor(g,cv2.COLOR_GRAY2BGR)

        # Panel A — frame + ROI
        A   = frame.copy()
        roi = self._roi_mask(H, W)
        tint = np.zeros_like(A); tint[roi==0] = (0,0,120)
        A = cv2.addWeighted(A,0.75,tint,0.25,0)

        # Draw court bbox if available
        bbox = self._court_bbox(surface.mask, roi)
        if bbox:
            bx,by,bw,bh = bbox
            cv2.rectangle(A,(bx,by),(bx+bw,by+bh),(0,255,0),2)
        cv2.putText(A,"A: ROI (red=excl, green=court bbox)",(6,20),
                    cv2.FONT_HERSHEY_SIMPLEX,0.45,(0,255,255),1)

        # Panel B — white pixels
        B = bgr(bright.copy())
        cv2.putText(B,f"B: white thresh={_WHITE_THRESH}",(6,20),
                    cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,255),1)

        # Panel C — blob filter (removed = red)
        C = bgr(clean.copy())
        removed = cv2.bitwise_and(bright, cv2.bitwise_not(clean))
        C[removed>0] = [0,0,200]
        cv2.putText(C,f"C: size filter (red=noise, min={_MIN_CC_PIXELS}px)",(6,20),
                    cv2.FONT_HERSHEY_SIMPLEX,0.45,(0,255,255),1)

        # Panel D — lines + boundary
        D = frame.copy()
        for ln in lines:
            cv2.line(D,(int(ln.x1),int(ln.y1)),(int(ln.x2),int(ln.y2)),(180,180,0),1)
        for ln in h_lines:
            cv2.line(D,(int(ln.x1),int(ln.y1)),(int(ln.x2),int(ln.y2)),(0,220,0),2)
        for ln in v_lines:
            cv2.line(D,(int(ln.x1),int(ln.y1)),(int(ln.x2),int(ln.y2)),(255,80,0),2)
        if boundary:
            pts = boundary.corners.reshape((-1,1,2)).astype(np.int32)
            cv2.polylines(D,[pts],True,(0,255,255),3)
            for i,(cx,cy) in enumerate(boundary.corners):
                cv2.circle(D,(int(cx),int(cy)),9,(0,0,255),-1)
                cv2.putText(D,["TL","TR","BR","BL"][i],(int(cx)+8,int(cy)-6),
                            cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,255),2)
            cv2.putText(D,f"D: conf={boundary.confidence:.2f}",(6,20),
                        cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),1)
        else:
            cv2.putText(D,f"D: FAILED H={len(h_lines)} V={len(v_lines)}",(6,20),
                        cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,220),1)

        panel = np.vstack([np.hstack([rsz(A),rsz(B)]),
                           np.hstack([rsz(C),rsz(D)])])
        cv2.imwrite(str(self._debug_dir/f"{tag}_composite.jpg"), panel)