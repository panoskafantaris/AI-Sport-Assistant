"""
Test harness for Court Detection V2.

Usage (from basketball_tracker/ root):
    python scripts/test_court_v2.py -i samples/match.mp4 -f 0 70 210 1170
    python scripts/test_court_v2.py -i samples/match.mp4 -f 70 --show
    python scripts/test_court_v2.py -i samples/match.mp4 --every 200
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path

import cv2
import numpy as np

# ── Path setup (run from basketball_tracker/ root) ───────────────────────────
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.court.v2.court_detector_v2 import CourtDetectorV2
from src.court.v2 import template as T
from src.court.v2.line_classifier import ClassifiedLines

_ROLE_COLOURS = {
    "far_baseline":   (0, 0, 255),
    "near_baseline":  (0, 0, 255),
    "far_service":    (0, 180, 255),
    "near_service":   (0, 180, 255),
    "net":            (255, 0, 255),
    "left_sideline":  (255, 100, 0),
    "right_sideline": (255, 100, 0),
    "center_service": (200, 200, 0),
}


def get_frame(path, fn=0):
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, fn)
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None


def get_video_info(path):
    cap = cv2.VideoCapture(path)
    info = {"frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "fps": cap.get(cv2.CAP_PROP_FPS),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}
    cap.release()
    return info


def draw_result(frame, corners, conf, fn, cl=None):
    out = frame.copy()
    H, W = out.shape[:2]

    # Draw classified lines
    if cl:
        for role in ("far_baseline", "far_service", "net",
                      "near_service", "near_baseline",
                      "left_sideline", "right_sideline",
                      "center_service"):
            ln = getattr(cl, role)
            if ln is None:
                continue
            c = _ROLE_COLOURS.get(role, (200, 200, 200))
            t = 3 if role == "net" else 2
            cv2.line(out, (int(ln.x1), int(ln.y1)),
                     (int(ln.x2), int(ln.y2)), c, t)
            mx, my = ln.midpoint
            cv2.putText(out, role.replace("_", " "),
                        (int(mx) - 40, int(my) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, c, 1)

    # Draw boundary
    if corners is not None:
        pts = corners.reshape(-1, 1, 2).astype(np.int32)
        cv2.polylines(out, [pts], True, (0, 255, 255), 3)
        labels = ["TL", "TR", "BR", "BL"]
        colors = [(0, 255, 0), (0, 200, 255), (255, 100, 0), (255, 0, 200)]
        for i, (cx, cy) in enumerate(corners):
            cv2.circle(out, (int(cx), int(cy)), 10, colors[i], -1)
            cv2.circle(out, (int(cx), int(cy)), 10, (255, 255, 255), 2)
            cv2.putText(out, labels[i], (int(cx) + 14, int(cy) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[i], 2)
        _draw_template(out, corners)

    # Status bar
    cv2.rectangle(out, (0, 0), (W, 40), (20, 20, 20), -1)
    status = (f"DETECTED conf={conf:.3f}"
              if corners is not None else "FAILED")
    color = (0, 255, 0) if corners is not None else (0, 0, 255)
    cv2.putText(out, f"Frame {fn}  |  {status}",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return out


def _draw_template(frame, corners):
    H_mat, _ = cv2.findHomography(
        corners, T.BOUNDARY_CORNERS, cv2.RANSAC)
    if H_mat is None:
        return
    try:
        H_inv = np.linalg.inv(H_mat)
    except np.linalg.LinAlgError:
        return
    for line in T.COURT_LINES:
        pts_w = np.array(
            [[[line.x1, line.y1]], [[line.x2, line.y2]]],
            dtype=np.float32)
        pts_i = cv2.perspectiveTransform(pts_w, H_inv)
        p1 = tuple(pts_i[0, 0].astype(int))
        p2 = tuple(pts_i[1, 0].astype(int))
        is_bndry = ("baseline" in line.name
                     or "sideline" in line.name)
        is_net = "net" in line.name
        colour = ((255, 0, 255) if is_net else
                  (0, 255, 100) if is_bndry else (100, 255, 100))
        cv2.line(frame, p1, p2,
                 colour, 2 if (is_bndry or is_net) else 1)


def test_frame(frame, fn, detector, out_dir, show=False):
    print(f"\n{'='*60}")
    print(f"  Frame {fn}  ({frame.shape[1]}×{frame.shape[0]})")
    print(f"{'='*60}")

    t0 = time.perf_counter()
    corners, conf, surface, cl = detector.detect(frame, fn)
    ms = (time.perf_counter() - t0) * 1000

    surface_name = surface.get("surface", "unknown")
    print(f"  Surface    : {surface_name}")
    print(f"  Confidence : {conf:.4f}")
    print(f"  Time       : {ms:.0f} ms")

    if cl:
        print(f"  Lines:\n{cl.summary()}")

    if corners is not None:
        for i, (cx, cy) in enumerate(corners):
            print(f"  {['TL','TR','BR','BL'][i]}:"
                  f" ({cx:.1f}, {cy:.1f})")
    else:
        print("  → No boundary detected")

    annotated = draw_result(frame, corners, conf, fn, cl)
    out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_dir / f"frame_{fn:06d}_result.jpg"),
                annotated)

    if show:
        win = f"V2 — Frame {fn}"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, 1440, 810)
        cv2.imshow(win, annotated)
        cv2.waitKey(0)
        cv2.destroyWindow(win)

    return {"frame_number": fn, "corners": corners,
            "confidence": conf, "surface": surface_name,
            "time_ms": ms}


def main():
    p = argparse.ArgumentParser(
        description="Test Court Detection V2 on video frames")
    p.add_argument("--input", "-i", required=True)
    p.add_argument("--frames", "-f", type=int, nargs="+",
                   default=[0])
    p.add_argument("--every", "-e", type=int, default=0)
    p.add_argument("--show", action="store_true")
    p.add_argument("--no-refine", action="store_true")
    p.add_argument("--output", "-o",
                   default="test_court_v2_output")
    args = p.parse_args()

    out_dir = Path(args.output)

    if not Path(args.input).exists():
        print(f"[ERROR] Not found: {args.input}")
        sys.exit(1)

    detector = CourtDetectorV2(refine=not args.no_refine)

    is_img = Path(args.input).suffix.lower() in (
        ".png", ".jpg", ".jpeg")
    if is_img:
        frame = cv2.imread(args.input)
        results = [test_frame(frame, 0, detector, out_dir,
                              args.show)]
    else:
        info = get_video_info(args.input)
        print(f"\nVideo: {args.input}  "
              f"{info['width']}×{info['height']}  "
              f"{info['fps']:.1f}fps  {info['frames']}f")
        fns = (list(range(0, info["frames"], args.every))
               if args.every > 0 else args.frames)
        results = []
        for fn in fns:
            frame = get_frame(args.input, fn)
            if frame is not None:
                results.append(
                    test_frame(frame, fn, detector, out_dir,
                               args.show))

    # Summary
    print(f"\n{'='*60}")
    print(f"  SUMMARY — {len(results)} frame(s)")
    print(f"{'='*60}")
    for r in results:
        s = (f"conf={r['confidence']:.3f}"
             if r["corners"] is not None else "FAILED")
        print(f"  Frame {r['frame_number']:>6}"
              f"  {r['surface']:<8}  {s:<16}"
              f"  {r['time_ms']:.0f}ms")
    detected = sum(1 for r in results
                   if r["corners"] is not None)
    print(f"\n  Detected: {detected}/{len(results)}")
    if results:
        avg = sum(r['time_ms'] for r in results) / len(results)
        print(f"  Avg time: {avg:.0f} ms")
    print(f"  Output: {out_dir}/")


if __name__ == "__main__":
    main()