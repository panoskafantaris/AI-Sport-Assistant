"""
Test court detection V3 on specific frames.

Usage:
  python scripts/test_court.py -i video.mp4 -f 0 70 210 1170
  python scripts/test_court.py -i video.mp4 -f 0 70 --calibration cal.json
  python scripts/test_court.py -i video.mp4 --calibrate -f 0 70 210
"""
import argparse
import cv2
import numpy as np
import sys
import os
import time
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.court.v3 import (
    CourtDetectorV3, CourtCalibrator,
    save_calibration, load_calibration)
from src.court.v3.debug_viz import draw_result
from src.court.v3.confidence import per_line_alignment


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True)
    ap.add_argument("-f", "--frames", type=int, nargs="+", default=[0])
    ap.add_argument("-o", "--outdir", default="debug_court_v3")
    ap.add_argument("--calibration", type=str, default=None,
                    help="Path to calibration JSON")
    ap.add_argument("--calibrate", action="store_true",
                    help="Run interactive calibration on first frame")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.input)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {w}x{h} {fps}fps {total} frames\n")

    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    # Load or run calibration
    cal_corners = None
    if args.calibration:
        cal_corners = load_calibration(args.calibration)
        if cal_corners is not None:
            print(f"Loaded calibration: {args.calibration}")
            _print_corners(cal_corners)
        else:
            print(f"WARNING: Could not load {args.calibration}")

    if args.calibrate:
        cal_corners = _run_calibration(
            cap, args.frames[0], args.input)

    # Create detector
    det = CourtDetectorV3(calibration_corners=cal_corners)

    for fn in args.frames:
        print(f"\n{'='*60}")
        print(f"  Frame {fn}")
        print(f"{'='*60}")

        cap.set(cv2.CAP_PROP_POS_FRAMES, fn)
        ret, frame = cap.read()
        if not ret:
            print(f"  Cannot read frame {fn}")
            continue

        t0 = time.time()
        corners, conf, info = det.detect(frame, fn)
        dt = int((time.time() - t0) * 1000)

        method = info.get("method", "none")
        seed = info.get("seed_score", 0.0)
        print(f"  Method     : {method}")
        print(f"  Seed score : {seed:.4f}")
        print(f"  Confidence : {conf:.3f}")
        print(f"  Time       : {dt} ms")

        if corners is not None:
            _print_corners(corners)
            scores = info.get("line_scores", {})
            _print_alignment(scores)
            vis = draw_result(
                frame, corners, conf, fn,
                h_lines=info.get("h_lines"),
                v_lines=info.get("v_lines"),
                line_scores=scores)
            out_path = Path(args.outdir) / f"frame_{fn:06d}_v3.jpg"
            cv2.imwrite(str(out_path), vis)
            print(f"  Saved -> {out_path}")
        else:
            print("  FAILED — no detection")

    cap.release()
    print(f"\nResults saved in {args.outdir}/")


def _run_calibration(cap, frame_num, video_path):
    """Run interactive calibration and save result."""
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    if not ret:
        print(f"Cannot read frame {frame_num}")
        return None

    print("=== Interactive Calibration ===")
    print("Click court keypoints, then press ENTER.\n")

    cal = CourtCalibrator()
    corners = cal.calibrate(frame)

    if corners is None:
        print("Calibration cancelled.")
        return None

    # Save next to video
    out_path = str(Path(video_path).with_suffix(".calibration.json"))
    save_calibration(corners, out_path)
    print(f"Saved: {out_path}")
    return corners


def _print_corners(corners):
    labels = ["BL", "BR", "TR", "TL"]
    for label, (cx, cy) in zip(labels, corners):
        print(f"    {label}: ({cx:.1f}, {cy:.1f})")


def _print_alignment(scores):
    print("  Line alignment:")
    order = ["near_baseline", "far_baseline", "left_sideline",
             "right_sideline", "center_service", "net"]
    for name in order:
        pct = scores.get(name, 0.0) * 100
        bar = "\u2588" * int(pct / 5)
        print(f"    {name:<16} {pct:3.0f}% {bar}")


if __name__ == "__main__":
    main()