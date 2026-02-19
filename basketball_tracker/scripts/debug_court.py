"""
Debug court detection on a single video frame.

Shows a 4-panel composite:
  A – frame with ROI exclusion zones highlighted
  B – white pixel threshold
  C – after connected-component noise filter (red = removed noise)
  D – detected lines and final court boundary

Usage:
    python scripts/debug_court.py -i samples/match.mp4 -f 70 --interactive
    python scripts/debug_court.py -i samples/match.mp4 --every 200
    python scripts/debug_court.py -i samples/match.mp4 -f 70 --thresh 165
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np

DEBUG_DIR = Path("debug_court")
DEBUG_DIR.mkdir(exist_ok=True)


# ── Fast frame seek ───────────────────────────────────────────────────────────

def get_frame(video_path: str, frame_number: int):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None


# ── Analysis ──────────────────────────────────────────────────────────────────

def analyse_frame(
    frame: np.ndarray,
    frame_number: int,
    white_thresh: int = 175,
    min_cc: int = 80,
) -> np.ndarray:
    """Run the full detection pipeline and return the composite debug image."""

    import config
    # Allow CLI overrides
    import src.court.auto_detector as _ad
    _ad._WHITE_THRESH  = white_thresh
    _ad._MIN_CC_PIXELS = min_cc

    from src.court.auto_detector import AutoCourtDetector

    H, W = frame.shape[:2]
    detector = AutoCourtDetector(debug=True, debug_dir=str(DEBUG_DIR))
    boundary, confidence, surface = detector.detect(frame, frame_number)

    print(f"\n  Frame      : {frame_number}")
    print(f"  Surface    : {surface.surface.value}  (conf={surface.confidence:.2f})")
    print(f"  Confidence : {confidence:.3f}")
    if boundary is not None:
        print("  Corners:")
        for label, (cx, cy) in zip(["TL","TR","BR","BL"], boundary.corners):
            print(f"    {label}: ({cx:.0f}, {cy:.0f})")
    else:
        print("  → No boundary detected")

    composite = str(DEBUG_DIR / f"{frame_number:06d}_composite.jpg")
    img = cv2.imread(composite)
    if img is None:
        print(f"  [Warning] Composite not saved by detector, building manually")
        img = frame  # fallback
    return img


# ── Interactive viewer ────────────────────────────────────────────────────────

def interactive_mode(video_path: str, frame_number: int,
                     white_thresh: int, min_cc: int):
    frame = get_frame(video_path, frame_number)
    if frame is None:
        print(f"[Error] Frame {frame_number} not found in {video_path}")
        sys.exit(1)

    panel = analyse_frame(frame, frame_number, white_thresh, min_cc)

    win = f"Court Debug — frame {frame_number} | any key=close"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 1440, 810)
    cv2.imshow(win, panel)
    print("\n  Window open — press any key to close")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Debug court detection on a video frame",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input",   "-i", required=True, help="Video path")
    p.add_argument("--frame",   "-f", type=int, default=0,
                   help="Frame number to analyse")
    p.add_argument("--every",   "-e", type=int, default=0,
                   help="Analyse every N frames (0=disabled)")
    p.add_argument("--interactive", action="store_true",
                   help="Show result in OpenCV window")
    p.add_argument("--thresh",  "-t", type=int, default=175,
                   help="White pixel threshold (lower=more pixels)")
    p.add_argument("--min-cc",  type=int, default=80,
                   help="Min connected component size to keep (higher=less noise)")
    return p.parse_args()


def main():
    args = parse_args()

    if args.every > 0:
        cap   = cv2.VideoCapture(args.input)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        frames = range(0, total, args.every)
        print(f"Analysing {len(list(frames))} frames from {args.input} …")
        for fn in frames:
            frame = get_frame(args.input, fn)
            if frame is not None:
                analyse_frame(frame, fn, args.thresh, args.min_cc)
        print(f"\nDone. Images saved to {DEBUG_DIR}/")
        return

    if args.interactive:
        interactive_mode(args.input, args.frame, args.thresh, args.min_cc)
    else:
        frame = get_frame(args.input, args.frame)
        if frame is None:
            print(f"[Error] Frame {args.frame} not found")
            sys.exit(1)
        analyse_frame(frame, args.frame, args.thresh, args.min_cc)
        print(f"\nDone. Images saved to {DEBUG_DIR}/")


if __name__ == "__main__":
    main()