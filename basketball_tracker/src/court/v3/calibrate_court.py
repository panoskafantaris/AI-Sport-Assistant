"""
Court calibration script — 4-corner click + paint snapping.

Usage:
  python scripts/calibrate_court.py -i video.mp4 [-f FRAME] [-o calibration.json]

Controls:
  Left click = place next corner (BL → BR → TR → TL)
  z          = undo last click
  ENTER      = compute & save
  ESC        = cancel
"""
import argparse, cv2, sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.court.v3.calibrator import (
    CourtCalibrator, save_calibration, load_calibration)
from src.court.v3.scoring import quick_score
from src.court.v3.detection_masks import DetectionMasks


def main():
    ap = argparse.ArgumentParser(description="Court calibration")
    ap.add_argument("-i", "--input", required=True, help="Video file")
    ap.add_argument("-f", "--frame", type=int, default=0)
    ap.add_argument("-o", "--output", default="court_calibration.json")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        print(f"Cannot open: {args.input}"); return

    cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print(f"Cannot read frame {args.frame}"); return

    print(f"Frame {args.frame}: {frame.shape[1]}x{frame.shape[0]}")
    print("Click 4 court corners (BL → BR → TR → TL), then ENTER.\n")

    cal = CourtCalibrator()
    corners = cal.calibrate(frame)
    if corners is None:
        print("Cancelled."); return

    # Score
    masks = DetectionMasks.from_frame(frame)
    fh, fw = frame.shape[:2]
    score = quick_score(corners, masks, fh, fw)

    print(f"\nCorners [BL, BR, TR, TL]:")
    for label, (cx, cy) in zip(["BL","BR","TR","TL"], corners):
        print(f"  {label}: ({cx:.1f}, {cy:.1f})")
    print(f"Alignment score: {score:.3f}")

    save_calibration(corners, args.output)
    print(f"\nUse with V3 detector:")
    print(f'  python scripts/test_court.py -i {args.input} '
          f'--calibration {args.output}')


if __name__ == "__main__":
    main()