"""
CLI entry point for Tennis Tracker.

Example usage:
    python main.py -i samples/match.mp4 --interactive
    python main.py -i samples/match.mp4 --no-court-cal --skip 1
    python main.py -i samples/match.mp4 --doubles --pose --max-frames 500
"""
import argparse
import sys
from pathlib import Path

from src.pipeline import Pipeline
import config


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Tennis match analysis pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input",  "-i", required=True,   help="Input video path")
    p.add_argument("--output", "-o", default=str(config.RESULTS_DIR),
                   help="Output directory")
    p.add_argument("--name",   "-n", default=None,    help="Base name for output files")
    p.add_argument("--skip",   "-s", type=int, default=config.DEFAULT_SKIP,
                   help="Frame skip (0 = every frame)")
    p.add_argument("--max-frames", "-m", type=int, default=None,
                   help="Maximum frames to process")

    # Modes
    p.add_argument("--doubles",       action="store_true", help="Doubles match (4 players)")
    p.add_argument("--pose",          action="store_true", help="Enable pose/kinesiology analysis")
    p.add_argument("--no-court-cal",  action="store_true",
                   help="Skip interactive court calibration (use auto-detection)")
    p.add_argument("--cal-map", default=None,
                   help="Path to an existing calibration map JSON to reuse")

    # Output control
    p.add_argument("--no-video", action="store_true", help="Do not save annotated video")
    p.add_argument("--no-json",  action="store_true", help="Do not save JSON")
    p.add_argument("--quiet",    action="store_true", help="Suppress progress bar")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    video_path = Path(args.input)
    if not video_path.exists():
        print(f"[Error] File not found: {args.input}")
        sys.exit(1)

    pipeline = Pipeline(
        output_dir        = args.output,
        frame_skip        = args.skip,
        save_video        = not args.no_video,
        save_json         = not args.no_json,
        show_progress     = not args.quiet,
        doubles           = args.doubles,
        enable_pose       = args.pose,
        interactive_court = not args.no_court_cal,
        cal_map_path      = args.cal_map,
    )

    print(f"\nTennis Tracker")
    print(f"  Input  : {args.input}")
    print(f"  Output : {args.output}")
    print(f"  Mode   : {'Doubles' if args.doubles else 'Singles'}")
    print(f"  Pose   : {'ON' if args.pose else 'OFF'}")

    try:
        result = pipeline.process(
            video_path  = str(video_path),
            max_frames  = args.max_frames,
            output_name = args.name,
        )

        # ── Summary ───────────────────────────────────────────────────────────
        total = len(result.frames)
        unique_ids = {p.track_id for f in result.frames for p in f.players}
        print(f"\n── Results ───────────────────────────────")
        print(f"  Frames processed : {total}")
        print(f"  Unique player IDs: {len(unique_ids)}")
        print(f"  Rallies detected : {len(result.rallies)}")
        if result.rallies:
            fastest = max(r.max_ball_speed_ms for r in result.rallies)
            print(f"  Fastest ball     : {fastest:.1f} m/s  ({fastest * 3.6:.0f} km/h)")

    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(0)
    except Exception as e:
        print(f"[Error] {e}")
        raise


if __name__ == "__main__":
    main()