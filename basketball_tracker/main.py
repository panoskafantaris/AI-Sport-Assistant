"""
CLI entry point for basketball tracker.
"""
import argparse
import sys
from pathlib import Path

from src.pipeline import Pipeline
import config


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Basketball player tracking from video",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to input video file"
    )
    
    parser.add_argument(
        "--output", "-o",
        default=str(config.RESULTS_DIR),
        help="Output directory for results"
    )
    
    parser.add_argument(
        "--name", "-n",
        default=None,
        help="Base name for output files (default: input filename)"
    )
    
    parser.add_argument(
        "--skip", "-s",
        type=int,
        default=config.DEFAULT_FRAME_SKIP,
        help="Frames to skip between processing (0 = process all)"
    )
    
    parser.add_argument(
        "--max-frames", "-m",
        type=int,
        default=None,
        help="Maximum frames to process (default: all)"
    )
    
    parser.add_argument(
        "--no-video",
        action="store_true",
        help="Don't save annotated output video"
    )
    
    parser.add_argument(
        "--no-json",
        action="store_true",
        help="Don't save JSON tracking data"
    )
    
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress progress bar"
    )
    
    # Team classification options
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Enable interactive calibration (click to identify teams/referee)"
    )
    
    parser.add_argument(
        "--no-teams",
        action="store_true",
        help="Disable team classification entirely"
    )
    
    parser.add_argument(
        "--load-calibration",
        type=str,
        default=None,
        help="Load color calibration from file"
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Validate input
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)
    
    # Create pipeline
    pipeline = Pipeline(
        output_dir=args.output,
        frame_skip=args.skip,
        save_video=not args.no_video,
        save_json=not args.no_json,
        show_progress=not args.quiet,
        enable_team_classification=not args.no_teams,
        interactive_calibration=args.interactive
    )
    
    # Load existing calibration if specified
    if args.load_calibration:
        if pipeline.team_classifier:
            if pipeline.team_classifier.load_calibration(args.load_calibration):
                print(f"Loaded calibration from: {args.load_calibration}")
            else:
                print(f"Failed to load calibration from: {args.load_calibration}")
    
    # Process video
    print(f"Processing: {args.input}")
    print(f"Output directory: {args.output}")
    
    if args.interactive:
        print("Interactive mode: You will be asked to identify players")
    
    try:
        result = pipeline.process(
            video_path=str(input_path),
            max_frames=args.max_frames,
            output_name=args.name
        )
        
        # Print summary
        print("\n--- Processing Complete ---")
        print(f"Frames processed: {len(result.frames)}")
        
        unique_tracks = set()
        team_counts = {
            "TEAM_A": 0,
            "TEAM_B": 0,
            "REFEREE": 0,
            "UNKNOWN": 0
        }
        
        for frame in result.frames:
            for obj in frame.tracked_objects:
                if obj.track_id not in unique_tracks:
                    unique_tracks.add(obj.track_id)
                    team_counts[obj.team.name] += 1
        
        print(f"Unique players tracked: {len(unique_tracks)}")
        
        if not args.no_teams:
            print(f"  Team A: {team_counts['TEAM_A']}")
            print(f"  Team B: {team_counts['TEAM_B']}")
            print(f"  Referees: {team_counts['REFEREE']}")
            if team_counts['UNKNOWN'] > 0:
                print(f"  Unclassified: {team_counts['UNKNOWN']}")
        
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"Error during processing: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()