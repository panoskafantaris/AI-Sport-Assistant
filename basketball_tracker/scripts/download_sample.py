"""
Download sample basketball video from YouTube.
"""
import argparse
import subprocess
import sys
from pathlib import Path

# Add parent directory to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent))
import config


def check_ytdlp_installed() -> bool:
    """Check if yt-dlp is installed."""
    try:
        subprocess.run(
            ["yt-dlp", "--version"],
            capture_output=True,
            check=True
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def download_video(
    url: str,
    output_dir: Path,
    filename: str = "sample",
    resolution: str = config.DOWNLOAD_RESOLUTION
) -> Path:
    """
    Download video from YouTube.
    
    Args:
        url: YouTube URL
        output_dir: Directory to save video
        filename: Output filename (without extension)
        resolution: Maximum resolution (e.g., '720', '480')
    
    Returns:
        Path to downloaded video
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_template = str(output_dir / f"{filename}.%(ext)s")
    
    cmd = [
        "yt-dlp",
        "-f", f"bestvideo[height<={resolution}][ext=mp4]+bestaudio[ext=m4a]/best[height<={resolution}][ext=mp4]/best",
        "--merge-output-format", "mp4",
        "-o", output_template,
        "--no-playlist",
        url
    ]
    
    print(f"Downloading: {url}")
    print(f"Output: {output_dir}")
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Download failed: {e}")
        sys.exit(1)
    
    # Find the downloaded file
    for ext in ["mp4", "mkv", "webm"]:
        output_path = output_dir / f"{filename}.{ext}"
        if output_path.exists():
            return output_path
    
    raise FileNotFoundError("Downloaded file not found")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Download sample basketball video from YouTube"
    )
    
    parser.add_argument(
        "--url", "-u",
        default=config.DEFAULT_SAMPLE_URL,
        help="YouTube URL to download"
    )
    
    parser.add_argument(
        "--output", "-o",
        default=str(config.SAMPLES_DIR),
        help="Output directory"
    )
    
    parser.add_argument(
        "--name", "-n",
        default="basketball_sample",
        help="Output filename (without extension)"
    )
    
    parser.add_argument(
        "--resolution", "-r",
        default=config.DOWNLOAD_RESOLUTION,
        help="Maximum resolution (e.g., 720, 480)"
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Check yt-dlp is installed
    if not check_ytdlp_installed():
        print("Error: yt-dlp is not installed.")
        print("Install with: pip install yt-dlp")
        sys.exit(1)
    
    # Download video
    output_dir = Path(args.output)
    
    try:
        output_path = download_video(
            url=args.url,
            output_dir=output_dir,
            filename=args.name,
            resolution=args.resolution
        )
        print(f"\nDownload complete: {output_path}")
        print(f"\nRun tracker with:")
        print(f"  python main.py --input {output_path}")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()