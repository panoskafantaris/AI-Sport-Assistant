"""
Download a tennis match video from YouTube.

Usage:
    python scripts/download_sample.py -u "https://youtu.be/XXXX"
    python scripts/download_sample.py -u "https://youtu.be/XXXX" -r 1080 -n my_match
    python scripts/download_sample.py --list-formats -u "https://youtu.be/XXXX"
"""
import argparse
import subprocess
import sys
import shutil
from pathlib import Path

# Allow running from project root or scripts/ folder
sys.path.insert(0, str(Path(__file__).parent.parent))
import config


# ── Helpers ───────────────────────────────────────────────────────────────────

def _require_ytdlp() -> None:
    """Exit with a helpful message if yt-dlp is not installed."""
    if shutil.which("yt-dlp") is None:
        print("[Error] yt-dlp is not installed.")
        print("  Install: pip install yt-dlp")
        print("  Docs   : https://github.com/yt-dlp/yt-dlp")
        sys.exit(1)


def _format_selector(resolution: str, fmt: str) -> str:
    """
    Build a yt-dlp format selector that prefers the requested resolution
    and falls back gracefully if not available.

    Examples:
        resolution="1080", fmt="mp4"  →  bestvideo[height<=1080][ext=mp4]+bestaudio/...
    """
    res = resolution
    return (
        f"bestvideo[height<={res}][ext={fmt}]+bestaudio[ext=m4a]"
        f"/bestvideo[height<={res}]+bestaudio"
        f"/best[height<={res}]"
        f"/best"
    )


def list_formats(url: str) -> None:
    """Print all available formats for a YouTube URL."""
    print(f"\nAvailable formats for: {url}\n")
    subprocess.run(["yt-dlp", "-F", url], check=False)


def download_video(
    url: str,
    output_dir: Path,
    filename: str = "tennis_sample",
    resolution: str = "1080",
    fmt: str = "mp4",
    quiet: bool = False,
) -> Path:
    """
    Download a video using yt-dlp.

    Args:
        url:        YouTube URL (watch, youtu.be, shorts, etc.)
        output_dir: Directory to save the file into.
        filename:   Base name without extension.
        resolution: Max vertical resolution (e.g. "720", "1080").
        fmt:        Preferred container format ("mp4" recommended).
        quiet:      Suppress yt-dlp progress output.

    Returns:
        Path to the downloaded file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # yt-dlp template – it will add the correct extension automatically
    output_template = str(output_dir / f"{filename}.%(ext)s")

    cmd = [
        "yt-dlp",
        "--format",         _format_selector(resolution, fmt),
        "--merge-output-format", fmt,
        "--output",         output_template,
        "--no-playlist",                   # single video only
        "--add-metadata",                  # embed title/date into file
        "--embed-thumbnail",               # embed thumbnail (optional, nice to have)
    ]

    if quiet:
        cmd += ["--quiet", "--no-warnings"]
    else:
        cmd += ["--progress"]

    cmd.append(url)

    print(f"\n[Download] URL        : {url}")
    print(f"[Download] Resolution : up to {resolution}p")
    print(f"[Download] Output dir : {output_dir}\n")

    result = subprocess.run(cmd, check=False)

    if result.returncode != 0:
        # yt-dlp sometimes exits non-zero for minor issues but still downloads.
        # Check if the file exists before giving up.
        print("[Warning] yt-dlp reported a non-zero exit. Checking for file…")

    # Find the downloaded file (extension may differ from requested)
    for ext in ("mp4", "mkv", "webm", "avi"):
        candidate = output_dir / f"{filename}.{ext}"
        if candidate.exists() and candidate.stat().st_size > 0:
            return candidate

    print("[Error] Download failed – no output file found.")
    print("  Try --list-formats to see what resolutions are available.")
    sys.exit(1)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Download a tennis video from YouTube for use with Tennis Tracker",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--url",  "-u", required=True,
                   help="YouTube video URL")
    p.add_argument("--output", "-o", default=str(config.SAMPLES_DIR),
                   help="Directory to save the video")
    p.add_argument("--name", "-n", default="tennis_sample",
                   help="Output filename (without extension)")
    p.add_argument("--resolution", "-r", default="1080",
                   choices=["480", "720", "1080", "1440", "2160"],
                   help="Maximum vertical resolution to download")
    p.add_argument("--format", "-f", default="mp4", dest="fmt",
                   help="Preferred container format")
    p.add_argument("--list-formats", "-F", action="store_true",
                   help="List available formats and exit (no download)")
    p.add_argument("--quiet", "-q", action="store_true",
                   help="Suppress yt-dlp progress output")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    _require_ytdlp()

    if args.list_formats:
        list_formats(args.url)
        return

    output_path = download_video(
        url        = args.url,
        output_dir = Path(args.output),
        filename   = args.name,
        resolution = args.resolution,
        fmt        = args.fmt,
        quiet      = args.quiet,
    )

    size_mb = output_path.stat().st_size / 1_000_000
    print(f"\n[Download] ✓ Complete!")
    print(f"  File : {output_path}")
    print(f"  Size : {size_mb:.1f} MB")
    print(f"\nNext step — run the tracker:")
    print(f"  python main.py -i {output_path} --interactive")


if __name__ == "__main__":
    main()