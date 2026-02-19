"""
Download all required (and optionally all optional) YOLO models.

The script uses Ultralytics' built-in auto-download as the primary method
(it handles checksums, caching, and CDN selection automatically).
A manual requests-based fallback is provided in case the network
blocks GitHub releases.

Usage:
    python models/download_models.py              # required only
    python models/download_models.py --all        # required + optional
    python models/download_models.py --model yolov8m-pose.pt
    python models/download_models.py --check      # show status, no download
"""
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

# Allow running from project root or models/ folder
sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from models.registry import MODELS, model_path, is_downloaded, missing_required


# ── Download strategies ───────────────────────────────────────────────────────

def _download_via_ultralytics(name: str) -> bool:
    """
    Let Ultralytics handle the download (preferred).
    It auto-places the file in its cache; we then copy to our models/ dir.
    """
    try:
        from ultralytics import YOLO
        import shutil

        print(f"  [ultralytics] Downloading {name} …")
        # YOLO("yolov8m.pt") triggers auto-download into ~/.ultralytics/
        model = YOLO(name)
        # Locate the cached file
        src = Path(model.ckpt_path) if hasattr(model, "ckpt_path") else None
        if src and src.exists():
            dest = model_path(name)
            shutil.copy2(src, dest)
            print(f"  [ultralytics] Copied → {dest}")
            return True
        # If ckpt_path not available, model is already in the right place
        return is_downloaded(name)
    except Exception as e:
        print(f"  [ultralytics] Failed: {e}")
        return False


def _download_via_requests(name: str) -> bool:
    """Manual HTTP download with progress bar as fallback."""
    import requests

    info = MODELS[name]
    url  = info["url"]
    dest = model_path(name)
    dest.parent.mkdir(parents=True, exist_ok=True)

    print(f"  [requests] GET {url}")
    try:
        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            written = 0
            start = time.time()
            with open(dest, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 20):  # 1 MB chunks
                    f.write(chunk)
                    written += len(chunk)
                    if total:
                        pct  = written / total * 100
                        mb   = written / 1e6
                        spd  = mb / max(time.time() - start, 0.01)
                        print(f"\r    {pct:5.1f}%  {mb:.1f}/{total/1e6:.1f} MB"
                              f"  {spd:.1f} MB/s", end="", flush=True)
            print()
        return is_downloaded(name)
    except Exception as e:
        print(f"\n  [requests] Failed: {e}")
        if dest.exists():
            dest.unlink()
        return False


def download_model(name: str) -> bool:
    """Download a single model. Returns True on success."""
    if is_downloaded(name):
        print(f"  ✓ Already downloaded: {name}")
        return True

    info = MODELS[name]
    print(f"\n→ {name}  ({info['size_mb']} MB)  [{info['phase']}]")

    # Try ultralytics first, fall back to direct HTTP
    if _download_via_ultralytics(name) and is_downloaded(name):
        return True
    return _download_via_requests(name)


# ── Status report ─────────────────────────────────────────────────────────────

def print_status() -> None:
    print("\n── Model Status ──────────────────────────────────────────────")
    for name, info in MODELS.items():
        ok   = is_downloaded(name)
        icon = "✓" if ok else "✗"
        req  = "required" if info["required"] else "optional"
        size = f"{info['size_mb']} MB"
        print(f"  {icon}  {name:<25}  {req:<8}  {size:<8}  {info['phase']}")
    print()


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Download YOLO models for Tennis Tracker",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--all",   action="store_true",
                   help="Download required AND optional models")
    p.add_argument("--model", default=None,
                   help="Download a specific model by filename")
    p.add_argument("--check", action="store_true",
                   help="Show download status and exit (no download)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    print_status()

    if args.check:
        return

    # Choose which models to download
    if args.model:
        if args.model not in MODELS:
            print(f"[Error] Unknown model: {args.model}")
            print(f"  Valid names: {list(MODELS.keys())}")
            sys.exit(1)
        targets = [args.model]
    elif args.all:
        targets = list(MODELS.keys())
    else:
        targets = [n for n, m in MODELS.items() if m["required"]]

    print(f"Downloading {len(targets)} model(s)…")
    failed = []
    for name in targets:
        ok = download_model(name)
        if not ok:
            failed.append(name)

    print("\n── Summary ───────────────────────────────────────────────────")
    for name in targets:
        ok = is_downloaded(name)
        print(f"  {'✓' if ok else '✗'}  {name}")

    if failed:
        print(f"\n[Error] {len(failed)} model(s) failed to download: {failed}")
        sys.exit(1)
    else:
        print("\nAll models ready. You can now run:")
        print("  python main.py -i samples/your_video.mp4 --interactive")


if __name__ == "__main__":
    main()