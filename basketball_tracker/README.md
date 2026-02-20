# Tennis Tracker

AI-powered tennis match analysis: court detection, player tracking, ball tracking, and advanced statistics.

## Installation & Setup

```bash
conda create -n tennis-tracker python=3.10
conda activate tennis-tracker

# 1. Install base dependencies
pip install -r requirements.txt

# 2. Install PyTorch with CUDA 12.1  (RTX 4050 / Ada Lovelace GPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# CPU-only fallback (slower):
# pip install torch torchvision

# 3. Verify GPU is detected
python models/check_gpu.py

# 4. Download YOLO models
python models/download_models.py           # required models only
python models/download_models.py --all     # + optional pose model
```

## Downloading a Match Video

```bash
# Download 1080p (default)
python scripts/download_sample.py -u "https://www.youtube.com/watch?v=OZz_ro54jYs"

# Choose resolution and set a name
python scripts/download_sample.py -u "https://www.youtube.com/watch?v=OZz_ro54jYs" -r 1280 -n tennis_match

# See all available formats before downloading
python scripts/download_sample.py -u "https://www.youtube.com/watch?v=OZz_ro54jYs" --list-formats
```

## GPU Notes (RTX 4050 6 GB)

```bash
# Quick benchmark to confirm speed
python models/check_gpu.py --benchmark
```

## Installation

```bash
conda create -n tennis-tracker python=3.10
conda activate tennis-tracker
cd basketball_tracker
pip install -r requirements.txt
```

## Quick Start

```bash
# Singles match – interactive court calibration
python main.py -i samples/tennis_match_2.mp4 --max-frames 3000

# Doubles, with pose/kinesiology, process every other frame
python main.py -i samples/tennis_match.mp4 --doubles --pose --skip 1

# Skip interactive calibration (auto line-detection only)
python main.py -i samples/tennis_match.mp4 --no-court-cal

# Fast preview (first 200 frames, no output files)
python main.py -i samples/tennis_match.mp4 --max-frames 200 --no-video --no-json


# Analyse the specific frames that look wrong
python scripts/debug_court.py -i samples/tennis_match.mp4 -f 70 --interactive
python scripts/debug_court.py -i samples/tennis_match.mp4 -f 210 --interactive
python scripts/debug_court.py -i samples/tennis_match.mp4 -f 1170 --interactive
```