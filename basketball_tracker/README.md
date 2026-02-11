## Installation

```bash
# Clone or navigate to the project
cd basketball_tracker

conda create -n ai-sport-assistant python=3.10

conda activate ai-sport-assistant

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### 1. Download a sample video

```bash
python scripts/download_sample.py

# Or with a custom YouTube URL
python scripts/download_sample.py --url "https://www.youtube.com/watch?v=HYMDxPO0L3M"
```

### 2. Run the tracker

```bash
python main.py --input samples/basketball_sample.f398.mp4 --output results/
```

## Usage

### Basic usage

```bash
python main.py --input VIDEO_PATH --output OUTPUT_DIR
```

### Options

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--input` | `-i` | Input video path | Required |
| `--output` | `-o` | Output directory | `results/` |
| `--name` | `-n` | Base name for outputs | Input filename |
| `--skip` | `-s` | Frames to skip | 0 (process all) |
| `--max-frames` | `-m` | Max frames to process | All |
| `--no-video` | | Don't save annotated video | False |
| `--no-json` | | Don't save JSON data | False |
| `--quiet` | `-q` | Hide progress bar | False |

### Examples

```bash
# Process entire video
python main.py -i game.mp4

# Process every 3rd frame (faster)
python main.py -i game.mp4 --skip 2

# Process first 100 frames only
python main.py -i samples/basketball_sample.f398.mp4 --max-frames 100

# Only save JSON, no video
python main.py -i game.mp4 --no-video
```

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=src

# Run specific test file
pytest tests/test_pipeline.py -v
```
