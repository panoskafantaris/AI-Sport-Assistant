"""
GPU / CUDA setup checker + diagnostic + benchmark.

Usage:
    python models/check_gpu.py                  # full check
    python models/check_gpu.py --benchmark      # + YOLO speed test
    python models/check_gpu.py --fix            # print exact fix commands
"""
from __future__ import annotations
import argparse
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# nvidia-smi search paths (Windows stores it outside PATH by default)
_NVIDIA_SMI_CANDIDATES = [
    "nvidia-smi",
    r"C:\Windows\System32\nvidia-smi.exe",
    r"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe",
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _find_nvidia_smi() -> str | None:
    for candidate in _NVIDIA_SMI_CANDIDATES:
        try:
            subprocess.run([candidate, "--version"], capture_output=True, timeout=5)
            return candidate
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
    return None


def _run(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(cmd, text=True, timeout=10,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return ""


# ── Checks ────────────────────────────────────────────────────────────────────

def check_torch() -> dict:
    result = {"installed": False, "version": "", "cuda_built": "", "is_cpu_only": True}
    try:
        import torch
        result["installed"]   = True
        result["version"]     = torch.__version__
        result["cuda_built"]  = torch.version.cuda or ""
        result["is_cpu_only"] = "+cpu" in torch.__version__ or not torch.version.cuda
    except ImportError:
        pass
    return result


def check_cuda() -> dict:
    result = {"available": False, "name": "", "vram_gb": 0.0, "compute": ""}
    try:
        import torch
        if not torch.cuda.is_available():
            return result
        props = torch.cuda.get_device_properties(0)
        result["available"] = True
        result["name"]      = torch.cuda.get_device_name(0)
        result["vram_gb"]   = props.total_memory / 1e9
        result["compute"]   = f"{props.major}.{props.minor}"
    except Exception:
        pass
    return result


def check_driver() -> dict:
    result = {"found": False, "path": "", "driver": "", "cuda_driver": "", "gpu": "", "vram_total": ""}
    smi = _find_nvidia_smi()
    if not smi:
        return result
    result["found"] = True
    result["path"]  = smi
    raw = _run([smi,
                "--query-gpu=name,driver_version,memory.total,cuda_version",
                "--format=csv,noheader"])
    if raw:
        parts = [p.strip() for p in raw.split(",")]
        if len(parts) >= 4:
            result["gpu"]         = parts[0]
            result["driver"]      = parts[1]
            result["vram_total"]  = parts[2]
            result["cuda_driver"] = parts[3]
    return result


def check_ultralytics() -> str:
    try:
        import ultralytics
        return ultralytics.__version__
    except ImportError:
        return ""


# ── Diagnosis ─────────────────────────────────────────────────────────────────

def diagnose(torch_info: dict, cuda_info: dict, driver_info: dict) -> list[str]:
    issues = []

    if not torch_info["installed"]:
        issues.append(
            "PROBLEM: PyTorch is not installed.\n"
            "  FIX: pip install torch torchvision "
            "--index-url https://download.pytorch.org/whl/cu121"
        )
        return issues

    if torch_info["is_cpu_only"]:
        issues.append(
            "PROBLEM: You have the CPU-only build of PyTorch "
            f"({torch_info['version']}).\n"
            "  pip install with --index-url is ignored when a cached version exists.\n"
            "  You must UNINSTALL first, then reinstall:\n\n"
            "    pip uninstall torch torchvision torchaudio -y\n"
            "    pip install torch torchvision "
            "--index-url https://download.pytorch.org/whl/cu121"
        )

    if not driver_info["found"]:
        issues.append(
            "PROBLEM: nvidia-smi was not found in PATH or common Windows paths.\n"
            "  This is common on Windows — it does not mean the driver is absent.\n"
            "  Check manually:\n"
            r"    C:\Windows\System32\nvidia-smi.exe" + "\n"
            r"    C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe" + "\n"
            "  If neither file exists, install/update the NVIDIA driver:\n"
            "    https://www.nvidia.com/drivers\n"
            "  After installing, REBOOT and re-run this script."
        )
    else:
        try:
            major = int(driver_info["cuda_driver"].split(".")[0])
            if major < 12:
                issues.append(
                    f"PROBLEM: Driver CUDA {driver_info['cuda_driver']} < 12.1 required.\n"
                    "  FIX: Update NVIDIA driver from https://www.nvidia.com/drivers"
                )
        except Exception:
            pass

    if driver_info["found"] and not cuda_info["available"] and not torch_info["is_cpu_only"]:
        issues.append(
            "PROBLEM: Driver found but torch.cuda.is_available() = False.\n"
            "  PyTorch CUDA version may not match the driver.\n"
            "    pip uninstall torch torchvision -y\n"
            "    pip install torch torchvision "
            "--index-url https://download.pytorch.org/whl/cu121"
        )

    return issues


# ── Benchmark ─────────────────────────────────────────────────────────────────

def benchmark_yolo(device: str) -> None:
    try:
        import numpy as np
        import torch
        from ultralytics import YOLO

        model_name = "yolov8n.pt"
        print(f"\n── YOLO Benchmark ({device.upper()} / {model_name}) ────────────────")
        model = YOLO(model_name)
        model.to(device)
        if device == "cuda":
            model.model.half()

        dummy = np.zeros((720, 1280, 3), dtype=np.uint8)
        N = 20

        for _ in range(3):
            model.predict(dummy, verbose=False, imgsz=640)
        if device == "cuda":
            torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(N):
            model.predict(dummy, verbose=False, imgsz=1280)
        if device == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        fps = N / elapsed
        ms  = elapsed / N * 1000
        print(f"  {N} frames in {elapsed:.2f}s  =>  {fps:.1f} FPS  ({ms:.0f} ms/frame)")

        if fps >= 30:
            print("  OK: Real-time capable (>= 30 FPS)")
        else:
            skip = max(1, round(30 / fps) - 1)
            print(f"  ~  Use --skip {skip} to approximate real-time")

        if device == "cuda":
            allocated = torch.cuda.memory_allocated(0) / 1e9
            reserved  = torch.cuda.memory_reserved(0) / 1e9
            print(f"  VRAM used : {allocated:.2f} GB allocated / {reserved:.2f} GB reserved")

    except Exception as e:
        print(f"  Benchmark failed: {e}")


# ── Fix instructions ──────────────────────────────────────────────────────────

def print_fix() -> None:
    print("""
COMPLETE FIX for Windows + RTX 4050 + CUDA 12.1
================================================

Step 1 – Update NVIDIA drivers (if not done yet)
  https://www.nvidia.com/drivers
  Select: GeForce RTX 4050 / Windows 11 / Game Ready Driver
  REBOOT after installing.

Step 2 – Activate your conda env
  conda activate tennis-tracker

Step 3 – Remove the CPU-only PyTorch (THIS is why CUDA didn't install)
  pip uninstall torch torchvision torchaudio -y

Step 4 – Install CUDA 12.1 PyTorch
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

Step 5 – Verify
  python -c "import torch; print(torch.cuda.is_available())"
  Should print: True

Step 6 – Re-run this script
  python models/check_gpu.py --benchmark
""")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GPU setup checker for Tennis Tracker")
    p.add_argument("--benchmark", action="store_true", help="Run YOLO speed benchmark")
    p.add_argument("--fix",       action="store_true", help="Print step-by-step fix guide")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.fix:
        print_fix()
        return

    print("\n── Environment ───────────────────────────────────────────────")
    torch_info = check_torch()
    if torch_info["installed"]:
        tag = "  <-- CPU-only build! [X]" if torch_info["is_cpu_only"] else "  [OK]"
        print(f"  PyTorch    : {torch_info['version']}{tag}")
        cuda_tag = torch_info["cuda_built"] or "N/A (CPU build)"
        print(f"  CUDA built : {cuda_tag}")
    else:
        print("  PyTorch    : NOT INSTALLED")

    print("\n── NVIDIA Driver ─────────────────────────────────────────────")
    driver_info = check_driver()
    if driver_info["found"]:
        print(f"  nvidia-smi   : {driver_info['path']}")
        print(f"  GPU          : {driver_info['gpu']}")
        print(f"  Driver ver   : {driver_info['driver']}")
        print(f"  CUDA (driver): {driver_info['cuda_driver']}")
        print(f"  VRAM (total) : {driver_info['vram_total']}")
    else:
        print("  nvidia-smi : NOT FOUND in PATH or common Windows locations")

    print("\n── CUDA (torch) ──────────────────────────────────────────────")
    cuda_info = check_cuda()
    if cuda_info["available"]:
        print(f"  GPU        : {cuda_info['name']}  [OK]")
        print(f"  VRAM       : {cuda_info['vram_gb']:.1f} GB")
        print(f"  Compute    : {cuda_info['compute']}  (Ada Lovelace = 8.9)")
    else:
        print("  torch.cuda.is_available() -> False")

    print("\n── Libraries ─────────────────────────────────────────────────")
    ul = check_ultralytics()
    print(f"  Ultralytics: {ul or 'NOT INSTALLED'}")

    print("\n── Diagnosis ─────────────────────────────────────────────────")
    issues = diagnose(torch_info, cuda_info, driver_info)

    if not issues and cuda_info["available"]:
        print("  [OK] Everything looks good — GPU is ready!")
        print("  [OK] FP16 will be enabled automatically (2x speedup on 4050)")
    else:
        for i, issue in enumerate(issues, 1):
            print(f"\n  [{i}] {issue}")
        print("\n  --> Run:  python models/check_gpu.py --fix  for full instructions")

    if args.benchmark:
        device = "cuda" if cuda_info["available"] else "cpu"
        benchmark_yolo(device)

    print()


if __name__ == "__main__":
    main()