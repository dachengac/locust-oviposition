"""Measure CPU model throughput and process memory without exporting video frames."""
import ctypes
import argparse
import ctypes.wintypes as wintypes
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cv2
import numpy as np
from behaviorlab.engine import YoloOnnx
from behaviorlab import __version__
from behaviorlab.modelpack import load_pack, install_pack

parser = argparse.ArgumentParser()
parser.add_argument("--model", help="已解压的检测模型包目录或 .behavior.zip；默认使用本机最新桌面构建的模型")
args = parser.parse_args()
if args.model:
    model_path = Path(args.model)
    pack = install_pack(model_path, ROOT / "build" / "benchmark-models") if model_path.is_file() else load_pack(model_path)
else:
    manifest = next((ROOT / "dist" / f"BehaviorLab-{__version__}" / "models").glob("*/manifest.json"), None)
    if manifest is None:
        parser.error("未找到桌面构建的模型，请传 --model path/to/model.behavior.zip")
    pack = load_pack(manifest.parent)
if pack.manifest["adapter"] != "yolo_onnx":
    parser.error("此基准测试只支持 YOLO ONNX 检测模型")
frame = np.random.default_rng(21).integers(0, 256, (480, 854, 3), dtype=np.uint8)
start = time.perf_counter()
adapter = YoloOnnx(pack)
load_seconds = time.perf_counter() - start
for _ in range(3):
    adapter.predict(frame, .7)
times = []
for _ in range(30):
    start = time.perf_counter()
    detections = adapter.predict(frame, .7)
    times.append(time.perf_counter() - start)


class MemoryCounters(ctypes.Structure):
    _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
        (name, ctypes.c_size_t) for name in ["PeakWorkingSetSize", "WorkingSetSize",
        "QuotaPeakPagedPoolUsage", "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage",
        "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage"]]


get_process = ctypes.windll.kernel32.GetCurrentProcess
get_process.restype = wintypes.HANDLE
get_memory = ctypes.windll.psapi.GetProcessMemoryInfo
get_memory.argtypes = [wintypes.HANDLE, ctypes.POINTER(MemoryCounters), wintypes.DWORD]
get_memory.restype = wintypes.BOOL
memory = MemoryCounters()
memory.cb = ctypes.sizeof(memory)
assert get_memory(get_process(), ctypes.byref(memory), memory.cb)
cache_estimates = []
for count in (2, 10):
    data = dict(tracks={str(tid): [[float(t), 100.12345678901234 + tid,
                200.23456789012345 + tid, 40.123456789, 25.23456789]
                for t in range(3600)] for tid in range(count)},
                samples=[dict(time=float(t), classes=[0]) for t in range(3600)])
    cache_estimates.append(dict(video_hours=1, sample_fps=1, animals=count,
                                estimated_json_cache_MB=len(json.dumps(data).encode()) / 1e6))
report = dict(benchmark="YOLO11s, CPU, synthetic 854x480 frame, model input 640x640",
              samples=len(times), opencv_threads=cv2.getNumThreads(), model_load_seconds=load_seconds,
              seconds_per_sample_mean=float(np.mean(times)), seconds_per_sample_median=float(np.median(times)),
              model_only_minutes_for_1h_video_at_1fps=float(np.mean(times) * 60),
              peak_working_set_MB=memory.PeakWorkingSetSize / 1e6,
              working_set_MB=memory.WorkingSetSize / 1e6,
              peak_committed_MB=memory.PeakPagefileUsage / 1e6,
              cache_estimates=cache_estimates,
              limitation="Synthetic inference only: excludes video hashing, decoding, tracking, temporal analysis and GUI. Cache sizes are estimates.")
(ROOT / "assets" / "verification" / "performance-estimate.json").write_text(
    json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(report, ensure_ascii=False, indent=2))
