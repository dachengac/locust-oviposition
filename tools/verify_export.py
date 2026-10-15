"""Raw-tensor PT/ONNX comparison on deterministic synthetic frames, not accuracy validation."""
import argparse
import json
import os
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / "assets" / "yolo-config"))
Path(os.environ["YOLO_CONFIG_DIR"]).mkdir(parents=True, exist_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--pack", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    import cv2
    import numpy as np
    import torch
    from ultralytics import YOLO
    from behaviorlab.modelpack import load_pack
    from behaviorlab.engine import YoloOnnx
    pack = load_pack(args.pack)
    adapter = YoloOnnx(pack)
    model = YOLO(args.weights)
    model.model.fuse()
    model.model.eval()
    torch.set_num_threads(2)
    cv2.setNumThreads(2)
    random = np.random.default_rng(42)
    frames = [np.zeros((480, 854, 3), np.uint8), np.full((480, 854, 3), 127, np.uint8),
              random.integers(0, 256, (480, 854, 3), dtype=np.uint8)]
    measurements = []
    for frame in frames:
        blob, *_ = adapter.prepare(frame)
        with torch.inference_mode():
            output = model.model(torch.from_numpy(blob))
        expected = output[0].numpy() if isinstance(output, tuple) else output.numpy()
        actual, *_ = adapter.raw(frame)
        diff = np.abs(expected - actual)
        # Box coordinates are in 640px model space; score differences are reported separately.
        measurements.append(dict(shape=list(actual.shape), max_box_error_px=float(diff[:, :4].max()),
                                 max_score_error=float(diff[:, 4:].max()), mean_error=float(diff.mean())))
        assert diff[:, :4].max() < .05 and diff[:, 4:].max() < 1e-4, measurements[-1]
    report = dict(ok=True, frames=len(frames), measurements=measurements,
                  limitation="仅验证转换数值一致性；没有真实实验录像，未验证部署版行为识别准确率。")
    Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
