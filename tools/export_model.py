"""Developer-only PT -> portable ONNX pack; never bundled in the application."""
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
    parser.add_argument("--output", required=True)
    parser.add_argument("--name", default="飞蝗产卵 · 已发布模型")
    parser.add_argument("--behavior", default="产卵")
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()
    from ultralytics import YOLO
    from behaviorlab.modelpack import create_pack, DEFAULTS, file_sha256
    model = YOLO(args.weights)
    names = [model.names[i] for i in range(len(model.names))]
    print(json.dumps(dict(labels=names, task=model.task, weights_sha256=file_sha256(args.weights)), ensure_ascii=False))
    if model.task != "detect":
        raise ValueError("此转换入口用于 YOLO 检测模型；分类/姿态需要对应的模型包接口")
    exported = model.export(format="onnx", imgsz=args.imgsz, opset=12, simplify=False,
                            dynamic=False, half=False, nms=False, batch=1, device="cpu")
    create_pack(exported, args.output, name=args.name, labels=names,
                input_size=(args.imgsz, args.imgsz), behavior_label=args.behavior,
                parameters=DEFAULTS, extra=dict(source_weights_sha256=file_sha256(args.weights),
                description="使用原项目净位移率与时序算法；Windows CPU ONNX 推理，需在实验视频上复核部署结果。"))
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
