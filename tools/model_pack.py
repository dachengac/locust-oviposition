"""CLI for creating versioned ONNX model packs with full contract options."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from behaviorlab.modelpack import create_pack


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--labels", nargs="+", required=True)
    parser.add_argument("--pipeline", choices=["oviposition_v1", "detection_presence_v1", "classification_v1"], default="oviposition_v1")
    parser.add_argument("--input-size", nargs=2, type=int, default=[640, 640])
    parser.add_argument("--behavior", default="产卵")
    parser.add_argument("--parameters", default="{}", help="JSON 分析参数")
    parser.add_argument("--output-kind", choices=["probabilities", "logits"], default="probabilities")
    parser.add_argument("--mean", nargs=3, type=float, default=[0, 0, 0])
    parser.add_argument("--std", nargs=3, type=float, default=[1, 1, 1])
    parser.add_argument("--layout", choices=["channels_first", "rows"], default="channels_first")
    args = parser.parse_args()
    create_pack(args.weights, args.output, name=args.name, labels=args.labels, pipeline=args.pipeline,
                adapter="frame_classifier_onnx" if args.pipeline == "classification_v1" else "yolo_onnx",
                input_size=args.input_size, behavior_label=args.behavior, parameters=json.loads(args.parameters),
                extra=dict(output_kind=args.output_kind, mean=args.mean, std=args.std, output_layout=args.layout))
    print(args.output)


if __name__ == "__main__":
    main()
