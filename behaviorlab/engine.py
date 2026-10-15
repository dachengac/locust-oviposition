"""CPU-only OpenCV DNN adapters and cancellable, content-keyed analysis."""
from __future__ import annotations

import hashlib
import json
import math
import os
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from .analysis import oviposition_events, presence_events
from .modelpack import file_sha256, validate_params
from .vendor.byte_tracker import BYTETracker

CACHE_VERSION = "opencv-yolo-bytetrack-1"


class YoloOnnx:
    def __init__(self, pack):
        self.pack = pack
        # OpenCV's Windows filename loader cannot read some Unicode paths.
        self.net = cv2.dnn.readNetFromONNX(np.frombuffer(pack.weights.read_bytes(), dtype=np.uint8))
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    def prepare(self, frame):
        width, height = self.pack.manifest["input_size"]
        h, w = frame.shape[:2]
        ratio = min(width / w, height / h)
        resized_w, resized_h = int(round(w * ratio)), int(round(h * ratio))
        left = int(round((width - resized_w) / 2 - .1))
        top = int(round((height - resized_h) / 2 - .1))
        canvas = np.full((height, width, 3), 114, dtype=np.uint8)
        canvas[top:top + resized_h, left:left + resized_w] = cv2.resize(frame, (resized_w, resized_h))
        return cv2.dnn.blobFromImage(canvas, 1 / 255., (width, height), swapRB=True), ratio, left, top

    def raw(self, frame):
        blob, ratio, left, top = self.prepare(frame)
        self.net.setInput(blob)
        names = self.net.getUnconnectedOutLayersNames()
        outputs = self.net.forward(names)
        if len(outputs) != 1:
            raise ValueError("检测模型必须只有一个输出；不支持姿态、分割或 end-to-end 输出")
        return outputs[0], ratio, left, top

    def predict(self, frame, confidence):
        output, ratio, left, top = self.raw(frame)
        if output.ndim != 3 or output.shape[0] != 1:
            raise ValueError(f"YOLO 输出形状不兼容：{output.shape}")
        rows = output[0].T if self.pack.manifest.get("output_layout", "channels_first") == "channels_first" else output[0]
        expected = 4 + len(self.pack.manifest["labels"])
        if rows.shape[1] != expected:
            raise ValueError(f"类别/模型不匹配：需要 {expected} 通道，实际 {rows.shape[1]}")
        if not np.isfinite(rows).all():
            raise ValueError("模型输出包含非有限数值")
        class_ids = np.argmax(rows[:, 4:], axis=1)
        scores = rows[np.arange(len(rows)), class_ids + 4]
        mask = scores >= confidence
        rows, class_ids, scores = rows[mask], class_ids[mask], scores[mask]
        if not len(rows):
            return np.zeros((0, 6), dtype=np.float64)
        xywh = rows[:, :4].copy()
        xywh[:, :2] -= xywh[:, 2:] / 2
        boxes = np.column_stack((xywh[:, :2], xywh[:, :2] + xywh[:, 2:]))
        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - left) / ratio
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - top) / ratio
        h, w = frame.shape[:2]
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h)
        valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
        boxes, class_ids, scores = boxes[valid], class_ids[valid], scores[valid]
        selected = []
        for class_id in np.unique(class_ids):
            indices = np.flatnonzero(class_ids == class_id)
            rects = np.column_stack((boxes[indices, :2], boxes[indices, 2:] - boxes[indices, :2]))
            keep = cv2.dnn.NMSBoxes(rects.tolist(), scores[indices].tolist(), 0.0, .7)
            if len(keep):
                selected.extend(indices[np.asarray(keep).reshape(-1)].tolist())
        selected.sort(key=lambda i: float(scores[i]), reverse=True)
        selected = selected[:300]
        return np.column_stack((boxes[selected], scores[selected], class_ids[selected])).astype(np.float64)


class FrameClassifierOnnx:
    """RGB, resize, NCHW, mean/std normalization; one score per declared label."""
    def __init__(self, pack):
        self.pack = pack
        self.net = cv2.dnn.readNetFromONNX(np.frombuffer(pack.weights.read_bytes(), dtype=np.uint8))
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    def predict(self, frame, confidence):
        manifest = self.pack.manifest
        rgb = cv2.cvtColor(cv2.resize(frame, tuple(manifest["input_size"])), cv2.COLOR_BGR2RGB).astype(np.float32) / 255
        rgb = (rgb - np.array(manifest.get("mean", [0, 0, 0]), dtype=np.float32)) / np.array(manifest.get("std", [1, 1, 1]), dtype=np.float32)
        self.net.setInput(rgb.transpose(2, 0, 1)[None])
        scores = self.net.forward().reshape(-1)
        if len(scores) != len(manifest["labels"]) or not np.isfinite(scores).all():
            raise ValueError("分类模型输出与模型包类别不匹配")
        if manifest.get("output_kind", "probabilities") == "logits":
            scores = np.exp(scores - scores.max())
            scores /= scores.sum()
        if np.any(scores < -1e-6) or np.any(scores > 1 + 1e-6):
            raise ValueError("概率必须在 0–1 之间；logits 模型请修改模型包配置")
        scores = scores.clip(0, 1)
        best = int(scores.argmax())
        return [best] if scores[best] >= confidence else []


ADAPTER_REGISTRY = {"yolo_onnx": YoloOnnx, "frame_classifier_onnx": FrameClassifierOnnx}


def video_info(path):
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise ValueError("视频无法打开，请检查文件格式或是否损坏")
        fps = cap.get(cv2.CAP_PROP_FPS)
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if not math.isfinite(fps) or fps <= 0 or count <= 0:
            raise ValueError("视频缺少有效帧率或时长信息")
        return dict(fps=fps, frame_count=count, duration=count / fps,
                    width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    finally:
        cap.release()


def cache_identity(pack, video_hash, params):
    # Temporal thresholds intentionally excluded: changing them only reruns temporal analysis.
    inputs = dict(version=CACHE_VERSION, opencv=cv2.__version__, model=pack.identity, video=video_hash,
                  conf=params["conf"], sample_fps=params["sample_fps"])
    return hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def extract(pack, video, params, cache_dir, cancel, progress):
    info = video_info(video)
    video_hash = file_sha256(video, cancel, lambda p: progress(p * .05, "检查视频与缓存"))
    identity = cache_identity(pack, video_hash, params)
    cache_file = Path(cache_dir) / (identity + ".json") if cache_dir else None
    if cache_file and cache_file.exists():
        try:
            data = json.loads(cache_file.read_text(encoding="utf-8"))
            if data["identity"] == identity:
                progress(.95, "已读取轨迹缓存")
                return data, True
        except (ValueError, KeyError, OSError):
            pass  # A partial/corrupt cache is recomputed, never treated as a video result.
    adapter = ADAPTER_REGISTRY[pack.manifest["adapter"]](pack)
    tracker = BYTETracker(SimpleNamespace(track_thresh=.25, new_track_thresh=.25,
                                         track_buffer=30, match_thresh=.8, mot20=False), frame_rate=30)
    cap = cv2.VideoCapture(str(video))
    stride = max(1, round(info["fps"] / params["sample_fps"]))
    tracks, samples, frame_index = {}, [], 0
    last_update = 0
    try:
        while cap.grab():
            if cancel and cancel.is_set():
                raise InterruptedError("已停止，当前视频可重新分析")
            if frame_index % stride == 0:
                ok, frame = cap.retrieve()
                if not ok:
                    raise ValueError(f"视频解码失败：第 {frame_index} 帧")
                timestamp = frame_index / info["fps"]
                if pack.manifest["adapter"] == "yolo_onnx":
                    detections = adapter.predict(frame, params["conf"])
                    classes = sorted(set(int(c) for c in detections[:, 5]))
                    target = pack.manifest.get("target_class", 0)
                    boxes = detections[detections[:, 5] == target, :5].copy()
                    # Input/output boxes remain in original video pixels, not resized coordinates.
                    active = tracker.update(boxes, (info["height"], info["width"]), (info["height"], info["width"]))
                    for track in active:
                        x, y, w, h = track.tlwh
                        tracks.setdefault(str(track.track_id), []).append(
                            [timestamp, float(x + w / 2), float(y + h / 2), float(w), float(h)])
                else:
                    classes = adapter.predict(frame, params["conf"])
                samples.append(dict(time=timestamp, classes=classes))
                if time.monotonic() - last_update > .2:
                    progress(.05 + .9 * frame_index / info["frame_count"], f"识别中 · {timestamp / 60:.1f} / {info['duration'] / 60:.1f} 分钟")
                    last_update = time.monotonic()
            frame_index += 1
    finally:
        cap.release()
    if frame_index < info["frame_count"] - max(2, info["fps"]):
        raise ValueError("视频提前结束或损坏；未保存不完整的分析缓存")
    data = dict(identity=identity, video_sha256=video_hash, info=info, tracks=tracks, samples=samples)
    if cache_file:
        atomic_json(cache_file, data)
    return data, False


def analyze(pack, video, params, cache_dir=None, cancel=None, progress=lambda p, text: None):
    params = validate_params(params)
    data, cached = extract(pack, video, params, cache_dir, cancel, progress)
    if cancel and cancel.is_set():
        raise InterruptedError("已停止")
    progress(.97, "计算行为时段")
    if pack.manifest["pipeline"] == "oviposition_v1":
        events = oviposition_events(data["tracks"], params, pack.manifest.get("behavior_label", "产卵"))
    else:
        events = presence_events(data["samples"], data["info"]["duration"], pack.manifest["labels"], params)
    for event in events:
        event.update(review="待复核", source="自动识别", note="",
                     original=dict(start=event["start"], end=event["end"], behavior=event["behavior"]))
    result = dict(video=str(Path(video).resolve()), status="完成" if events else "未检出（待复核）",
                  events=events, info=data["info"], model_name=pack.name, model_id=pack.identity,
                  model_path=str(pack.root), parameters=params, video_sha256=data["video_sha256"],
                  cached=cached, cache_file=str(Path(cache_dir) / (data["identity"] + ".json")) if cache_dir else "",
                  error="", video_review="待复核")
    progress(1, "完成")
    return result
