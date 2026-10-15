"""Versioned model contract; packs contain data, never executable plugins."""
from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import stat
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

DEFAULTS = dict(conf=0.7, sample_fps=1.0, win_sec=180.0,
                speed_percentile=25.0, max_drift=150.0, min_duration=900.0,
                frag_min_duration=900.0, stitch_gap=60.0, stitch_dist=100.0,
                min_track_len=20, merge_gap=120.0)
ADAPTERS = {"yolo_onnx": {"oviposition_v1", "detection_presence_v1"},
            "frame_classifier_onnx": {"classification_v1"}}


def file_sha256(path, cancel=None, progress=None):
    digest = hashlib.sha256()
    size = Path(path).stat().st_size
    done = 0
    with open(path, "rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            if cancel and cancel.is_set():
                raise InterruptedError("已停止")
            digest.update(block)
            done += len(block)
            if progress:
                progress(done / max(size, 1))
    return digest.hexdigest()


def validate_params(params):
    result = dict(DEFAULTS)
    if not isinstance(params, dict) or set(params) - set(DEFAULTS):
        raise ValueError("模型包包含未知分析参数")
    result.update(params)
    for key, value in result.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"参数 {key} 必须是有限数值")
        if value < 0 or (key in {"sample_fps", "win_sec", "min_track_len"} and value <= 0):
            raise ValueError(f"参数 {key} 超出范围")
    if not 0 < result["conf"] <= 1 or not 0 <= result["speed_percentile"] <= 100:
        raise ValueError("置信度或分位数超出范围")
    if result["sample_fps"] > 120 or result["min_track_len"] != int(result["min_track_len"]):
        raise ValueError("抽帧率或最短轨迹长度超出范围")
    return result


@dataclass(frozen=True)
class ModelPack:
    root: Path
    manifest: dict
    weights: Path
    digest: str

    @property
    def name(self):
        return self.manifest["name"]

    @property
    def params(self):
        return validate_params(self.manifest.get("parameters", {}))

    @property
    def identity(self):
        data = json.dumps(self.manifest, sort_keys=True, ensure_ascii=False).encode()
        return hashlib.sha256(data + self.digest.encode()).hexdigest()


def load_pack(root):
    root = Path(root).resolve()
    manifest_file = root / "manifest.json"
    if manifest_file.stat().st_size > 64 * 1024:
        raise ValueError("模型说明文件过大")
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise ValueError("不支持的模型包版本，需要 schema_version=1")
    if not isinstance(manifest.get("name"), str) or not manifest["name"].strip():
        raise ValueError("模型名称不能为空")
    adapter, pipeline = manifest.get("adapter"), manifest.get("pipeline")
    if adapter not in ADAPTERS or pipeline not in ADAPTERS[adapter]:
        raise ValueError("模型输入输出接口不兼容，需使用已支持的模型适配器")
    labels = manifest.get("labels")
    if not isinstance(labels, list) or not labels or len(labels) > 1000:
        raise ValueError("类别列表不合法")
    if any(not isinstance(label, str) or not label.strip() for label in labels) or len(set(labels)) != len(labels):
        raise ValueError("类别名称必须非空且不能重复")
    relative = manifest.get("weights", "")
    if not isinstance(relative, str) or not relative or "\\" in relative or ":" in relative:
        raise ValueError("模型文件路径不合法")
    rel = PurePosixPath(relative)
    if rel.is_absolute() or ".." in rel.parts:
        raise ValueError("模型文件必须位于模型包内部")
    weights = (root / relative).resolve()
    if not weights.is_relative_to(root) or weights.suffix.lower() != ".onnx" or not weights.is_file():
        raise ValueError("找不到模型包内部的 ONNX 文件")
    digest = file_sha256(weights)
    if not re.fullmatch(r"[0-9a-f]{64}", str(manifest.get("sha256", ""))) or digest != manifest["sha256"]:
        raise ValueError("模型校验失败：文件不完整或已被修改")
    size = manifest.get("input_size")
    if not isinstance(size, list) or len(size) != 2 or any(type(n) is not int or not 16 <= n <= 4096 for n in size):
        raise ValueError("input_size 必须为 [宽, 高]，范围 16–4096")
    validate_params(manifest.get("parameters", {}))
    if pipeline == "oviposition_v1":
        target = manifest.get("target_class", 0)
        if type(target) is not int or not 0 <= target < len(labels):
            raise ValueError("target_class 不合法")
        if not isinstance(manifest.get("behavior_label", "产卵"), str):
            raise ValueError("behavior_label 不合法")
    if adapter == "yolo_onnx":
        if manifest.get("output_layout", "channels_first") not in {"channels_first", "rows"}:
            raise ValueError("只支持不带 NMS 的 YOLO11/8 xywh 输出")
    else:
        if manifest.get("output_kind", "probabilities") not in {"probabilities", "logits"}:
            raise ValueError("分类输出必须是 probabilities 或 logits")
        for key, default in [("mean", [0, 0, 0]), ("std", [1, 1, 1])]:
            values = manifest.get(key, default)
            if not isinstance(values, list) or len(values) != 3 or any(type(n) not in (int, float) or not math.isfinite(n) for n in values):
                raise ValueError(f"{key} 必须为三个有限数值")
            if key == "std" and min(values) <= 0:
                raise ValueError("std 必须大于零")
    return ModelPack(root, manifest, weights, digest)


def install_pack(archive, destination):
    """Validate in a staging folder, then install atomically; reject zip slip/symlinks."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".import-", dir=destination) as staging:
        staging = Path(staging)
        with zipfile.ZipFile(archive) as bundle:
            infos = bundle.infolist()
            if len(infos) > 100 or sum(i.file_size for i in infos) > 512 * 1024 * 1024:
                raise ValueError("模型包超过 512 MB 或文件过多")
            seen = set()
            for item in infos:
                name = item.filename
                path = PurePosixPath(name)
                if (path.is_absolute() or ".." in path.parts or "\\" in name or ":" in name
                        or not path.parts or name.casefold() in seen
                        or stat.S_ISLNK(item.external_attr >> 16)):
                    raise ValueError("模型包包含不安全或重复的文件路径")
                seen.add(name.casefold())
                target = (staging / name).resolve()
                if not target.is_relative_to(staging.resolve()):
                    raise ValueError("模型包路径越界")
                if item.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with bundle.open(item) as source, open(target, "wb") as out:
                        shutil.copyfileobj(source, out)
        pack = load_pack(staging)
        installed = destination / pack.identity[:24]
        if installed.exists():
            return load_pack(installed)
        staging.rename(installed)
        # TemporaryDirectory's cleanup tolerates the renamed staging directory.
        return load_pack(installed)


def create_pack(weights, output, *, name="自定义识别模型", labels=None,
                pipeline="oviposition_v1", input_size=(640, 640), behavior_label="产卵",
                parameters=None, adapter="yolo_onnx", extra=None):
    weights, output = Path(weights), Path(output)
    if weights.suffix.lower() != ".onnx" or not weights.is_file():
        raise ValueError("请选择已导出的 ONNX 模型")
    if output.resolve() == weights.resolve() or output.exists():
        raise ValueError("输出文件已存在，请选择新的模型包文件名")
    manifest = dict(schema_version=1, name=name, version="1.0.0", adapter=adapter,
                    pipeline=pipeline, weights="model.onnx", sha256=file_sha256(weights),
                    labels=labels or ["locust"], input_size=list(input_size),
                    parameters=parameters or {}, behavior_label=behavior_label,
                    target_class=0, output_layout="channels_first")
    if extra:
        manifest.update(extra)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".pack-", dir=output.parent) as staging:
        stage = Path(staging)
        shutil.copy2(weights, stage / "model.onnx")
        (stage / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        load_pack(stage)
        output.parent.mkdir(parents=True, exist_ok=True)
        try:
            with zipfile.ZipFile(output, "x", zipfile.ZIP_DEFLATED, compresslevel=6) as bundle:
                bundle.write(stage / "manifest.json", "manifest.json")
                bundle.write(stage / "model.onnx", "model.onnx")
        except Exception:
            if output.exists():
                output.unlink()
            raise
    return output
