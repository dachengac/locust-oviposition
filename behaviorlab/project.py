"""Reviewable projects and CSV exports that retain automatic predictions."""
import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from .engine import atomic_json


def format_time(seconds):
    milliseconds = max(0, round(float(seconds) * 1000))
    s, ms = divmod(milliseconds, 1000)
    return f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}.{ms:03d}"


def parse_time(text):
    fields = str(text).strip().split(":")
    if not 1 <= len(fields) <= 3:
        raise ValueError("时间格式为 时:分:秒 或秒数")
    values = [float(f) for f in fields]
    if any(not math.isfinite(v) or v < 0 for v in values):
        raise ValueError("时间必须是非负有限数值")
    if len(values) > 1 and any(v >= 60 for v in values[1:]):
        raise ValueError("分钟与秒数须小于 60")
    return sum(v * 60 ** i for i, v in enumerate(reversed(values)))


def validate_interval(start, end, duration):
    if not 0 <= start < end <= duration:
        raise ValueError("必须满足 0 ≤ 起点 < 终点 ≤ 视频时长")


def save_project(path, results, model_path=""):
    atomic_json(path, dict(schema_version=1, saved_at=datetime.now(timezone.utc).isoformat(),
                          model_path=str(model_path), results=results))


def load_project(path):
    path = Path(path)
    if path.stat().st_size > 100 * 1024 * 1024:
        raise ValueError("项目文件过大")
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema_version") != 1 or not isinstance(data.get("results"), list):
        raise ValueError("不是有效的 BehaviorLab 项目")
    for result in data["results"]:
        if not isinstance(result.get("video"), str) or not isinstance(result.get("events"), list):
            raise ValueError("项目视频记录不完整")
        for event in result["events"]:
            validate_interval(event["start"], event["end"], result["info"]["duration"])
            if not isinstance(event.get("behavior"), str) or event.get("review") not in {"待复核", "已确认", "已修正", "已排除"}:
                raise ValueError("项目事件记录不完整")
    return data


def export_csv(path, results):
    headers = ["视频", "分析状态", "行为", "起点秒", "终点秒", "起点", "终点", "时长分钟",
               "复核状态", "来源", "备注", "原始行为", "原始起点秒", "原始终点秒",
               "支持轨迹数", "轨迹ID", "模型", "模型标识", "视频SHA256", "分析参数", "错误"]
    with open(path, "w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(headers)
        for result in results:
            for event in result["events"] or [None]:
                original = event.get("original", {}) if event else {}
                s, e = (event["start"], event["end"]) if event else (None, None)
                writer.writerow([result["video"], result["status"], event["behavior"] if event else "",
                                 s, e, format_time(s) if s is not None else "", format_time(e) if e is not None else "",
                                 round((e - s) / 60, 4) if s is not None else "",
                                 event["review"] if event else result.get("video_review", "待复核"),
                                 event["source"] if event else "", event.get("note", "") if event else "",
                                 original.get("behavior", ""), original.get("start", ""), original.get("end", ""),
                                 event.get("support", 0) if event else "", "+".join(map(str, event.get("track_ids", []))) if event else "",
                                 result.get("model_name", ""), result.get("model_id", ""), result.get("video_sha256", ""),
                                 json.dumps(result.get("parameters", {}), ensure_ascii=False, sort_keys=True), result.get("error", "")])
