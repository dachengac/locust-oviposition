#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
app.py  ——  飞蝗产卵行为自动分析工具（图形界面）
================================================
把已验证的产卵检测流程包装成浏览器界面，供不熟悉命令行的人使用。

验证精度：产卵起点误差 2.3 分钟、终点 5.0 分钟（12 个视频对照人工标注）

适用范围：静止型长时程行为（产卵）。
不适用于精细动作区分（如探索 vs 清洁）——检测框特征不足以区分，
详见 WORKLOG.md 2026-09-11 的记录。

启动：
  python app.py                      # 本机访问 http://127.0.0.1:7860
  python app.py --share              # 生成公网链接（临时，供他人访问）
  python app.py --port 8080
"""
import argparse
import csv
import sys
import tempfile
import traceback
from pathlib import Path

import gradio as gr

# 复用已验证的核心逻辑
sys.path.insert(0, str(Path(__file__).parent))
try:
    from scripts.detect_oviposition import (extract_tracks, compute_speed,
                                    adaptive_threshold, find_still_periods,
                                    stitch_segments, fmt_hms)
except ImportError as e:
    raise SystemExit(f"无法加载 scripts/detect_oviposition.py：{e}")

import numpy as np
from ultralytics import YOLO

# 定版参数（2026-09-09 在 12 个视频上验证）
DEFAULTS = dict(
    conf=0.7, imgsz=640, sample_fps=1.0,
    win_sec=180, speed_percentile=25, max_drift=150,
    min_duration=900, frag_min_duration=900,
    stitch_gap=60, stitch_dist=100, min_track_len=20,
)

_MODEL = {"path": None, "obj": None}


def get_model(weights):
    """懒加载模型，避免每次分析都重新载入。"""
    if _MODEL["path"] != weights or _MODEL["obj"] is None:
        _MODEL["obj"] = YOLO(weights)
        _MODEL["path"] = weights
    return _MODEL["obj"]


def analyze_one(model, video, params, cache_dir, progress=None):
    """分析单个视频，返回 (起点秒, 终点秒, 时长分钟) 或 None。"""
    tracks, fps = extract_tracks(model, video, params["sample_fps"],
                                 params["imgsz"], params["conf"], cache_dir)
    if not tracks:
        return None

    all_speed, per_track = [], {}
    for tid, traj in tracks.items():
        if len(traj) < params["min_track_len"]:
            continue
        t, sp = compute_speed(traj, params["win_sec"])
        if len(sp) == 0:
            continue
        pos = np.array([(p[1], p[2]) for p in traj])[:len(sp)]
        per_track[tid] = (t, sp, pos)
        all_speed.append(sp)
    if not per_track:
        return None

    thr = adaptive_threshold(np.concatenate(all_speed), params["speed_percentile"])

    raw = []
    for tid, (t, sp, pos) in per_track.items():
        fm = min(params["min_duration"], params["frag_min_duration"])
        for (s, e, d, drift) in find_still_periods(t, sp, thr, fm, pos,
                                                   params["max_drift"]):
            i0 = int(np.searchsorted(t, s))
            i1 = min(int(np.searchsorted(t, e)), len(pos) - 1)
            raw.append((s, e, d, drift, tid, tuple(pos[i0]), tuple(pos[i1])))
    if not raw:
        return None

    stitched = stitch_segments(raw, params["stitch_gap"], params["stitch_dist"],
                               params["max_drift"])
    cands = [(s, e, d, tids) for s, e, d, _, tids in stitched
             if d >= params["min_duration"]]
    if not cands:
        return None

    def support(s, e):
        n = 0
        for rs, re_, rd, _, _, _, _ in raw:
            ov = min(e, re_) - max(s, rs)
            if ov > 0 and ov >= 0.5 * min(rd, e - s):
                n += 1
        return n

    scored = sorted(((support(s, e), d, s, e) for s, e, d, _ in cands),
                    reverse=True)
    sup, d, s, e = scored[0]
    return s, e, d / 60, sup


def run_analysis(video_files, weights, min_dur_min, conf, sample_fps,
                 use_cache, progress=gr.Progress()):
    """界面主入口：批量分析上传的视频。"""
    if not video_files:
        return None, "请先上传视频文件。", None

    try:
        model = get_model(weights)
    except Exception as e:
        return None, f"❌ 模型加载失败：{e}\n检查权重路径是否正确。", None

    params = dict(DEFAULTS)
    params.update(conf=float(conf), sample_fps=float(sample_fps),
                  min_duration=float(min_dur_min) * 60)
    cache_dir = "track_cache_app" if use_cache else None

    rows, logs = [], []
    n = len(video_files)
    for i, vf in enumerate(video_files):
        path = Path(vf.name if hasattr(vf, "name") else vf)
        progress((i / n), desc=f"分析中 {i+1}/{n}：{path.name}")
        try:
            out = analyze_one(model, path, params, cache_dir)
        except Exception:
            logs.append(f"❌ {path.name}：分析出错\n{traceback.format_exc(limit=2)}")
            rows.append([path.stem, "出错", "", "", ""])
            continue

        if out is None:
            logs.append(f"○ {path.name}：未检出产卵"
                        f"（无持续 ≥{min_dur_min:.0f} 分钟的静止时段）")
            rows.append([path.stem, "未检出", "", "", ""])
        else:
            s, e, dmin, sup = out
            logs.append(f"✓ {path.name}：{fmt_hms(s)} ~ {fmt_hms(e)}"
                        f"（{dmin:.1f} 分钟，支持轨迹 {sup} 条）")
            rows.append([path.stem, "产卵", fmt_hms(s), fmt_hms(e), f"{dmin:.1f}"])

    progress(1.0, desc="完成")

    # 写出 csv 供下载
    tmp = Path(tempfile.gettempdir()) / "oviposition_results.csv"
    with open(tmp, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["视频", "结果", "产卵起点", "产卵终点", "产卵时长(分钟)"])
        w.writerows(rows)

    n_ovi = sum(1 for r in rows if r[1] == "产卵")
    summary = (f"分析完成：{n} 个视频，检出产卵 {n_ovi} 个"
               f"（{100*n_ovi/n:.0f}%）\n\n" + "\n".join(logs))
    return rows, summary, str(tmp)


def build_ui(default_weights):
    with gr.Blocks(title="飞蝗产卵行为自动分析") as demo:
        gr.Markdown(
            "# 飞蝗产卵行为自动分析工具\n"
            "上传视频 → 自动输出每段产卵的起止时刻与时长。\n\n"
            "**验证精度**：产卵起点误差 2.3 分钟、终点 5.0 分钟"
            "（12 个视频对照人工标注，检出率 12/12）"
        )

        with gr.Row():
            with gr.Column(scale=1):
                videos = gr.File(label="视频文件（可多选）",
                                 file_count="multiple",
                                 file_types=[".mp4", ".avi", ".mov", ".mkv"])
                weights = gr.Textbox(label="模型权重路径", value=default_weights)

                with gr.Accordion("高级参数（一般无需改动）", open=False):
                    min_dur = gr.Slider(5, 60, value=15, step=1,
                                        label="最短产卵时长（分钟）",
                                        info="短于此时长的静止时段不算产卵")
                    conf = gr.Slider(0.3, 0.95, value=0.7, step=0.05,
                                     label="检测置信度阈值",
                                     info="调低会检出更多目标，但误检增加")
                    fps = gr.Slider(0.2, 2.0, value=1.0, step=0.1,
                                    label="抽帧率（帧/秒）",
                                    info="产卵是慢行为，1 fps 足够；调低更快")
                    cache = gr.Checkbox(value=True, label="启用轨迹缓存",
                                        info="同一视频再次分析时秒出结果")

                btn = gr.Button("开始分析", variant="primary", size="lg")

            with gr.Column(scale=2):
                table = gr.Dataframe(
                    headers=["视频", "结果", "产卵起点", "产卵终点", "产卵时长(分钟)"],
                    label="分析结果", wrap=True)
                download = gr.File(label="下载结果 CSV")
                log = gr.Textbox(label="运行日志", lines=12, max_lines=25)

        btn.click(run_analysis,
                  inputs=[videos, weights, min_dur, conf, fps, cache],
                  outputs=[table, log, download])

        gr.Markdown(
            "---\n"
            "### 适用范围与限制\n"
            "- **适用**：静止型长时程行为（产卵）。原理是检测飞蝗→追踪→"
            "识别「长时间不离开原地」的时段\n"
            "- **不适用**：精细动作的区分（如探索 vs 清洁）。"
            "检测框特征不足以区分二者，需关键点级信息\n"
            "- **已知局限**：探索期若长时间原地停留，可能被误判为产卵起点；"
            "产卵末期追踪丢失会导致终点偏早\n"
            "- 首次分析每个视频需数分钟（检测+追踪）；启用缓存后再次分析秒级完成"
        )
    return demo


def main():
    ap = argparse.ArgumentParser(description="飞蝗产卵分析工具")
    ap.add_argument("--weights", default="runs/detect/runs/v2/weights/best.pt")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true", help="生成公网访问链接")
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()

    if not Path(args.weights).exists():
        print(f"⚠️ 权重不存在：{args.weights}（可在界面中修改路径）")

    demo = build_ui(args.weights)
    demo.launch(server_name=args.host, server_port=args.port,
                share=args.share, inbrowser=False)


if __name__ == "__main__":
    main()
