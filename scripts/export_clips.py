#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
export_clips.py  ——  导出带检测框的验证片段
============================================
人工核对模型判断是否正确。不导出整段（1-3 小时太慢），
只截取模型判断的产卵【起点】和【终点】前后各若干分钟，
画上检测框和时间戳，便于快速判断：
  - 起点片段：飞蝗是不是刚插入沙中开始产卵？
  - 终点片段：飞蝗是不是刚拔出离开？

用法：
  # 单个视频
  python export_clips.py --weights runs/detect/runs/v2/weights/best.pt \\
      --video "video/两种湿度下的产卵选择对照/对照1.mp4" \\
      --results results_control.csv --out clips

  # 批量导出若干个（随机抽查用）
  python export_clips.py --weights ... --video-dir "video/..." \\
      --results results_control.csv --out clips --limit 5

  # 导出整段（慢，仅在需要时用）
  python export_clips.py --weights ... --video xxx.mp4 \\
      --results ... --out clips --full
"""
import argparse
import csv
import random
from pathlib import Path

import cv2
from ultralytics import YOLO


def fmt_hms(sec):
    sec = int(round(sec))
    return f"{sec//3600}:{(sec%3600)//60:02d}:{sec%60:02d}"


def load_results(path):
    out = {}
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            out[r["video"].strip()] = (float(r["pred_start"]), float(r["pred_end"]))
    return out


def export_clip(model, video, out_path, t_start, t_end, label,
                imgsz, conf, sample_every=2):
    """
    导出 [t_start, t_end] 区间的带框视频。
    sample_every: 每隔几帧取一帧（降低帧率加快导出，产卵是慢行为不影响判断）
    """
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    f0 = max(0, int(t_start * fps))
    f1 = int(t_end * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, f0)

    out_fps = max(1.0, fps / sample_every)
    vw = cv2.VideoWriter(str(out_path),
                         cv2.VideoWriter_fourcc(*"mp4v"), out_fps, (w, h))

    n = 0
    fi = f0
    while fi < f1:
        ok, frame = cap.read()
        if not ok:
            break
        if (fi - f0) % sample_every == 0:
            r = model.predict(frame, imgsz=imgsz, conf=conf, verbose=False)[0]
            for b in r.boxes:
                x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                cf = float(b.conf)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 255), 2)
                cv2.putText(frame, f"{cf:.2f}", (x1, max(15, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)
            # 左上角打时间戳与标记
            t_now = fi / fps
            cv2.putText(frame, f"{fmt_hms(t_now)}  {label}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            vw.write(frame)
            n += 1
        fi += 1
    cap.release()
    vw.release()
    return n


def main():
    ap = argparse.ArgumentParser(description="导出带框验证片段")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--video", help="单个视频")
    ap.add_argument("--video-dir", help="批量：视频目录")
    ap.add_argument("--results", required=True, help="检测结果 csv")
    ap.add_argument("--out", default="clips")
    ap.add_argument("--margin", type=float, default=120,
                    help="起止点前后各取多少秒，默认 120=2 分钟")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.7)
    ap.add_argument("--sample-every", type=int, default=3,
                    help="每隔几帧取一帧，越大越快")
    ap.add_argument("--limit", type=int, help="批量时最多导出几个视频")
    ap.add_argument("--full", action="store_true",
                    help="导出整个产卵时段（慢），默认只导出起止点片段")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    results = load_results(args.results)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(args.weights)

    if args.video:
        videos = [Path(args.video)]
    else:
        videos = sorted(Path(args.video_dir).glob("*.mp4"))
        videos = [v for v in videos if v.stem in results]
        if args.limit and len(videos) > args.limit:
            random.seed(args.seed)
            videos = sorted(random.sample(videos, args.limit))
            print(f"随机抽取 {args.limit} 个视频核对：{[v.stem for v in videos]}\n")

    for v in videos:
        if v.stem not in results:
            print(f"[跳过] {v.stem} 无检测结果")
            continue
        ts, te = results[v.stem]
        print(f"=== {v.stem} ===")
        print(f"  模型判断产卵: {fmt_hms(ts)} ~ {fmt_hms(te)} "
              f"({(te-ts)/60:.1f} 分钟)")

        if args.full:
            op = out_dir / f"{v.stem}_full.mp4"
            n = export_clip(model, v, op, ts - args.margin, te + args.margin,
                            "OVIPOSITION", args.imgsz, args.conf,
                            args.sample_every)
            print(f"  → {op.name}（{n} 帧）")
        else:
            # 起点片段：判断是不是刚开始产卵
            op = out_dir / f"{v.stem}_start.mp4"
            n = export_clip(model, v, op, ts - args.margin, ts + args.margin,
                            "START", args.imgsz, args.conf, args.sample_every)
            print(f"  → {op.name}（{n} 帧，起点前后各 {args.margin/60:.0f} 分钟）")

            # 终点片段：判断是不是刚结束
            op = out_dir / f"{v.stem}_end.mp4"
            n = export_clip(model, v, op, te - args.margin, te + args.margin,
                            "END", args.imgsz, args.conf, args.sample_every)
            print(f"  → {op.name}（{n} 帧，终点前后各 {args.margin/60:.0f} 分钟）")

    print(f"\n全部导出到 {out_dir}/，下载到本地查看。")
    print("核对要点：")
    print("  START 片段 — 标记时刻飞蝗是否刚把腹部插入沙中？")
    print("  END   片段 — 标记时刻飞蝗是否刚拔出、开始活动？")
    print("  框是否准确套住飞蝗？有无漏检或框到沙子？")


if __name__ == "__main__":
    main()
