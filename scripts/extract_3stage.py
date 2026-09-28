#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
extract_3stage.py  ——  三阶段抽帧（OpenCV，无需 ffmpeg）
========================================================
读 stages.csv，对每个视频的 探索/产卵/清洁 三段各均匀抽 N 帧，
自动归到 frames/{exploration,ovipositing,cleaning}/ 三个类别目录。

为什么"每段均匀抽固定 N 帧"：
  - 三个阶段时长差异极大（探索几十秒~90分钟、清洁仅几分钟），
    若按固定间隔抽，产卵段会淹没其它两类。固定 N 帧保证三类样本平衡。
  - 段内均匀采样天然去冗余（产卵几乎静止，密抽全是重复帧）。

用法：
  python extract_3stage.py \
      --videos video/Training_Set --stages stages.csv \
      --out frames --per-stage 40
"""
import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

# (类别名, stages.csv 里的起列, 止列)
STAGES = [
    ("exploration", "explore_start", "explore_end"),
    ("ovipositing", "ovi_start",     "ovi_end"),
    ("cleaning",    "clean_start",   "clean_end"),
]


def to_sec(t):
    """'1:32:53' / '0:18:51' -> 秒；空值或非法返回 None"""
    t = (t or "").strip()
    if not t:
        return None
    try:
        parts = [int(x) for x in t.split(":")]
    except ValueError:
        return None
    if len(parts) == 3:
        h, m, s = parts
    elif len(parts) == 2:
        h, m, s = 0, parts[0], parts[1]
    else:
        return None
    return h * 3600 + m * 60 + s


def main():
    ap = argparse.ArgumentParser(description="三阶段抽帧（OpenCV）")
    ap.add_argument("--videos", required=True, help="视频所在目录")
    ap.add_argument("--stages", required=True, help="stages.csv 路径")
    ap.add_argument("--out", default="frames", help="抽帧输出根目录")
    ap.add_argument("--per-stage", type=int, default=40,
                    help="每个视频每个阶段抽多少帧（默认 40）")
    ap.add_argument("--ext", default="mp4", help="视频扩展名")
    args = ap.parse_args()

    out = Path(args.out)
    for stage, _, _ in STAGES:
        (out / stage).mkdir(parents=True, exist_ok=True)

    with open(args.stages, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    total = 0
    for row in rows:
        vid = row["video"].strip()
        vpath = Path(args.videos) / f"{vid}.{args.ext}"
        if not vpath.exists():
            print(f"[跳过] 找不到视频 {vpath}")
            continue

        cap = cv2.VideoCapture(str(vpath))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        nframe = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if nframe <= 0:
            print(f"[跳过] {vid} 读不出帧（文件可能损坏）")
            cap.release()
            continue

        for stage, c0, c1 in STAGES:
            s0, s1 = to_sec(row.get(c0)), to_sec(row.get(c1))
            if s0 is None or s1 is None or s1 <= s0:
                continue  # 该阶段无数据（如某视频无清洁段）

            f0 = int(s0 * fps)
            f1 = min(int(s1 * fps), nframe - 1)
            frame_ids = np.linspace(f0, f1, args.per_stage).astype(int)

            saved = 0
            for k, fid in enumerate(frame_ids):
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(fid))
                ok, frame = cap.read()
                if ok:
                    fn = out / stage / f"{stage}_{vid}_{k:03d}.jpg"
                    cv2.imwrite(str(fn), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    saved += 1
            total += saved
            print(f"{vid} [{stage}] {s0/60:.1f}–{s1/60:.1f}min -> {saved} 帧")
        cap.release()

    print(f"\n完成，共抽 {total} 帧 -> {out}/")
    for stage, _, _ in STAGES:
        n = len(list((out / stage).glob("*.jpg")))
        print(f"  {stage:12s}: {n} 帧")
    print("\n下一步：把 frames/ 下三个目录上传 Roboflow，按目录名分别标注类别、框出飞蝗。")


if __name__ == "__main__":
    main()
