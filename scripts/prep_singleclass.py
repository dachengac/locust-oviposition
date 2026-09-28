#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prep_singleclass.py  ——  Roboflow 导出 → 单类 locust 训练集
============================================================
做三件事：
  1. 把所有标注的类别 id 统一为 0（单类 locust），忽略 Roboflow 里混乱的类别
  2. 按【视频】分组划分 train/val —— 同一视频的帧只进一边，杜绝帧泄漏
     （Roboflow 默认随机划分会让相邻帧同时进 train 和 val，导致精度虚高）
  3. 生成正确的 locust.yaml

用法：
  python prep_singleclass.py --in labeled_raw/train --out dataset --val-ratio 0.2

输入目录需含 images/ 与 labels/ 两个子目录。
"""
import argparse
import random
import re
import shutil
from collections import defaultdict
from pathlib import Path

random.seed(42)
# 从 Roboflow 文件名里提取视频标识，如
# exploration_train03_030_jpg.rf.<hash>.jpg -> train03
VID_PAT = re.compile(r"(train\d+)")


def video_key(fname):
    m = VID_PAT.search(fname)
    return m.group(1) if m else "unknown"


def convert_label(src, dst):
    """复制标注并把类别 id 全部改为 0；返回框数。"""
    lines = []
    for line in src.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) >= 5:            # cls cx cy w h
            parts[0] = "0"             # 统一为单类 locust
            lines.append(" ".join(parts))
    dst.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return len(lines)


def main():
    ap = argparse.ArgumentParser(description="单类 locust 数据集准备")
    ap.add_argument("--in", dest="inp", required=True,
                    help="Roboflow 导出的目录（含 images/ labels/）")
    ap.add_argument("--out", default="dataset")
    ap.add_argument("--val-ratio", type=float, default=0.2)
    ap.add_argument("--val-videos", default="",
                    help="显式指定验证集视频，逗号分隔，如 train03,train07；"
                         "不指定则按 --val-ratio 随机选")
    args = ap.parse_args()

    inp, out = Path(args.inp), Path(args.out)
    img_dir, lbl_dir = inp / "images", inp / "labels"
    imgs = sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")))
    if not imgs:
        raise SystemExit(f"在 {img_dir} 没找到图片")

    # 按视频分组
    by_video = defaultdict(list)
    for img in imgs:
        by_video[video_key(img.name)].append(img)

    videos = sorted(by_video)
    print(f"共 {len(imgs)} 张图，来自 {len(videos)} 个视频：")
    for v in videos:
        print(f"  {v}: {len(by_video[v])} 张")

    # 划分：优先用显式指定；否则按比例随机选；只有 1 个视频时退化为按文件划分
    if args.val_videos.strip():
        val_videos = {v.strip() for v in args.val_videos.split(",") if v.strip()}
        unknown = val_videos - set(videos)
        if unknown:
            raise SystemExit(f"指定的验证集视频不存在：{sorted(unknown)}\n可选：{videos}")
        if val_videos == set(videos):
            raise SystemExit("验证集不能包含全部视频，训练集会为空")
        print(f"\n验证集视频（手动指定）：{sorted(val_videos)}")
        split_of = lambda img: "val" if video_key(img.name) in val_videos else "train"
    elif len(videos) >= 2:
        shuffled = videos[:]
        random.shuffle(shuffled)
        n_val = max(1, round(len(shuffled) * args.val_ratio))
        val_videos = set(shuffled[:n_val])
        print(f"\n验证集视频：{sorted(val_videos)}")
        split_of = lambda img: "val" if video_key(img.name) in val_videos else "train"
    else:
        print("\n⚠️ 只有 1 个视频，无法按视频划分，退化为随机划分。")
        print("   验证集与训练集来自同一视频，精度会偏乐观，仅供跑通流程。")
        shuffled = imgs[:]
        random.shuffle(shuffled)
        n_val = max(1, int(len(shuffled) * args.val_ratio))
        val_set = set(shuffled[:n_val])
        split_of = lambda img: "val" if img in val_set else "train"

    for split in ("train", "val"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)

    counts = {"train": 0, "val": 0}
    boxes = {"train": 0, "val": 0}
    for img in imgs:
        split = split_of(img)
        shutil.copy2(img, out / "images" / split / img.name)
        src_lbl = lbl_dir / (img.stem + ".txt")
        dst_lbl = out / "labels" / split / (img.stem + ".txt")
        boxes[split] += convert_label(src_lbl, dst_lbl) if src_lbl.exists() else 0
        if not src_lbl.exists():
            dst_lbl.touch()            # 无标注 = 纯背景负样本
        counts[split] += 1

    print(f"\ntrain: {counts['train']} 张 / {boxes['train']} 框")
    print(f"val  : {counts['val']} 张 / {boxes['val']} 框")

    yaml = out / "locust.yaml"
    yaml.write_text(
        f"path: {out.resolve()}\n"
        "train: images/train\n"
        "val: images/val\n\n"
        "nc: 1\n"
        "names:\n"
        "  0: locust\n",
        encoding="utf-8")
    print(f"\n已写出 {yaml}")
    print(f"下一步：sbatch scripts/04_train.sh（记得把 data 指向 {yaml}）")


if __name__ == "__main__":
    main()
