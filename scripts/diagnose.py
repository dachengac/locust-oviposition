#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
diagnose.py  ——  诊断单个视频的判断误差来源
============================================
读轨迹缓存，把净位移随时间的变化用 ASCII 图打印出来，
叠加人工标注的产卵时段，一眼看出模型为什么判断偏差。

用法：
  python diagnose.py --cache track_cache --stages stages.csv --video train01
  python diagnose.py --cache track_cache --stages stages.csv --video train10 --win-sec 180
"""
import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np


def load_algo(script="scripts/detect_oviposition.py"):
    src = Path(script).read_text(encoding="utf-8")
    ns = {"np": np}
    for fn in ["fmt_hms", "to_sec", "compute_speed", "adaptive_threshold",
               "merge_close", "find_still_periods"]:
        m = re.search(rf"^def {fn}\(.*?(?=^def |\Z)", src, re.S | re.M)
        if m:
            exec(m.group(0), ns)
    return ns


def ascii_plot(times, values, thr, truth, width=100, height=20):
    """把净位移曲线画成 ASCII 图，标出阈值线和人工标注的产卵区间。"""
    if len(values) == 0:
        return
    t0, t1 = times[0], times[-1]
    # 按列聚合
    cols = np.linspace(t0, t1, width + 1)
    colv = []
    for i in range(width):
        m = (times >= cols[i]) & (times < cols[i + 1])
        colv.append(np.median(values[m]) if m.any() else np.nan)
    colv = np.array(colv)

    vmax = np.nanpercentile(values, 97)
    vmax = max(vmax, thr * 1.5, 1e-6)

    print(f"\n  净位移曲线（纵轴 0~{vmax:.1f} px/s，横轴 {times[0]/60:.0f}~{times[-1]/60:.0f} 分钟）")
    print(f"  '#'=曲线  '-'=阈值{thr:.2f}  下方 '='=人工标注产卵段\n")

    for row in range(height, -1, -1):
        y = vmax * row / height
        line = ""
        for c in range(width):
            v = colv[c]
            if np.isnan(v):
                line += " "
            elif abs(v - y) <= vmax / height / 2:
                line += "#"
            elif abs(thr - y) <= vmax / height / 2:
                line += "-"
            else:
                line += " "
        label = f"{y:6.1f} |"
        print(f"  {label}{line}")

    # 人工标注区间
    if truth:
        ts, te = truth
        bar = ""
        for c in range(width):
            ct = (cols[c] + cols[c + 1]) / 2
            bar += "=" if ts <= ct <= te else " "
        print(f"  {'人工':>6s} |{bar}")

    # 时间刻度
    ticks = ""
    for c in range(width):
        ct = (cols[c] + cols[c + 1]) / 2
        ticks += "|" if c % 10 == 0 else " "
    print(f"  {'':6s} |{ticks}")
    labs = ""
    for c in range(0, width, 10):
        ct = (cols[c] + cols[c + 1]) / 2
        labs += f"{ct/60:<10.0f}"
    print(f"  {'分钟':>6s} |{labs}")


def main():
    ap = argparse.ArgumentParser(description="诊断单视频误差来源")
    ap.add_argument("--cache", default="track_cache")
    ap.add_argument("--stages", required=True)
    ap.add_argument("--video", required=True)
    ap.add_argument("--script", default="scripts/detect_oviposition.py")
    ap.add_argument("--win-sec", type=float, default=180)
    ap.add_argument("--min-duration", type=float, default=900)
    ap.add_argument("--max-drift", type=float, default=150)
    ap.add_argument("--min-track-len", type=int, default=20)
    args = ap.parse_args()

    algo = load_algo(args.script)

    truth = None
    with open(args.stages, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["video"].strip() == args.video:
                s = algo["to_sec"](r.get("ovi_start"))
                e = algo["to_sec"](r.get("ovi_end"))
                if s is not None and e is not None:
                    truth = (s, e)

    jf = next(Path(args.cache).glob(f"{args.video}_*.json"), None)
    if jf is None:
        raise SystemExit(f"找不到 {args.video} 的缓存")
    data = json.loads(jf.read_text(encoding="utf-8"))
    tracks = {int(k): [tuple(p) for p in v] for k, v in data["tracks"].items()}

    print(f"=== {args.video} ===")
    if truth:
        print(f"人工标注产卵: {algo['fmt_hms'](truth[0])} ~ {algo['fmt_hms'](truth[1])} "
              f"({(truth[1]-truth[0])/60:.1f} 分钟)")
    print(f"\n共 {len(tracks)} 条轨迹：")

    all_speed, per_track = [], {}
    for tid, traj in sorted(tracks.items()):
        dur = traj[-1][0] - traj[0][0] if traj else 0
        print(f"  track {tid:>5d}: {len(traj):>5d} 点, "
              f"{algo['fmt_hms'](traj[0][0])} ~ {algo['fmt_hms'](traj[-1][0])} "
              f"(跨度 {dur/60:.1f} 分钟)")
        if len(traj) < args.min_track_len:
            continue
        t, sp = algo["compute_speed"](traj, args.win_sec)
        if len(sp) == 0:
            continue
        pos = np.array([(x, y) for _, x, y in traj])[:len(sp)]
        per_track[tid] = (t, sp, pos)
        all_speed.append(sp)

    if not per_track:
        raise SystemExit("无有效轨迹")

    thr = algo["adaptive_threshold"](np.concatenate(all_speed), 25)
    print(f"\n自适应阈值 = {thr:.2f} px/s")

    # 找出最长的那条轨迹（通常是产卵个体）
    main_tid = max(per_track, key=lambda k: len(per_track[k][0]))
    print(f"\n主轨迹 track {main_tid}（点数最多）：")
    t, sp, pos = per_track[main_tid]
    ascii_plot(t, sp, thr, truth)

    print(f"\n各轨迹检出的静止时段（min_dur={args.min_duration/60:.0f}min, "
          f"max_drift={args.max_drift}）：")
    for tid, (t, sp, pos) in per_track.items():
        evs = algo["find_still_periods"](t, sp, thr, args.min_duration,
                                         pos, args.max_drift)
        for s, e, d, drift in evs:
            mark = ""
            if truth:
                mark = f"  [起点差 {(s-truth[0])/60:+.1f}m, 终点差 {(e-truth[1])/60:+.1f}m]"
            print(f"  track {tid:>5d}: {algo['fmt_hms'](s)} ~ {algo['fmt_hms'](e)} "
                  f"({d/60:.1f}min, 漂移{drift:.0f}px){mark}")


if __name__ == "__main__":
    main()
