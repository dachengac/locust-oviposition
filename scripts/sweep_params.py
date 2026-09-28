#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sweep_params.py  ——  时序参数网格搜索
=====================================
读 detect_oviposition.py 缓存下来的轨迹（track_cache/*.json），
在不重跑检测追踪的前提下，把各种时序参数组合都试一遍，
按与人工标注 stages.csv 的误差排序，找出最优参数。

前提：先用 detect_oviposition.py 跑一遍生成缓存（检测追踪只需一次）。

用法：
  # 单视频调参
  python sweep_params.py --cache track_cache --stages stages.csv --video train01

  # 全部视频一起评估（推荐，避免过拟合单个视频）
  python sweep_params.py --cache track_cache --stages stages.csv
"""
import argparse
import csv
import itertools
import json
from pathlib import Path

import numpy as np

# 复用主脚本的算法实现，避免两处逻辑不一致
import importlib.util


def load_algo(script="scripts/detect_oviposition.py"):
    """只加载算法函数，跳过 ultralytics 依赖。"""
    import re
    src = Path(script).read_text(encoding="utf-8")
    ns = {"np": np}
    for fn in ["fmt_hms", "to_sec", "compute_speed", "adaptive_threshold",
               "merge_close", "stitch_segments", "find_still_periods"]:
        m = re.search(rf"^def {fn}\(.*?(?=^def |\Z)", src, re.S | re.M)
        if m:
            exec(m.group(0), ns)
    return ns


def analyze_cached(tracks, algo, win_sec, pct, min_dur, max_drift, min_track_len,
                   frag_min=600, stitch_gap=120, stitch_dist=100):
    """
    对一份缓存轨迹跑完整时序分析（与 detect_oviposition.py 逻辑一致）：
    收集碎片 → 轨迹拼接 → 证据密度选择。返回 (start, end, dur) 或 None。
    """
    all_speed, per_track = [], {}
    for tid, traj in tracks.items():
        if len(traj) < min_track_len:
            continue
        t, sp = algo["compute_speed"](traj, win_sec)
        if len(sp) == 0:
            continue
        pos = np.array([(x, y) for _, x, y in traj])[:len(sp)]
        per_track[tid] = (t, sp, pos)
        all_speed.append(sp)
    if not per_track:
        return None

    thr = algo["adaptive_threshold"](np.concatenate(all_speed), pct)

    # 收集碎片：门槛比 min_dur 低，让被 ID 断裂切碎的短段也进入拼接
    raw_segs = []
    for tid, (t, sp, pos) in per_track.items():
        fm = min(min_dur, frag_min)
        for (s, e, d, drift) in algo["find_still_periods"](
                t, sp, thr, fm, pos, max_drift):
            i0 = int(np.searchsorted(t, s))
            i1 = min(int(np.searchsorted(t, e)), len(pos) - 1)
            raw_segs.append((s, e, d, drift, tid, tuple(pos[i0]), tuple(pos[i1])))
    if not raw_segs:
        return None

    stitched = algo["stitch_segments"](raw_segs, stitch_gap, stitch_dist, max_drift)
    cands = [(s, e, d, tids) for s, e, d, drift, tids in stitched if d >= min_dur]
    if not cands:
        return None

    # 证据密度：真正的产卵时段应有多条轨迹共同支持
    def support(s, e):
        n = 0
        for rs, re_, rd, _, _, _, _ in raw_segs:
            ov = min(e, re_) - max(s, rs)
            if ov > 0 and ov >= 0.5 * min(rd, e - s):
                n += 1
        return n

    scored = sorted(((support(s, e), d, s, e) for s, e, d, _ in cands),
                    reverse=True)
    _, d, s, e = scored[0]
    return s, e, d


def main():
    ap = argparse.ArgumentParser(description="时序参数网格搜索")
    ap.add_argument("--cache", default="track_cache", help="轨迹缓存目录")
    ap.add_argument("--stages", required=True, help="stages.csv")
    ap.add_argument("--script", default="scripts/detect_oviposition.py")
    ap.add_argument("--video", help="只评估某个视频，如 train01；不填则全部")
    ap.add_argument("--top", type=int, default=15, help="显示前几名")
    ap.add_argument("--min-track-len", type=int, default=20)
    args = ap.parse_args()

    algo = load_algo(args.script)

    # 读人工标注
    truth = {}
    with open(args.stages, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            s = algo["to_sec"](r.get("ovi_start"))
            e = algo["to_sec"](r.get("ovi_end"))
            if s is not None and e is not None:
                truth[r["video"].strip()] = (s, e)

    # 读所有缓存轨迹
    cached = {}
    for jf in sorted(Path(args.cache).glob("*.json")):
        stem = jf.stem.split("_fps")[0]
        if args.video and stem != args.video:
            continue
        if stem not in truth:
            continue
        with open(jf, encoding="utf-8") as f:
            data = json.load(f)
        cached[stem] = {int(k): [tuple(p) for p in v]
                        for k, v in data["tracks"].items()}

    if not cached:
        raise SystemExit(f"{args.cache} 里没有可用缓存（或与 stages.csv 对不上）。\n"
                         f"先跑 detect_oviposition.py 生成缓存。")
    print(f"载入 {len(cached)} 个视频的轨迹缓存：{sorted(cached)}\n")

    # 参数网格
    grid = {
        "win_sec":     [180, 240],
        "pct":         [25],
        "max_drift":   [150],
        "min_dur":     [900],
        "frag_min":    [600, 900, 1200],
        "stitch_gap":  [60, 120, 180],
        "stitch_dist": [60, 100, 150],
    }
    combos = list(itertools.product(*grid.values()))
    print(f"共 {len(combos)} 种参数组合，评估中…\n")

    results = []
    for win_sec, pct, max_drift, min_dur, frag_min, sgap, sdist in combos:
        errs, n_hit = [], 0
        for v, tracks in cached.items():
            ts, te = truth[v]
            out = analyze_cached(tracks, algo, win_sec, pct, min_dur,
                                 max_drift, args.min_track_len,
                                 frag_min, sgap, sdist)
            if out is None:
                errs.append((np.nan, np.nan, np.nan))
                continue
            s, e, d = out
            errs.append((abs(s - ts), abs(e - te), abs(d - (te - ts))))
            n_hit += 1
        arr = np.array(errs, dtype=float)
        valid = ~np.isnan(arr[:, 0])
        if valid.sum() == 0:
            continue
        results.append({
            "win_sec": win_sec, "pct": pct, "max_drift": max_drift,
            "frag_min": frag_min, "stitch_gap": sgap, "stitch_dist": sdist,
            "n_hit": n_hit, "n_total": len(cached),
            "start_err": np.nanmean(arr[:, 0]) / 60,
            "end_err": np.nanmean(arr[:, 1]) / 60,
            "dur_err": np.nanmean(arr[:, 2]) / 60,
        })

    # 排序：优先检出率高，其次起点+终点误差小
    results.sort(key=lambda r: (-r["n_hit"], r["start_err"] + r["end_err"]))

    print(f"{'win':>4s} {'frag':>5s} {'gap':>4s} {'dist':>5s} {'检出':>7s} "
          f"{'起点误差':>9s} {'终点误差':>9s} {'时长误差':>9s}")
    print("-" * 66)
    for r in results[:args.top]:
        print(f"{r['win_sec']:>4d} {r['frag_min']:>5.0f} {r['stitch_gap']:>4.0f} "
              f"{r['stitch_dist']:>5.0f} {r['n_hit']:>3d}/{r['n_total']:<3d} "
              f"{r['start_err']:>8.1f}m {r['end_err']:>8.1f}m {r['dur_err']:>8.1f}m")

    if results:
        b = results[0]
        print(f"\n最优参数：--win-sec {b['win_sec']} "
              f"--frag-min-duration {b['frag_min']:.0f} "
              f"--stitch-gap {b['stitch_gap']:.0f} "
              f"--stitch-dist {b['stitch_dist']:.0f}")
        print(f"检出 {b['n_hit']}/{b['n_total']}，"
              f"平均起点误差 {b['start_err']:.1f} 分钟、终点误差 {b['end_err']:.1f} 分钟")


if __name__ == "__main__":
    main()
