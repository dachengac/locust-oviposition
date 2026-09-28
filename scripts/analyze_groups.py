#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze_groups.py  ——  两组行为指标统计比较
============================================
把 results_control.csv 与 results_antenna.csv 合并，计算行为指标，
做组间统计检验，输出汇总表与箱线图。

行为指标：
  - 产卵率：该组有多少比例的个体最终产卵（未检出 = 未产卵）
  - 产卵潜伏期：视频开始到产卵起点的时间（反映决策快慢）
  - 产卵时长：产卵持续时间

检验方法：
  - 产卵率 → Fisher 精确检验（样本量小，优于卡方）
  - 潜伏期 / 时长 → Mann-Whitney U（不假设正态，样本量小时更稳健）
    同时报告 t 检验以便对照

用法：
  python analyze_groups.py --control results_control.csv \\
      --antenna results_antenna.csv --video-dir video --out group_stats
"""
import argparse
import csv
from pathlib import Path

import numpy as np


def read_results(path, group):
    """读结果 csv，返回 {video: {...}}"""
    rows = {}
    if not Path(path).exists():
        print(f"[警告] 找不到 {path}")
        return rows
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows[r["video"]] = {
                "video": r["video"], "group": group,
                "start": float(r["pred_start"]),
                "end": float(r["pred_end"]),
                "duration_min": float(r["pred_duration_min"]),
                "ovipositied": 1,
            }
    return rows


def list_videos(d):
    """列出目录下所有视频名（不含扩展名）"""
    if not d or not Path(d).exists():
        return []
    return sorted(p.stem for p in Path(d).glob("*.mp4"))


def fisher_exact_2x2(a, b, c, d):
    """
    Fisher 精确检验（双尾）。样本量小时优于卡方。
    表格:  [[a, b], [c, d]]
    """
    from math import comb
    n = a + b + c + d
    r1, r2 = a + b, c + d
    c1 = a + c

    def p_of(x):
        # 超几何分布概率
        if x < 0 or x > min(r1, c1) or (c1 - x) > r2:
            return 0.0
        return comb(r1, x) * comb(r2, c1 - x) / comb(n, c1)

    p_obs = p_of(a)
    total = 0.0
    for x in range(0, min(r1, c1) + 1):
        p = p_of(x)
        if p <= p_obs + 1e-12:
            total += p
    return min(1.0, total)


def mannwhitney_u(x, y):
    """Mann-Whitney U 检验（正态近似，含连续性校正与并列秩校正）。"""
    x, y = np.asarray(x, float), np.asarray(y, float)
    nx, ny = len(x), len(y)
    if nx == 0 or ny == 0:
        return np.nan, np.nan
    allv = np.concatenate([x, y])
    order = np.argsort(allv)
    ranks = np.empty(len(allv), float)
    ranks[order] = np.arange(1, len(allv) + 1)
    # 并列秩取平均
    uniq, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    for i, c in enumerate(cnt):
        if c > 1:
            ranks[inv == i] = ranks[inv == i].mean()
    r1 = ranks[:nx].sum()
    u1 = r1 - nx * (nx + 1) / 2
    u2 = nx * ny - u1
    u = min(u1, u2)
    mu = nx * ny / 2
    # 并列校正
    tie = sum(c ** 3 - c for c in cnt if c > 1)
    n = nx + ny
    sigma = np.sqrt(nx * ny / 12 * ((n + 1) - tie / (n * (n - 1))))
    if sigma == 0:
        return u, np.nan
    z = (abs(u - mu) - 0.5) / sigma
    # 正态双尾 p
    p = 2 * (1 - 0.5 * (1 + np.math.erf(z / np.sqrt(2)))) if hasattr(np, "math") \
        else 2 * (1 - _norm_cdf(z))
    return u, min(1.0, p)


def _norm_cdf(z):
    from math import erf, sqrt
    return 0.5 * (1 + erf(z / sqrt(2)))


def welch_t(x, y):
    """Welch t 检验（不假设方差齐性）。"""
    x, y = np.asarray(x, float), np.asarray(y, float)
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return np.nan, np.nan
    vx, vy = x.var(ddof=1), y.var(ddof=1)
    se = np.sqrt(vx / nx + vy / ny)
    if se == 0:
        return np.nan, np.nan
    t = (x.mean() - y.mean()) / se
    df = (vx / nx + vy / ny) ** 2 / ((vx / nx) ** 2 / (nx - 1) + (vy / ny) ** 2 / (ny - 1))
    # t 分布双尾 p（用正态近似，df 较大时足够；df 小时略保守）
    p = 2 * (1 - _norm_cdf(abs(t)))
    return t, p


def describe(v, name):
    v = np.asarray(v, float)
    if len(v) == 0:
        return f"  {name}: 无数据"
    return (f"  {name}: n={len(v)}, 均值={v.mean():.1f}, 中位数={np.median(v):.1f}, "
            f"标准差={v.std(ddof=1) if len(v)>1 else 0:.1f}, "
            f"范围=[{v.min():.1f}, {v.max():.1f}]")


def box_ascii(groups, title, width=50):
    """简单 ASCII 箱线图，便于在终端直接看分布。"""
    print(f"\n  {title}")
    allv = np.concatenate([np.asarray(v, float) for v in groups.values() if len(v)])
    if len(allv) == 0:
        return
    lo, hi = allv.min(), allv.max()
    rng = max(hi - lo, 1e-9)
    for name, v in groups.items():
        v = np.asarray(v, float)
        if len(v) == 0:
            print(f"  {name:>8s} | 无数据")
            continue
        q1, med, q3 = np.percentile(v, [25, 50, 75])
        line = [" "] * width
        def pos(x):
            return int((x - lo) / rng * (width - 1))
        for i in range(pos(v.min()), pos(v.max()) + 1):
            line[i] = "-"
        for i in range(pos(q1), pos(q3) + 1):
            line[i] = "="
        line[pos(med)] = "|"
        print(f"  {name:>8s} |{''.join(line)}|  中位数 {med:.1f}")
    print(f"  {'':>8s}  {lo:<.1f}{' ' * (width - 10)}{hi:.1f}")


def main():
    ap = argparse.ArgumentParser(description="两组行为指标统计比较")
    ap.add_argument("--control", default="results_control.csv")
    ap.add_argument("--antenna", default="results_antenna.csv")
    ap.add_argument("--control-dir", default="video/两种湿度下的产卵选择对照",
                    help="对照组视频目录，用于统计未产卵个体")
    ap.add_argument("--antenna-dir", default="video/两种湿度下的产卵选择触角剪除")
    ap.add_argument("--out", default="group_stats")
    args = ap.parse_args()

    ctrl = read_results(args.control, "对照")
    ante = read_results(args.antenna, "触角剪除")

    # 补上"未产卵"的个体：视频存在但结果里没有 = 未检出产卵
    ctrl_all = list_videos(args.control_dir)
    ante_all = list_videos(args.antenna_dir)
    for v in ctrl_all:
        if v not in ctrl:
            ctrl[v] = {"video": v, "group": "对照", "start": np.nan,
                       "end": np.nan, "duration_min": np.nan, "ovipositied": 0}
    for v in ante_all:
        if v not in ante:
            ante[v] = {"video": v, "group": "触角剪除", "start": np.nan,
                       "end": np.nan, "duration_min": np.nan, "ovipositied": 0}

    rows = list(ctrl.values()) + list(ante.values())

    # ---------- 产卵率 ----------
    c_yes = sum(r["ovipositied"] for r in ctrl.values())
    c_no = len(ctrl) - c_yes
    a_yes = sum(r["ovipositied"] for r in ante.values())
    a_no = len(ante) - a_yes

    print("=" * 70)
    print("一、产卵率")
    print("=" * 70)
    print(f"  对照组    : {c_yes}/{len(ctrl)} = {c_yes/max(len(ctrl),1)*100:.1f}%")
    print(f"  触角剪除组: {a_yes}/{len(ante)} = {a_yes/max(len(ante),1)*100:.1f}%")
    if c_no + a_no > 0 or True:
        p = fisher_exact_2x2(c_yes, c_no, a_yes, a_no)
        print(f"  Fisher 精确检验: p = {p:.4f} "
              f"{'（差异显著）' if p < 0.05 else '（差异不显著）'}")
    if c_no:
        print(f"  对照组未产卵: {[r['video'] for r in ctrl.values() if not r['ovipositied']]}")
    if a_no:
        print(f"  触角剪除组未产卵: {[r['video'] for r in ante.values() if not r['ovipositied']]}")

    # ---------- 产卵潜伏期 ----------
    c_lat = [r["start"] / 60 for r in ctrl.values() if r["ovipositied"]]
    a_lat = [r["start"] / 60 for r in ante.values() if r["ovipositied"]]

    print("\n" + "=" * 70)
    print("二、产卵潜伏期（分钟，视频开始到产卵起点）")
    print("=" * 70)
    print(describe(c_lat, "对照组    "))
    print(describe(a_lat, "触角剪除组"))
    u, pu = mannwhitney_u(c_lat, a_lat)
    t, pt = welch_t(c_lat, a_lat)
    print(f"  Mann-Whitney U = {u:.1f}, p = {pu:.4f} "
          f"{'（显著）' if pu < 0.05 else '（不显著）'}")
    print(f"  Welch t = {t:.3f}, p = {pt:.4f}")
    box_ascii({"对照": c_lat, "触角剪除": a_lat}, "潜伏期分布")

    # ---------- 产卵时长 ----------
    c_dur = [r["duration_min"] for r in ctrl.values() if r["ovipositied"]]
    a_dur = [r["duration_min"] for r in ante.values() if r["ovipositied"]]

    print("\n" + "=" * 70)
    print("三、产卵时长（分钟）")
    print("=" * 70)
    print(describe(c_dur, "对照组    "))
    print(describe(a_dur, "触角剪除组"))
    u, pu = mannwhitney_u(c_dur, a_dur)
    t, pt = welch_t(c_dur, a_dur)
    print(f"  Mann-Whitney U = {u:.1f}, p = {pu:.4f} "
          f"{'（显著）' if pu < 0.05 else '（不显著）'}")
    print(f"  Welch t = {t:.3f}, p = {pt:.4f}")
    box_ascii({"对照": c_dur, "触角剪除": a_dur}, "产卵时长分布")

    # ---------- 输出合并表 ----------
    out_csv = f"{args.out}.csv"
    with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["video", "group", "ovipositied",
                    "latency_min", "duration_min", "start_s", "end_s"])
        for r in sorted(rows, key=lambda x: (x["group"], x["video"])):
            w.writerow([r["video"], r["group"], r["ovipositied"],
                        "" if np.isnan(r["start"]) else round(r["start"] / 60, 1),
                        "" if np.isnan(r["duration_min"]) else round(r["duration_min"], 1),
                        "" if np.isnan(r["start"]) else round(r["start"], 1),
                        "" if np.isnan(r["end"]) else round(r["end"], 1)])
    print(f"\n合并数据已写出 {out_csv}（可直接导入 SPSS/R/Excel）")

    # ---------- 出图 ----------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))

        axes[0].bar(["Control", "Antennae\nremoved"],
                    [c_yes / max(len(ctrl), 1) * 100, a_yes / max(len(ante), 1) * 100],
                    color=["#4C72B0", "#DD8452"], width=0.55)
        axes[0].set_ylabel("Oviposition rate (%)")
        axes[0].set_title("Oviposition rate")
        axes[0].set_ylim(0, 105)

        for ax, (cv, av, title, ylab) in zip(
                axes[1:], [(c_lat, a_lat, "Oviposition latency", "Latency (min)"),
                           (c_dur, a_dur, "Oviposition duration", "Duration (min)")]):
            bp = ax.boxplot([cv, av], tick_labels=["Control", "Antennae\nremoved"],
                            patch_artist=True, widths=0.5)
            for patch, c in zip(bp["boxes"], ["#4C72B0", "#DD8452"]):
                patch.set_facecolor(c)
                patch.set_alpha(0.65)
            for i, v in enumerate([cv, av], start=1):
                if len(v):
                    ax.scatter(np.random.normal(i, 0.05, len(v)), v,
                               s=16, color="k", alpha=0.55, zorder=3)
            ax.set_title(title)
            ax.set_ylabel(ylab)

        plt.tight_layout()
        png = f"{args.out}.png"
        plt.savefig(png, dpi=200)
        print(f"图已保存 {png}")
    except Exception as ex:
        print(f"[提示] 出图跳过：{ex}")


if __name__ == "__main__":
    main()
