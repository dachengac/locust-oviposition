#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
detect_oviposition.py  ——  时序分析识别产卵时段
================================================
思路（动物行为自动量化的通用四层做法）：
  1. 轨迹提取：YOLO 检测 + ByteTrack 追踪 → 每个个体逐帧 (t, x, y)
  2. 速度而非位移：产卵时有小幅晃动（低速非零），爬行是高速；用速度才能分开
  3. 滑窗平滑 + 自适应阈值：窗口内平均速度抹掉晃动；阈值从本视频速度分布自适应，
     不写死像素值（不同视频缩放/虫体大小不同）
  4. 状态机 + 最短时长：连续低速 ≥ min-duration 才算产卵事件，滤掉短暂停留

判据 = 低速 AND 位置稳定 AND 持续够久（三者同时满足）

用法：
  # 单个视频
  python detect_oviposition.py --weights runs/detect/runs/v2/weights/best.pt \\
      --video video/Training_Set/train01.mp4 --stages stages.csv

  # 批量跑全部并输出误差报告
  python detect_oviposition.py --weights runs/detect/runs/v2/weights/best.pt \\
      --video-dir video/Training_Set --stages stages.csv --out results.csv
"""
import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


DEVICE = None    # 由 main() 设置


# ---------- 工具 ----------

def to_sec(t):
    """'1:32:53' -> 秒；空值返回 None"""
    t = (t or "").strip()
    if not t:
        return None
    try:
        p = [int(x) for x in t.split(":")]
    except ValueError:
        return None
    if len(p) == 3:
        return p[0] * 3600 + p[1] * 60 + p[2]
    if len(p) == 2:
        return p[0] * 60 + p[1]
    return None


def fmt_hms(sec):
    """秒 -> '1:32:53'"""
    if sec is None:
        return "-"
    sec = int(round(sec))
    return f"{sec // 3600}:{(sec % 3600) // 60:02d}:{sec % 60:02d}"


# ---------- 第 1 层：轨迹提取 ----------

def extract_tracks(model, video, sample_fps, imgsz, conf, cache_dir=None):
    """
    按 sample_fps 抽帧跑检测+追踪，返回 {track_id: [(t_sec, cx, cy, w, h), ...]}

    注：w/h（框宽高）是行为分类的关键姿态特征——产卵时身体坐沉、框变扁，
    仅靠中心点无法区分姿态。旧版缓存只有 (t, cx, cy)，读入时会自动补 NaN。

    检测追踪是最耗时的一步（几分钟到几十分钟），而后续时序参数（阈值、窗口、
    最短时长等）完全不影响轨迹。因此把轨迹缓存到磁盘，调参时直接复用，
    避免每换一个参数就重跑一遍推理。
    """
    video = Path(video)
    cache_file = None
    if cache_dir:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / f"{video.stem}_fps{sample_fps}_conf{conf}.json"
        # 旧缓存（3 元组，无 w/h）仍可读，但缺姿态特征
        if cache_file.exists():
            import json
            with open(cache_file, encoding="utf-8") as f:
                data = json.load(f)
            tracks = {int(k): [tuple(p) for p in v] for k, v in data["tracks"].items()}
            print(f"  ✓ 读取轨迹缓存 {cache_file.name}（{len(tracks)} 条轨迹）")
            return tracks, data["fps"]

    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    stride = max(1, int(round(fps / sample_fps)))   # 每隔 stride 帧取一帧
    print(f"  视频 {fps:.1f}fps, {total} 帧, 每 {stride} 帧取一帧 "
          f"(≈{sample_fps} fps, 共约 {total // stride} 帧)")

    tracks = {}
    # vid_stride 让 Ultralytics 内部跳帧，比自己 seek 快得多
    track_kw = dict(source=str(video), tracker="bytetrack.yaml",
                    imgsz=imgsz, conf=conf, vid_stride=stride,
                    stream=True, verbose=False, persist=True)
    if DEVICE is not None:
        track_kw["device"] = DEVICE
    results = model.track(**track_kw)

    for i, r in enumerate(results):
        t = i * stride / fps                        # 该帧对应的真实时间（秒）
        if r.boxes is None or r.boxes.id is None:
            continue
        ids = r.boxes.id.int().tolist()
        xywh = r.boxes.xywh.cpu().numpy()           # 中心点 + 宽高（像素）
        for tid, (cx, cy, w, h) in zip(ids, xywh):
            tracks.setdefault(tid, []).append(
                (t, float(cx), float(cy), float(w), float(h)))

    if cache_file:
        import json
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump({"fps": fps, "tracks": {str(k): v for k, v in tracks.items()}}, f)
        print(f"  ✓ 轨迹已缓存到 {cache_file.name}，后续调参可秒出结果")
    return tracks, fps


# ---------- 第 2-3 层：速度 + 滑窗 + 自适应阈值 ----------

def compute_speed(traj, win_sec):
    """
    返回 (times, 净位移率 px/s)。

    关键：不能用逐帧速度——产卵时的小幅随机晃动，其瞬时速度可能比平滑爬行还高
    （抖动每帧都在改变方向）。真正的区别在于**净位移**：产卵时虫子原地晃，
    一分钟后仍在原处（净位移≈0）；爬行时一分钟后已移出很远（净位移大）。
    因此对每个时刻，计算前后 win_sec 窗口内首尾两点的直线距离 / 时间跨度。
    """
    if len(traj) < 3:
        return np.array([]), np.array([])
    arr = np.array(traj, dtype=float)[:, :3]   # 只取 t, cx, cy（兼容新旧格式）
    t, x, y = arr[:, 0], arr[:, 1], arr[:, 2]

    step = np.median(np.diff(t))
    half = max(1, int(round((win_sec / 2) / max(step, 1e-6))))

    n = len(t)
    disp_rate = np.zeros(n)
    for i in range(n):
        a, b = max(0, i - half), min(n - 1, i + half)
        dt = t[b] - t[a]
        if dt <= 0:
            disp_rate[i] = 0.0
            continue
        # 窗口内首尾直线距离（净位移），而非路径长度
        disp_rate[i] = np.hypot(x[b] - x[a], y[b] - y[a]) / dt
    return t, disp_rate


def adaptive_threshold(speed, percentile):
    """
    自适应阈值：速度分布通常是双峰的（静止的小幅晃动 vs 爬行），
    优先用 Otsu 法在两峰之间自动找分割点；分布不明显时退回分位数。
    不写死像素值，因为不同视频的缩放/虫体大小不同。
    """
    if len(speed) == 0:
        return 0.0
    s = np.asarray(speed, dtype=float)
    s = s[np.isfinite(s)]
    if len(s) == 0:
        return 0.0

    # Otsu：在直方图上找类间方差最大的分割点
    lo, hi = np.percentile(s, 1), np.percentile(s, 99)
    if hi <= lo:
        return float(np.percentile(s, percentile))
    hist, edges = np.histogram(s, bins=64, range=(lo, hi))
    hist = hist.astype(float)
    total = hist.sum()
    if total == 0:
        return float(np.percentile(s, percentile))
    centers = (edges[:-1] + edges[1:]) / 2
    w0 = np.cumsum(hist) / total                     # 前景权重
    w1 = 1.0 - w0
    mu0 = np.cumsum(hist * centers) / np.maximum(np.cumsum(hist), 1e-9)
    total_mu = (hist * centers).sum() / total
    mu1 = (total_mu - w0 * mu0) / np.maximum(w1, 1e-9)
    var_between = w0 * w1 * (mu0 - mu1) ** 2
    k = int(np.nanargmax(var_between))
    thr_otsu = float(centers[k])

    # 兜底：Otsu 结果异常时退回分位数
    thr_pct = float(np.percentile(s, percentile))
    if not np.isfinite(thr_otsu) or thr_otsu <= 0:
        return thr_pct
    return thr_otsu


# ---------- 第 4 层：状态机 ----------

def merge_close(events, max_gap):
    """合并间隔很短的相邻静止段——一次产卵可能因瞬时检测抖动被切成两段。"""
    if not events:
        return events
    events = sorted(events, key=lambda e: e[0])
    merged = [list(events[0])]
    for s, e, d, drift in events[1:]:
        if s - merged[-1][1] <= max_gap:
            merged[-1][1] = e
            merged[-1][2] = merged[-1][1] - merged[-1][0]
            merged[-1][3] = max(merged[-1][3], drift)
        else:
            merged.append([s, e, d, drift])
    return [tuple(m) for m in merged]


def stitch_segments(segments, max_time_gap=300, max_space_dist=200,
                    max_total_drift=None):
    """
    轨迹拼接（tracklet stitching）——多目标追踪的标准后处理。

    ByteTrack 的 ID 会因遮挡、漏检而断裂，一次产卵常被拆成多条轨迹
    （实测 train12 拆成 1908/1910 两段、train10 拆成 858/871 两段）。
    本函数按判据把属于同一次产卵的碎片重新拼起来：

      1. 时间连续性：两段间隔 ≤ max_time_gap（允许短暂断开或重叠）
      2. 空间连续性：前段终点与后段起点距离 ≤ max_space_dist（个体不会瞬移）
      3. 整体稳定性：拼接后首尾距离 ≤ max_total_drift（产卵全程应在原地，
         若拼完发现整体移动很远，说明把不同行为段错连了——实测 train08 的
         探索期静止段就是这样被误连的）
      4. 行为一致性：两段都已被判为静止（调用前已过滤）

    segments: [(start, end, dur, drift, tid, start_xy, end_xy), ...]
    返回合并后的 [(start, end, dur, drift, [tid...]), ...]
    """
    if not segments:
        return []
    segs = sorted(segments, key=lambda s: s[0])
    merged = []
    # cur: [start, end, dur, drift, [tids], start_xy, end_xy]
    cur = list(segs[0][:4]) + [[segs[0][4]], segs[0][5], segs[0][6]]

    def flush():
        merged.append(tuple(cur[:5]))

    for s, e, d, drift, tid, sxy, exy in segs[1:]:
        gap = s - cur[1]                       # 负值表示时间上有重叠
        dist = float(np.hypot(sxy[0] - cur[6][0], sxy[1] - cur[6][1]))
        ok = gap <= max_time_gap and dist <= max_space_dist
        if ok and max_total_drift is not None:
            # 拼接后整体首尾距离不能太大
            total = float(np.hypot(exy[0] - cur[5][0], exy[1] - cur[5][1]))
            ok = total <= max_total_drift
        if ok:
            if e > cur[1]:
                cur[1] = e
                cur[6] = exy
            cur[2] = cur[1] - cur[0]
            cur[3] = max(cur[3], drift)
            cur[4].append(tid)
        else:
            flush()
            cur = [s, e, d, drift, [tid], sxy, exy]
    flush()
    return merged


def find_still_periods(times, speed, thr, min_dur, positions, max_drift,
                       merge_gap=120):
    """
    找出满足三条件的时段：低速 + 位置稳定 + 持续够久。
    positions: 与 times 对齐的 (x, y)，用于检查是否只是慢慢挪走。
    """
    if len(times) == 0:
        return []
    still = speed < thr
    raw, i, n = [], 0, len(still)
    while i < n:
        if not still[i]:
            i += 1
            continue
        j = i
        while j < n and still[j]:
            j += 1
        seg = positions[i:j]
        drift = float(np.max(np.ptp(seg, axis=0))) if len(seg) else 0.0
        # 先按漂移过滤：漂移过大说明是慢慢挪走，不是原地产卵。
        # 必须在合并之前过滤，否则合并会把静止段和相邻的移动段连成一片。
        if drift <= max_drift:
            raw.append((times[i], times[j - 1], times[j - 1] - times[i], drift))
        i = j

    # 再合并被瞬时抖动切开的相邻合格段，最后按最短时长过滤
    return [(s, e, d, drift) for s, e, d, drift in merge_close(raw, merge_gap)
            if d >= min_dur]


# ---------- 主流程 ----------

def analyze_video(model, video, args):
    print(f"\n=== {Path(video).name} ===")
    tracks, fps = extract_tracks(model, video, args.sample_fps, args.imgsz,
                                 args.conf, getattr(args, 'cache_tracks', None))
    if not tracks:
        print("  未检测到任何轨迹")
        return None

    # 汇总所有轨迹的速度，用于自适应阈值（全局分布比单条轨迹稳）
    all_speed = []
    per_track = {}
    for tid, traj in tracks.items():
        if len(traj) < args.min_track_len:
            continue
        t_mid, sp = compute_speed(traj, args.win_sec)
        if len(sp) == 0:
            continue
        pos = np.array([(p[1], p[2]) for p in traj])[:len(sp)]
        per_track[tid] = (t_mid, sp, pos)
        all_speed.append(sp)

    if not per_track:
        print("  无足够长的轨迹")
        return None

    thr = adaptive_threshold(np.concatenate(all_speed), args.speed_percentile)
    print(f"  轨迹数 {len(per_track)}，自适应速度阈值 {thr:.2f} px/s")

    # 收集所有轨迹的静止段。注意：不能直接"取最长轨迹"——实测 train08 中
    # 最长的 track 从探索期就开始，误差 -17.5 分钟，而较短的 track 532
    # 反而只差 1.5 分钟。正确做法是先拼接碎片，再在拼好的段里选。
    raw_segs = []
    for tid, (t_mid, sp, pos) in per_track.items():
        # 按更宽松的时长收集碎片（拼接后再按 min_duration 过滤），
        # 否则被 ID 断裂切碎的短段会在拼接前就被丢掉
        frag_min = min(args.min_duration, args.frag_min_duration)
        for (s, e, d, drift) in find_still_periods(t_mid, sp, thr, frag_min,
                                                   pos, args.max_drift):
            i0 = int(np.searchsorted(t_mid, s))
            i1 = min(int(np.searchsorted(t_mid, e)), len(pos) - 1)
            raw_segs.append((s, e, d, drift, tid,
                             tuple(pos[i0]), tuple(pos[i1])))

    if not raw_segs:
        print(f"  未找到持续 ≥{args.min_duration/60:.0f} 分钟的静止时段")
        return None

    # 轨迹拼接：把被 ID 断裂切开的同一次产卵重新连起来
    stitched = stitch_segments(raw_segs, args.stitch_gap, args.stitch_dist,
                               args.max_drift)
    candidates = [(s, e, d, tids) for s, e, d, drift, tids in stitched
                  if d >= args.min_duration]
    if not candidates:
        print(f"  拼接后仍无 ≥{args.min_duration/60:.0f} 分钟的时段")
        return None

    # 选择策略：不能只按时长取最长——实测 train08 中最长的 track 从探索期
    # 就开始（误差 -17.5 分钟），而较短的 track 532 只差 1.5 分钟。
    # 改用「证据密度」：真正的产卵时段应有多条轨迹共同支持（雌雄同框），
    # 而探索期的静止往往只有单条轨迹。按 支持轨迹数 × 时长 打分。
    def support_count(s, e):
        """统计有多少条独立轨迹的静止段与 [s,e] 显著重叠。"""
        n = 0
        for rs, re_, rd, _, tid, _, _ in raw_segs:
            ov = min(e, re_) - max(s, rs)
            if ov > 0 and ov >= 0.5 * min(rd, e - s):
                n += 1
        return n

    scored = []
    for s, e, d, tids in candidates:
        sup = support_count(s, e)
        scored.append((sup, d, s, e, tids))
    scored.sort(reverse=True)              # 先比支持数，再比时长

    sup, d, s, e, tids = scored[0]
    tid_str = "+".join(str(x) for x in dict.fromkeys(tids))
    extra = f"（{len(set(tids))} 条轨迹拼接）" if len(set(tids)) > 1 else ""
    print(f"  → 产卵候选: track {tid_str}  {fmt_hms(s)} ~ {fmt_hms(e)}  "
          f"时长 {d/60:.1f} 分钟  支持轨迹 {sup} 条{extra}")
    if len(scored) > 1:
        print(f"    （另有 {len(scored)-1} 个候选时段）")
    return {"video": Path(video).stem, "track_id": tid_str,
            "pred_start": s, "pred_end": e, "pred_dur": d}


def compare_with_truth(rows, stages_csv):
    """与人工标注的 stages.csv 对比，输出误差。"""
    truth = {}
    with open(stages_csv, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            s, e = to_sec(r.get("ovi_start")), to_sec(r.get("ovi_end"))
            if s is not None and e is not None:
                truth[r["video"].strip()] = (s, e)

    print("\n" + "=" * 78)
    print("与人工标注对比（正值=模型判断偏晚，负值=偏早）")
    print("=" * 78)
    print(f"{'视频':10s} {'人工产卵时段':24s} {'模型判断':24s} "
          f"{'起点误差':>9s} {'终点误差':>9s} {'时长误差':>9s}")
    print("-" * 78)

    errs = []
    for row in rows:
        v = row["video"]
        if v not in truth:
            continue
        ts, te = truth[v]
        es = row["pred_start"] - ts
        ee = row["pred_end"] - te
        ed = row["pred_dur"] - (te - ts)
        errs.append((abs(es), abs(ee), abs(ed)))
        print(f"{v:10s} {fmt_hms(ts)+' ~ '+fmt_hms(te):24s} "
              f"{fmt_hms(row['pred_start'])+' ~ '+fmt_hms(row['pred_end']):24s} "
              f"{es/60:>8.1f}m {ee/60:>8.1f}m {ed/60:>8.1f}m")

    if errs:
        a = np.array(errs) / 60
        print("-" * 78)
        print(f"{'平均绝对误差':10s} {'':24s} {'':24s} "
              f"{a[:,0].mean():>8.1f}m {a[:,1].mean():>8.1f}m {a[:,2].mean():>8.1f}m")
        print(f"{'中位数':10s} {'':24s} {'':24s} "
              f"{np.median(a[:,0]):>8.1f}m {np.median(a[:,1]):>8.1f}m "
              f"{np.median(a[:,2]):>8.1f}m")
        print(f"\n共对比 {len(errs)} 个视频")


def main():
    ap = argparse.ArgumentParser(description="时序分析识别产卵时段")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--video", help="单个视频路径")
    ap.add_argument("--video-dir", help="批量：视频目录")
    ap.add_argument("--stages", help="stages.csv，提供则对比误差")
    ap.add_argument("--out", help="结果输出 csv")
    # 检测参数
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.7)
    ap.add_argument("--sample-fps", type=float, default=1.0,
                    help="每秒取几帧（产卵是慢行为，1 fps 足够，比逐帧快 30 倍）")
    # 时序参数
    ap.add_argument("--win-sec", type=float, default=60,
                    help="速度滑窗秒数，抹平小幅晃动")
    ap.add_argument("--speed-percentile", type=float, default=40,
                    help="自适应阈值取速度分布的第几分位（越小越严格）")
    ap.add_argument("--min-duration", type=float, default=900,
                    help="最短产卵时长（秒），默认 900=15 分钟")
    ap.add_argument("--max-drift", type=float, default=150,
                    help="产卵期间中心点最大允许漂移（像素）")
    ap.add_argument("--min-track-len", type=int, default=20,
                    help="轨迹最少点数，过短的忽略")
    # 轨迹拼接（ByteTrack 的 ID 会因遮挡/漏检断裂，需把碎片重连）
    ap.add_argument("--stitch-gap", type=float, default=300,
                    help="拼接允许的最大时间间隔（秒），默认 300=5 分钟")
    ap.add_argument("--stitch-dist", type=float, default=200,
                    help="拼接允许的最大空间距离（像素）")
    ap.add_argument("--frag-min-duration", type=float, default=300,
                    help="拼接前收集碎片的最短时长（秒），默认 300=5 分钟")
    ap.add_argument("--cache-tracks", default="track_cache",
                    help="轨迹缓存目录。检测追踪只跑一次，调参时秒出结果；"
                         "设为空字符串则禁用缓存")
    ap.add_argument("--device", default=None,
                    help="推理设备，如 0 表示第一块 DCU；默认自动")
    args = ap.parse_args()

    if not args.video and not args.video_dir:
        raise SystemExit("需指定 --video 或 --video-dir")

    global DEVICE
    DEVICE = args.device
    model = YOLO(args.weights)
    videos = ([Path(args.video)] if args.video
              else sorted(Path(args.video_dir).glob("*.mp4")))

    rows = []
    for v in videos:
        try:
            r = analyze_video(model, v, args)
            if r:
                rows.append(r)
        except Exception as ex:
            print(f"  [错误] {v.name}: {ex}")

    if args.out and rows:
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["video", "track_id", "pred_start", "pred_end",
                        "pred_start_hms", "pred_end_hms", "pred_duration_min"])
            for r in rows:
                w.writerow([r["video"], r["track_id"],
                            round(r["pred_start"], 1), round(r["pred_end"], 1),
                            fmt_hms(r["pred_start"]), fmt_hms(r["pred_end"]),
                            round(r["pred_dur"] / 60, 1)])
        print(f"\n结果已写出 {args.out}")

    if args.stages and rows:
        compare_with_truth(rows, args.stages)


if __name__ == "__main__":
    main()
