#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
behavior_classifier.py  ——  通用行为分类器
==========================================
不再写死"找长时间不动的时段"，而是从标注学习任意行为。

思路（对标 SimBA / DeepEthogram 的做法）：
  轨迹 (t,x,y,w,h) → 滑窗时序特征 → 随机森林 → 逐帧行为概率
                                              → 时序平滑 → 行为时段

关键设计：
  - 检测与追踪是行为无关的，只跑一次；换行为只需换标签重训（秒级）
  - 按【视频】留出验证，绝不按帧随机划分（相邻帧几乎相同，会导致精度虚高）
  - 特征含姿态（框宽高比）——产卵时身体坐沉框变扁，仅靠位移无法区分姿态

标注格式（通用版 stages.csv）：
  video,behavior,start,end
  train01,exploration,0:00:00,0:18:51
  train01,ovipositing,0:18:51,1:32:53

用法：
  # 训练 + 交叉验证
  python behavior_classifier.py train --cache track_cache \\
      --labels labels.csv --model behavior_model.pkl

  # 预测新视频
  python behavior_classifier.py predict --cache track_cache \\
      --model behavior_model.pkl --out predictions.csv
"""
import argparse
import csv
import json
import pickle
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

# 每个滑窗提取的特征名（顺序必须与 extract_features 一致）
FEATURE_NAMES = [
    "disp_rate",        # 净位移率：窗口首尾直线距离/时长——是否离开原地
    "speed_mean",       # 平均瞬时速度
    "speed_std",        # 速度波动——抖动还是平滑移动
    "speed_max",
    "path_ratio",       # 净位移/路径长度——直线移动接近1，原地打转接近0
    "aspect_mean",      # 框宽高比均值——姿态！产卵时身体坐沉变扁
    "aspect_std",       # 宽高比波动——姿态是否在变化
    "area_mean",        # 框面积
    "area_std",
    "turn_mean",        # 平均转向角
    "x_rel",            # 画面内相对位置
    "y_rel",
    "x_std",            # 位置波动范围
    "y_std",
]


def to_sec(t):
    t = str(t).strip()
    if not t:
        return None
    try:
        p = [int(x) for x in t.split(":")]
    except ValueError:
        try:
            return float(t)
        except ValueError:
            return None
    if len(p) == 3:
        return p[0] * 3600 + p[1] * 60 + p[2]
    if len(p) == 2:
        return p[0] * 60 + p[1]
    return None


def fmt_hms(sec):
    sec = int(round(sec))
    return f"{sec//3600}:{(sec%3600)//60:02d}:{sec%60:02d}"


# ---------- 特征提取 ----------

def extract_features(traj, win_sec, step_sec, frame_w=854, frame_h=480):
    """
    把一条轨迹切成滑窗，每窗提取一组特征。
    traj: [(t, cx, cy, w, h), ...]，旧格式 (t,cx,cy) 时姿态特征为 NaN
    返回 (times, X)：窗口中心时刻、特征矩阵
    """
    arr = np.array(traj, dtype=float)
    if len(arr) < 5:
        return np.array([]), np.zeros((0, len(FEATURE_NAMES)))

    t, x, y = arr[:, 0], arr[:, 1], arr[:, 2]
    has_wh = arr.shape[1] >= 5
    w = arr[:, 3] if has_wh else np.full(len(arr), np.nan)
    h = arr[:, 4] if has_wh else np.full(len(arr), np.nan)

    dt = np.median(np.diff(t)) or 1.0
    half = max(2, int(round((win_sec / 2) / dt)))
    stride = max(1, int(round(step_sec / dt)))

    times, feats = [], []
    for i in range(half, len(arr) - half, stride):
        a, b = i - half, i + half
        sx, sy = x[a:b], y[a:b]
        st = t[a:b]
        span = st[-1] - st[0]
        if span <= 0:
            continue

        # 位移类
        net = float(np.hypot(sx[-1] - sx[0], sy[-1] - sy[0]))
        disp_rate = net / span
        seg = np.hypot(np.diff(sx), np.diff(sy))
        path = float(seg.sum())
        sdt = np.diff(st)
        sdt[sdt <= 0] = 1e-6
        spd = seg / sdt
        path_ratio = net / path if path > 1e-6 else 0.0

        # 转向角
        if len(sx) >= 3:
            v = np.stack([np.diff(sx), np.diff(sy)], axis=1)
            n = np.linalg.norm(v, axis=1, keepdims=True)
            vn = v / np.maximum(n, 1e-9)
            cosang = np.clip((vn[:-1] * vn[1:]).sum(axis=1), -1, 1)
            turn_mean = float(np.degrees(np.arccos(cosang)).mean())
        else:
            turn_mean = 0.0

        # 姿态类（框宽高）
        sw, sh = w[a:b], h[a:b]
        with np.errstate(invalid="ignore", divide="ignore"):
            aspect = sw / np.maximum(sh, 1e-6)
            area = sw * sh
        aspect_mean = float(np.nanmean(aspect)) if has_wh else np.nan
        aspect_std = float(np.nanstd(aspect)) if has_wh else np.nan
        area_mean = float(np.nanmean(area)) if has_wh else np.nan
        area_std = float(np.nanstd(area)) if has_wh else np.nan

        feats.append([
            disp_rate,
            float(spd.mean()), float(spd.std()), float(spd.max()),
            path_ratio,
            aspect_mean, aspect_std, area_mean, area_std,
            turn_mean,
            float(sx.mean() / frame_w), float(sy.mean() / frame_h),
            float(sx.std()), float(sy.std()),
        ])
        times.append(float(t[i]))

    return np.array(times), np.array(feats)


def load_cache(cache_dir, video=None):
    """读轨迹缓存 → {video: {tid: traj}}"""
    out = {}
    for jf in sorted(Path(cache_dir).glob("*.json")):
        stem = jf.stem.split("_fps")[0]
        if video and stem != video:
            continue
        data = json.loads(jf.read_text(encoding="utf-8"))
        out[stem] = {int(k): [tuple(p) for p in v]
                     for k, v in data["tracks"].items()}
    return out


def load_labels(path):
    """读标注 → {video: [(behavior, start, end), ...]}"""
    lab = defaultdict(list)
    with open(path, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            s, e = to_sec(r.get("start")), to_sec(r.get("end"))
            b = (r.get("behavior") or "").strip()
            if s is None or e is None or not b or e <= s:
                continue
            lab[r["video"].strip()].append((b, s, e))
    return lab


def build_dataset(cache, labels, win_sec, step_sec, min_track_len):
    """
    对每个视频的每条轨迹提特征，按标注时段打标签。
    返回 X, y, groups(视频名)——groups 用于按视频留出验证。
    """
    X, y, groups = [], [], []
    for vid, tracks in cache.items():
        if vid not in labels:
            continue
        segs = labels[vid]
        for tid, traj in tracks.items():
            if len(traj) < min_track_len:
                continue
            times, feats = extract_features(traj, win_sec, step_sec)
            if len(times) == 0:
                continue
            for t, f in zip(times, feats):
                # 找该时刻落在哪个标注行为里
                hit = [b for b, s, e in segs if s <= t <= e]
                if len(hit) != 1:      # 无标注或标注重叠，跳过
                    continue
                X.append(f)
                y.append(hit[0])
                groups.append(vid)
    return np.array(X), np.array(y), np.array(groups)


def smooth_predictions(times, labels_pred, win):
    """滑窗众数平滑，消除逐窗抖动。"""
    if len(labels_pred) == 0:
        return labels_pred
    out = labels_pred.copy()
    half = max(1, win // 2)
    for i in range(len(labels_pred)):
        lo, hi = max(0, i - half), min(len(labels_pred), i + half + 1)
        out[i] = Counter(labels_pred[lo:hi]).most_common(1)[0][0]
    return out


def segments_from_labels(times, labs, min_dur):
    """把逐窗标签聚合成行为时段。"""
    if len(labs) == 0:
        return []
    segs, i = [], 0
    while i < len(labs):
        j = i
        while j < len(labs) and labs[j] == labs[i]:
            j += 1
        s, e = times[i], times[j - 1]
        if e - s >= min_dur:
            segs.append((labs[i], s, e, e - s))
        i = j
    return segs


# ---------- 训练 ----------

def cmd_train(args):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import classification_report, confusion_matrix

    cache = load_cache(args.cache)
    labels = load_labels(args.labels)
    print(f"轨迹缓存 {len(cache)} 个视频，标注 {len(labels)} 个视频")

    X, y, groups = build_dataset(cache, labels, args.win_sec, args.step_sec,
                                 args.min_track_len)
    if len(X) == 0:
        raise SystemExit("没有可用样本：检查标注里的 video 名与缓存文件名是否一致")

    print(f"样本 {len(X)} 个，特征 {X.shape[1]} 维")
    for b, n in sorted(Counter(y).items()):
        print(f"  {b:15s}: {n:6d} 窗")

    has_pose = not np.all(np.isnan(X[:, FEATURE_NAMES.index("aspect_mean")]))
    if not has_pose:
        print("\n⚠️ 轨迹缓存不含框宽高，姿态特征全为 NaN。"
              "\n   需用新版 detect_oviposition.py 重跑轨迹提取才能用姿态特征。")

    # 按视频留出交叉验证（绝不按帧随机划分——相邻帧几乎相同会导致精度虚高）
    vids = sorted(set(groups))
    print(f"\n按视频留一交叉验证（{len(vids)} 个视频）：")
    accs = []
    for v in vids:
        tr, te = groups != v, groups == v
        if te.sum() == 0 or len(set(y[tr])) < 2:
            continue
        imp = SimpleImputer(strategy="median")
        clf = RandomForestClassifier(n_estimators=args.n_trees,
                                     min_samples_leaf=3, n_jobs=-1,
                                     class_weight="balanced",
                                     random_state=0)
        clf.fit(imp.fit_transform(X[tr]), y[tr])
        acc = (clf.predict(imp.transform(X[te])) == y[te]).mean()
        accs.append(acc)
        print(f"  留出 {v:10s}: 窗级准确率 {acc:.3f}")
    if accs:
        print(f"  平均 {np.mean(accs):.3f} ± {np.std(accs):.3f}")

    # 用全部数据训最终模型
    imp = SimpleImputer(strategy="median")
    Xi = imp.fit_transform(X)
    clf = RandomForestClassifier(n_estimators=args.n_trees, min_samples_leaf=3,
                                 n_jobs=-1, class_weight="balanced",
                                 random_state=0)
    clf.fit(Xi, y)

    print("\n特征重要性（前 8）：")
    for i in np.argsort(clf.feature_importances_)[::-1][:8]:
        print(f"  {FEATURE_NAMES[i]:12s} {clf.feature_importances_[i]:.4f}")

    with open(args.model, "wb") as f:
        pickle.dump({"clf": clf, "imputer": imp, "features": FEATURE_NAMES,
                     "win_sec": args.win_sec, "step_sec": args.step_sec,
                     "classes": list(clf.classes_)}, f)
    print(f"\n模型已保存 {args.model}")


# ---------- 预测 ----------

def cmd_predict(args):
    with open(args.model, "rb") as f:
        m = pickle.load(f)
    clf, imp = m["clf"], m["imputer"]
    win_sec, step_sec = m["win_sec"], m["step_sec"]

    cache = load_cache(args.cache, args.video)
    if not cache:
        raise SystemExit(f"{args.cache} 里没有轨迹缓存")

    rows = []
    for vid, tracks in sorted(cache.items()):
        print(f"\n=== {vid} ===")
        # 取最长的轨迹作为主个体（拼接逻辑见 detect_oviposition.py）
        best = None
        for tid, traj in tracks.items():
            if len(traj) < args.min_track_len:
                continue
            times, feats = extract_features(traj, win_sec, step_sec)
            if len(times) == 0:
                continue
            if best is None or len(times) > len(best[1]):
                best = (tid, times, feats)
        if best is None:
            print("  无有效轨迹")
            continue

        tid, times, feats = best
        pred = clf.predict(imp.transform(feats))
        pred = smooth_predictions(times, pred, args.smooth_win)
        segs = segments_from_labels(times, pred, args.min_duration)

        for b, s, e, d in segs:
            print(f"  {b:15s} {fmt_hms(s)} ~ {fmt_hms(e)}  ({d/60:.1f} 分钟)")
            rows.append([vid, tid, b, round(s, 1), round(e, 1),
                         fmt_hms(s), fmt_hms(e), round(d / 60, 1)])

    if args.out and rows:
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["video", "track_id", "behavior", "start_sec", "end_sec",
                        "start_hms", "end_hms", "duration_min"])
            w.writerows(rows)
        print(f"\n结果已写出 {args.out}")


def main():
    ap = argparse.ArgumentParser(description="通用行为分类器")
    sub = ap.add_subparsers(dest="cmd", required=True)

    common = dict(win_sec=60.0, step_sec=10.0, min_track_len=30)

    t = sub.add_parser("train", help="训练行为模型")
    t.add_argument("--cache", default="track_cache")
    t.add_argument("--labels", required=True, help="标注 csv: video,behavior,start,end")
    t.add_argument("--model", default="behavior_model.pkl")
    t.add_argument("--win-sec", type=float, default=common["win_sec"])
    t.add_argument("--step-sec", type=float, default=common["step_sec"])
    t.add_argument("--min-track-len", type=int, default=common["min_track_len"])
    t.add_argument("--n-trees", type=int, default=300)
    t.set_defaults(func=cmd_train)

    p = sub.add_parser("predict", help="预测行为时段")
    p.add_argument("--cache", default="track_cache")
    p.add_argument("--model", default="behavior_model.pkl")
    p.add_argument("--video", help="只预测某个视频")
    p.add_argument("--out", default="behavior_predictions.csv")
    p.add_argument("--min-track-len", type=int, default=common["min_track_len"])
    p.add_argument("--smooth-win", type=int, default=15,
                   help="滑窗众数平滑的窗口数")
    p.add_argument("--min-duration", type=float, default=300,
                   help="行为时段最短秒数，短于此的碎片丢弃")
    p.set_defaults(func=cmd_predict)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
