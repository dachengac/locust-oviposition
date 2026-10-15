"""Reuse the published temporal algorithm; weights and rules travel together."""
import numpy as np
from scripts.detect_oviposition import (compute_speed, adaptive_threshold,
                                      find_still_periods, stitch_segments)


def oviposition_events(tracks, params, label="产卵"):
    per_track, speeds = {}, []
    for tid, trajectory in tracks.items():
        if len(trajectory) < params["min_track_len"]:
            continue
        times, speed = compute_speed(trajectory, params["win_sec"])
        if len(speed):
            positions = np.array([(p[1], p[2]) for p in trajectory])[:len(speed)]
            per_track[tid] = times, speed, positions
            speeds.append(speed)
    if not speeds:
        return []
    threshold = adaptive_threshold(np.concatenate(speeds), params["speed_percentile"])
    raw = []
    for tid, (times, speed, positions) in per_track.items():
        for start, end, duration, drift in find_still_periods(
                times, speed, threshold, min(params["min_duration"], params["frag_min_duration"]),
                positions, params["max_drift"], params["merge_gap"]):
            a = int(np.searchsorted(times, start))
            b = min(int(np.searchsorted(times, end)), len(positions) - 1)
            raw.append((start, end, duration, drift, tid, tuple(positions[a]), tuple(positions[b])))
    stitched = stitch_segments(raw, params["stitch_gap"], params["stitch_dist"], params["max_drift"])
    candidates = []
    for start, end, duration, _, ids in stitched:
        if duration < params["min_duration"]:
            continue
        support = sum(min(end, re) - max(start, rs) > 0 and
                      min(end, re) - max(start, rs) >= .5 * min(rd, end - start)
                      for rs, re, rd, *_ in raw)
        candidates.append(dict(start=float(start), end=float(end), behavior=label,
                               support=int(support), track_ids=sorted(set(int(i) for i in ids))))
    # Preserve the original support-first, duration-second selection: one best interval/video.
    candidates.sort(key=lambda e: (e["support"], e["end"] - e["start"], e["start"], e["end"]), reverse=True)
    return candidates[:1]


def presence_events(samples, duration, labels, params):
    """One interval per class; not an individual-animal behavior classifier."""
    events = []
    for class_id, label in enumerate(labels):
        intervals = []
        for index, sample in enumerate(samples):
            if class_id not in sample["classes"]:
                continue
            start = sample["time"]
            end = samples[index + 1]["time"] if index + 1 < len(samples) else duration
            if intervals and start - intervals[-1][1] <= params["stitch_gap"]:
                intervals[-1][1] = end
            else:
                intervals.append([start, end])
        events.extend(dict(start=float(s), end=float(e), behavior=label, support=0, track_ids=[])
                      for s, e in intervals if e - s >= params["min_duration"])
    return sorted(events, key=lambda event: event["start"])
