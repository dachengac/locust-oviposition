"""NumPy IoU and optimal assignment for ByteTrack, no lap/scipy extension."""
import numpy as np


def hungarian(cost):
    """Minimum-cost assignment for a finite square matrix (O(n^3))."""
    n = len(cost)
    if not n:
        return np.empty(0, dtype=int)
    u, v = np.zeros(n + 1), np.zeros(n + 1)
    p, way = np.zeros(n + 1, dtype=int), np.zeros(n + 1, dtype=int)
    for i in range(1, n + 1):
        p[0] = i
        j0, minimum, used = 0, np.full(n + 1, np.inf), np.zeros(n + 1, dtype=bool)
        while True:
            used[j0] = True
            i0, delta, j1 = p[j0], np.inf, 0
            for j in range(1, n + 1):
                if not used[j]:
                    current = cost[i0 - 1, j - 1] - u[i0] - v[j]
                    if current < minimum[j]:
                        minimum[j], way[j] = current, j0
                    if minimum[j] < delta:
                        delta, j1 = minimum[j], j
            for j in range(n + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minimum[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    assignment = np.empty(n, dtype=int)
    for j in range(1, n + 1):
        assignment[p[j] - 1] = j - 1
    return assignment


def linear_assignment(cost_matrix, thresh):
    m, n = cost_matrix.shape
    if not m or not n:
        return np.empty((0, 2), dtype=int), tuple(range(m)), tuple(range(n))
    # Dummy rows/columns model rejection at thresh, as lapjv(cost_limit=thresh).
    padded = np.full((m + n, m + n), thresh / 2, dtype=float)
    padded[:m, :n] = np.where(cost_matrix <= thresh, cost_matrix, 1e6)
    padded[m:, n:] = 0
    assignment = hungarian(padded)
    matches = [(i, int(j)) for i, j in enumerate(assignment[:m]) if j < n and cost_matrix[i, j] <= thresh]
    used_rows, used_cols = {i for i, _ in matches}, {j for _, j in matches}
    return np.asarray(matches, dtype=int).reshape(-1, 2), tuple(i for i in range(m) if i not in used_rows), tuple(j for j in range(n) if j not in used_cols)


def iou_distance(a, b):
    if not a or not b:
        return np.zeros((len(a), len(b)), dtype=float)
    aa, bb = np.asarray([t.tlbr for t in a]), np.asarray([t.tlbr for t in b])
    low = np.maximum(aa[:, None, :2], bb[None, :, :2])
    high = np.minimum(aa[:, None, 2:], bb[None, :, 2:])
    wh = np.maximum(high - low, 0)
    intersection = wh[:, :, 0] * wh[:, :, 1]
    area_a = np.prod(np.maximum(aa[:, 2:] - aa[:, :2], 0), axis=1)
    area_b = np.prod(np.maximum(bb[:, 2:] - bb[:, :2], 0), axis=1)
    return 1 - intersection / np.maximum(area_a[:, None] + area_b[None, :] - intersection, 1e-9)


def fuse_score(cost_matrix, detections):
    return 1 - (1 - cost_matrix) * np.array([d.score for d in detections])[None, :]
