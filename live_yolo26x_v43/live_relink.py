"""V4.3 lightweight live-only second-pass SR reattachment.

Only reuses the OSNet embeddings already computed for detections. A spatial
and size gate prevents a close-looking but distant newcomer from hijacking
an unmatched track; long exits are handled by identity_memory instead.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment


def unit(x):
    if x is None:
        return None
    a = np.asarray(x, dtype=np.float32).reshape(-1)
    if not a.size or not np.isfinite(a).all():
        return None
    norm = float(np.linalg.norm(a))
    return a / norm if norm > 1e-10 else None


def _box(obj):
    if hasattr(obj, 'to_tlwh'):
        b = obj.to_tlwh()
    else:
        b = getattr(obj, 'tlwh', None)
    if b is None:
        return None
    b = np.asarray(b, dtype=np.float32).reshape(-1)
    if b.size != 4 or not np.isfinite(b).all() or b[2] <= 1 or b[3] <= 1:
        return None
    return b


def _near(a, b, max_distance=.90):
    if a is None or b is None:
        return False
    scale = max(24., .5 * float(a[3] + b[3]))
    dist = float(np.hypot(a[0]+a[2]/2-b[0]-b[2]/2,
                          a[1]+a[3]/2-b[1]-b[3]/2)/scale)
    size = max(.01, float(a[2]*a[3]))/max(.01, float(b[2]*b[3]))
    return dist <= max_distance and .25 <= size <= 4.


def _iou(a,b):
    if a is None or b is None:
        return 0.
    x1=max(a[0],b[0]);y1=max(a[1],b[1])
    x2=min(a[0]+a[2],b[0]+b[2]);y2=min(a[1]+a[3],b[1]+b[3])
    inter=max(0.,x2-x1)*max(0.,y2-y1)
    return float(inter/max(1e-9,a[2]*a[3]+b[2]*b[3]-inter))


def reconnect_unmatched(tracks, detections, unmatched_tracks, unmatched_dets,
                        min_similarity=.88, margin=.08, max_missed=12):
    """Conservative, 1:1, gated extra matches after the original SR matcher."""
    ti = [i for i in unmatched_tracks
          if 0 <= i < len(tracks) and tracks[i].time_since_update <= max_missed
          and bool(tracks[i].features) and not tracks[i].is_deleted()]
    di = [j for j in unmatched_dets
          if 0 <= j < len(detections) and unit(detections[j].feature) is not None]
    if not ti or not di:
        return [], list(unmatched_tracks), list(unmatched_dets)
    feats = [unit(detections[j].feature) for j in di]
    det_boxes = [_box(detections[j]) for j in di]
    # Never perform speculative 2nd-pass reattachment inside a dense crossing.
    # The underlying StrongSORT matcher still runs as usual; this safeguard
    # only disables the *fallback* that might glue the wrong internal IDs.
    all_recent_boxes = [(i, _box(t)) for i,t in enumerate(tracks)
                        if not t.is_deleted() and t.time_since_update<=max_missed]
    crowded_t = {i for i in ti if any(k!=i and _iou(_box(tracks[i]),b)>.18
                                         for k,b in all_recent_boxes)}
    crowded_d = {col for col,b in enumerate(det_boxes)
                 if any(col != k and _iou(b,bb)>.18
                        for k,bb in enumerate(det_boxes))}
    scores = np.full((len(ti), len(di)), -np.inf, dtype=np.float32)
    for row, i in enumerate(ti):
        if i in crowded_t:
            continue
        prev = unit(tracks[i].features[-1])
        tb = _box(tracks[i])
        if prev is None or tb is None:
            continue
        for col, f in enumerate(feats):
            if col in crowded_d:
                continue
            if f is None or f.shape != prev.shape or not _near(tb, det_boxes[col]):
                continue
            sim = float(np.dot(prev, f))
            if sim >= min_similarity:
                scores[row, col] = sim
    if not np.isfinite(scores).any():
        return [], list(unmatched_tracks), list(unmatched_dets)
    n, m = scores.shape
    costs = np.full((n, m+n), 1e4, dtype=np.float32)
    costs[:, :m] = np.where(np.isfinite(scores), -scores, 1e4)
    for row in range(n):
        costs[row, m+row] = -min_similarity + 1e-6
    rows, cols = linear_sum_assignment(costs)
    restored = []
    for row, col in zip(rows, cols):
        if col >= m or not np.isfinite(scores[row, col]):
            continue
        s = float(scores[row, col])
        others = ([float(scores[row, k]) for k in range(m)
                   if k != col and np.isfinite(scores[row, k])]
                  + [float(scores[k, col]) for k in range(n)
                     if k != row and np.isfinite(scores[k, col])])
        if others and s - max(others) < margin:
            continue
        restored.append((ti[row], di[col]))
    used_t = {a for a, _ in restored}
    used_d = {b for _, b in restored}
    return (restored,
            [i for i in unmatched_tracks if i not in used_t],
            [j for j in unmatched_dets if j not in used_d])


def make_live_sr_tracker(SR, *, min_similarity=.88, margin=.08, max_missed=12,
                         enabled=True):
    class LiveSR(SR):
        def __init__(self):
            super().__init__()
            self.live_reattach_count = 0

        def _match(self, detections):
            matches, ut, ud = super()._match(detections)
            if enabled and ut and ud:
                recovered, ut, ud = reconnect_unmatched(
                    self.tracks, detections, ut, ud,
                    min_similarity, margin, max_missed)
                self.live_reattach_count += len(recovered)
                matches.extend(recovered)
            return matches, ut, ud

    return LiveSR
