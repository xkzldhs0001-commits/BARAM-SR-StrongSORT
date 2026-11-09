from pathlib import Path
from collections import defaultdict
from scipy.optimize import linear_sum_assignment
import argparse
import shutil

import numpy as np
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs/identity_shield"
OUT = ROOT / "outputs/recall_boost"

PROFILES = [
    ("sr_recover3", 3, None, 0),
    ("sr_recover6", 6, None, 0),
    ("sr_recover10", 10, None, 0),
    ("sr_spawn55", 6, 0.55, 4),
    ("sr_spawn40", 6, 0.40, 4),
]


def read_mot(path):
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.stat().st_size == 0:
        return np.empty((0, 10), dtype=float)

    data = np.loadtxt(path, delimiter=",", ndmin=2)
    if data.shape[1] != 10:
        raise ValueError(f"Invalid MOT format: {path}")
    return data


def write_mot(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not len(data):
        path.write_text("")
        return

    data = data[np.lexsort((data[:, 1], data[:, 0]))]
    np.savetxt(
        path, data, delimiter=",",
        fmt=[
            "%d", "%d", "%.2f", "%.2f", "%.2f",
            "%.2f", "%.5f", "%d", "%d", "%d"
        ],
    )


def iou(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[0] + a[2], b[0] + b[2])
    y2 = min(a[1] + a[3], b[1] + b[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / max(union, 1e-8)


def center(box):
    return np.array([
        box[0] + box[2] / 2,
        box[1] + box[3] / 2
    ], dtype=float)


def make_row(frame, tid, box, confidence):
    return np.array([
        frame, tid,
        *box,
        confidence, -1, -1, -1,
    ], dtype=float)


def group_frames(data):
    grouped = defaultdict(list)
    for row in data:
        grouped[int(row[0])].append(row)
    return grouped


def get_yolo_cache(model, dataset, sequence, imgsz):
    cache_dir = OUT / "detcache" / dataset
    cache_dir.mkdir(parents=True, exist_ok=True)

    cache = cache_dir / f"{sequence}_img{imgsz}_c008.npz"

    if cache.is_file():
        return np.load(cache)["detections"]

    img_dir = (
        ROOT / "datasets" / dataset
        / "train" / sequence / "img1"
    )
    if not img_dir.is_dir():
        raise FileNotFoundError(img_dir)

    records = []
    print(f"[YOLO CACHE] {dataset} {sequence} imgsz={imgsz}")

    results = model.predict(
        source=str(img_dir),
        stream=True,
        batch=8,
        imgsz=imgsz,
        conf=0.08,
        iou=0.70,
        max_det=300,
        classes=[0],
        device=0,
        verbose=False,
    )

    frame_count = 0
    for result in results:
        frame = int(Path(result.path).stem)
        frame_count += 1

        if result.boxes is None:
            continue

        xyxy = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()

        for box, score in zip(xyxy, scores):
            x1, y1, x2, y2 = map(float, box)
            if x2 <= x1 or y2 <= y1:
                continue

            records.append([
                frame,
                float(score),
                x1, y1, x2 - x1, y2 - y1,
            ])

        if frame_count % 200 == 0:
            print(f"[YOLO] {sequence} {frame_count} frames")

    data = np.asarray(
        records, dtype=float
    ).reshape(-1, 6)

    np.savez_compressed(cache, detections=data)
    print(f"[CACHE SAVED] {cache} ({len(data)} detections)")
    return data


def update_memory(memory, tid, frame, box):
    box = np.asarray(box, dtype=float).copy()

    if tid not in memory:
        memory[tid] = (
            frame, box, np.zeros(2, dtype=float)
        )
        return

    old_frame, old_box, old_velocity = memory[tid]
    dt = max(1, frame - old_frame)

    observed = (
        center(box) - center(old_box)
    ) / dt

    limit = max(box[3], 20.0) * 1.5

    if np.linalg.norm(observed) > limit:
        observed = old_velocity

    velocity = (
        0.50 * old_velocity
        + 0.50 * observed
    )

    memory[tid] = (frame, box, velocity)


def predicted_box(state, frame):
    last_frame, box, velocity = state
    dt = frame - last_frame

    predicted = box.copy()
    predicted[:2] += velocity * dt
    return predicted


def gated_cost(predicted, detected, dt):
    height_ratio = (
        max(predicted[3], detected[3])
        / max(min(predicted[3], detected[3]), 1.0)
    )

    if height_ratio > 1.65:
        return 1e6

    overlap = iou(predicted, detected)
    dist = np.linalg.norm(
        center(predicted) - center(detected)
    )
    normalized_dist = dist / max(
        predicted[3], detected[3], 20.0
    )

    # Do not reconnect distant or geometrically
    # inconsistent observations.
    max_dist = min(1.3, 0.40 + 0.10 * dt)

    if normalized_dist > max_dist:
        return 1e6

    if overlap < 0.08 and normalized_dist > 0.30:
        return 1e6

    return (
        0.60 * (1.0 - overlap)
        + 0.40 * normalized_dist
    )


def recover(base, detections, max_age, spawn_score, min_hits):
    baseline = group_frames(base)
    detected = group_frames(detections)

    memory = {}
    new_candidates = {}
    next_candidate = 0
    next_track_id = (
        int(np.max(base[:, 1])) + 1
        if len(base) else 1
    )

    recovered = []
    recovered_count = 0

    frames = sorted(set(baseline) | set(detected))

    for frame in frames:
        original = baseline.get(frame, [])
        candidates = detected.get(frame, [])

        original_ids = set()

        for row in original:
            tid = int(row[1])
            original_ids.add(tid)
            update_memory(
                memory, tid, frame, row[2:6]
            )

        # Exclude YOLO boxes that duplicate an existing
        # high-quality SR tracking box.
        available = []

        for item in candidates:
            det_box = np.asarray(item[2:6])

            if any(
                iou(det_box, row[2:6]) > 0.40
                for row in original
            ):
                continue

            available.append(item)

        # Stage A: recover a temporarily lost SR identity.
        lost_ids = [
            tid for tid, (seen, _, _) in memory.items()
            if tid not in original_ids
            and 0 < frame - seen <= max_age
        ]

        used_detections = set()

        if lost_ids and available:
            cost = np.full(
                (len(lost_ids), len(available)),
                1e6,
                dtype=float,
            )

            for i, tid in enumerate(lost_ids):
                state = memory[tid]
                dt = frame - state[0]
                predicted = predicted_box(state, frame)

                for j, det in enumerate(available):
                    score = float(det[1])

                    if score < 0.10:
                        continue

                    c = gated_cost(
                        predicted, det[2:6], dt
                    )

                    # Prefer reliable detections when
                    # geometric evidence is comparable.
                    cost[i, j] = c + 0.05 * (1 - score)

            rows, cols = linear_sum_assignment(cost)

            for i, j in zip(rows, cols):
                if cost[i, j] > 0.73:
                    continue

                tid = lost_ids[i]
                det = available[j]
                box = det[2:6]

                recovered.append(
                    make_row(
                        frame, tid, box, det[1]
                    )
                )

                update_memory(
                    memory, tid, frame, box
                )
                used_detections.add(j)
                recovered_count += 1

        # Stage B: optionally initialize a new track
        # only after repeated unmatched detections.
        if spawn_score is None:
            continue

        remainder = [
            item for j, item in enumerate(available)
            if j not in used_detections
            and item[1] >= spawn_score
            and item[5] >= 18
        ]

        active_candidates = [
            (cid, value)
            for cid, value in new_candidates.items()
            if 0 < frame - value["frame"] <= 2
        ]

        matched_detections = set()

        if active_candidates and remainder:
            cost = np.full(
                (len(active_candidates), len(remainder)),
                1e6,
            )

            for i, (_, state) in enumerate(active_candidates):
                for j, det in enumerate(remainder):
                    overlap = iou(
                        state["box"], det[2:6]
                    )

                    if overlap >= 0.22:
                        cost[i, j] = 1 - overlap

            rows, cols = linear_sum_assignment(cost)

            for i, j in zip(rows, cols):
                if cost[i, j] >= 0.78:
                    continue

                cid, state = active_candidates[i]
                det = remainder[j]

                state["frame"] = frame
                state["box"] = np.asarray(det[2:6]).copy()
                state["rows"].append(
                    (frame, state["box"].copy(), float(det[1]))
                )

                matched_detections.add(j)

        for j, det in enumerate(remainder):
            if j in matched_detections:
                continue

            new_candidates[next_candidate] = {
                "frame": frame,
                "box": np.asarray(det[2:6]).copy(),
                "rows": [
                    (frame, np.asarray(det[2:6]).copy(),
                     float(det[1]))
                ],
            }
            next_candidate += 1

    # Confirm candidate identities after seeing
    # enough observations. No GT is used.
    new_count = 0

    for state in new_candidates.values():
        rows = state["rows"]

        if len(rows) < min_hits:
            continue

        mean_conf = np.mean(
            [item[2] for item in rows]
        )

        if mean_conf < spawn_score:
            continue

        tid = next_track_id
        next_track_id += 1

        for frame, box, score in rows:
            recovered.append(
                make_row(frame, tid, box, score)
            )
            new_count += 1

    if recovered:
        return (
            np.vstack([base, np.asarray(recovered)]),
            recovered_count,
            new_count,
        )

    return base.copy(), 0, 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices=["MOT17", "MOT20", "both"],
        default="MOT17",
    )
    parser.add_argument(
        "--imgsz", type=int, default=960
    )
    args = parser.parse_args()

    datasets = (
        ["MOT17", "MOT20"]
        if args.dataset == "both"
        else [args.dataset]
    )

    weight = ROOT / "weights/yolo26n.pt"
    if not weight.is_file():
        raise FileNotFoundError(weight)

    model = YOLO(str(weight))

    for dataset in datasets:
        source_dir = (
            BASE / dataset / "trackers"
            / "srguardpp" / "data"
        )

        sequences = sorted(
            path.stem for path in source_dir.glob("*.txt")
        )
        expected = 7 if dataset == "MOT17" else 4

        if len(sequences) != expected:
            raise RuntimeError(
                f"Expected {expected} sequences, got "
                f"{len(sequences)} in {source_dir}"
            )

        for sequence in sequences:
            base = read_mot(
                source_dir / f"{sequence}.txt"
            )

            detections = get_yolo_cache(
                model, dataset, sequence, args.imgsz
            )

            for name, max_age, threshold, hits in PROFILES:
                result, recovered_n, new_n = recover(
                    base,
                    detections,
                    max_age=max_age,
                    spawn_score=threshold,
                    min_hits=hits,
                )

                target = (
                    OUT / dataset / "trackers"
                    / name / "data" / f"{sequence}.txt"
                )
                write_mot(target, result)

                original_stats = (
                    ROOT / "outputs/full_3way"
                    / dataset / "stats"
                    / "srstrongsort" / f"{sequence}.json"
                )

                target_stats = (
                    OUT / dataset / "stats"
                    / name / f"{sequence}.json"
                )
                target_stats.parent.mkdir(
                    parents=True, exist_ok=True
                )

                if not original_stats.is_file():
                    raise FileNotFoundError(original_stats)

                # Metadata for full-sequence validation only.
                shutil.copy2(original_stats, target_stats)

                print(
                    f"[{dataset} {sequence} {name}] "
                    f"recovered={recovered_n}, "
                    f"new_boxes={new_n}"
                )

    print(f"\n[DONE] {OUT}")


if __name__ == "__main__":
    main()
