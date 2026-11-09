from pathlib import Path
from dataclasses import dataclass
from collections import defaultdict
import argparse
import shutil

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs/full_3way"
OUTPUT = ROOT / "outputs/identity_shield"


@dataclass
class Identity:
    frame: int
    box: np.ndarray
    velocity: np.ndarray
    appearance: np.ndarray | None
    danger_until: int = 0


def center(box):
    x, y, w, h = box
    return np.array([x + w / 2, y + h / 2], dtype=float)


def iou(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[0] + a[2], b[0] + b[2])
    y2 = min(a[1] + a[3], b[1] + b[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    union = a[2] * a[3] + b[2] * b[3] - intersection
    return intersection / max(union, 1e-8)


def appearance_histogram(image, box):
    if image is None:
        return None

    height, width = image.shape[:2]
    x, y, w, h = map(float, box)

    # Torso crop: reduce background and leg contamination.
    x1 = max(0, int(x + 0.15 * w))
    x2 = min(width, int(x + 0.85 * w))
    y1 = max(0, int(y + 0.20 * h))
    y2 = min(height, int(y + 0.65 * h))

    if x2 - x1 < 6 or y2 - y1 < 6:
        return None

    crop = image[y1:y2, x1:x2]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    histogram = cv2.calcHist(
        [hsv],
        [0, 1],
        None,
        [16, 8],
        [0, 180, 0, 256],
    ).astype(np.float32)

    total = float(histogram.sum())
    if total <= 0:
        return None

    return histogram / total


def appearance_distance(a, b):
    if a is None or b is None:
        return 1.0

    return float(
        cv2.compareHist(
            a,
            b,
            cv2.HISTCMP_BHATTACHARYYA,
        )
    )


def motion_distance(state, box, frame):
    dt = max(1, frame - state.frame)

    predicted = (
        center(state.box)
        + state.velocity * dt
    )

    distance = np.linalg.norm(
        predicted - center(box)
    )

    scale = max(
        state.box[3],
        box[3],
        20.0,
    )

    return float(distance / scale)


def association_cost(state, box, hist, frame):
    appearance = appearance_distance(
        state.appearance,
        hist,
    )

    motion = motion_distance(
        state, box, frame
    )

    shape = abs(
        np.log(
            max(box[3], 1)
            / max(state.box[3], 1)
        )
    )

    return (
        0.78 * appearance
        + 0.30 * min(motion, 3.0)
        + 0.08 * min(shape, 2.0)
    )


def read_mot(path):
    if not path.is_file():
        raise FileNotFoundError(path)

    if path.stat().st_size == 0:
        return np.empty((0, 10), dtype=float)

    data = np.loadtxt(
        path, delimiter=",", ndmin=2
    )

    if data.shape[1] != 10:
        raise ValueError(f"Invalid MOT file: {path}")

    return data


def write_mot(path, data):
    path.parent.mkdir(
        parents=True, exist_ok=True
    )

    if not len(data):
        path.write_text("")
        return

    order = np.lexsort(
        (data[:, 1], data[:, 0])
    )
    data = data[order]

    np.savetxt(
        path,
        data,
        delimiter=",",
        fmt=[
            "%d", "%d",
            "%.2f", "%.2f", "%.2f", "%.2f",
            "%.6f", "%d", "%d", "%d"
        ],
    )


def load_image(image_dir, frame):
    for extension in (".jpg", ".png"):
        path = image_dir / f"{frame:06d}{extension}"
        if path.is_file():
            return cv2.imread(str(path))
    return None


def repair_identities(data, image_dir, max_lost=25):
    if not len(data):
        return data.copy(), 0, 0

    grouped = defaultdict(list)

    for row in data:
        grouped[int(row[0])].append(row.copy())

    # Original tracker IDs are only initial identity proposals.
    raw_to_identity = {}
    memories = {}

    next_id = int(np.max(data[:, 1])) + 1

    repaired = []
    swap_count = 0
    recovery_count = 0

    for frame in sorted(grouped):
        rows = np.asarray(grouped[frame])
        boxes = rows[:, 2:6].copy()
        raw_ids = rows[:, 1].astype(int)

        image = load_image(image_dir, frame)

        histograms = [
            appearance_histogram(image, box)
            for box in boxes
        ]

        n = len(rows)
        overlapping = np.zeros(n, dtype=bool)

        for i in range(n):
            for j in range(i + 1, n):
                if iou(boxes[i], boxes[j]) > 0.12:
                    overlapping[i] = True
                    overlapping[j] = True

        # 1. Recover lost identities when a new raw ID appears.
        for i, raw_id in enumerate(raw_ids):
            if raw_id in raw_to_identity:
                continue

            used = {
                raw_to_identity[int(r)]
                for r in raw_ids
                if int(r) in raw_to_identity
            }

            candidates = []

            for identity_id, state in memories.items():
                if identity_id in used:
                    continue

                dt = frame - state.frame
                if dt <= 0 or dt > max_lost:
                    continue

                if state.appearance is None:
                    continue

                app = appearance_distance(
                    state.appearance,
                    histograms[i],
                )
                motion = motion_distance(
                    state, boxes[i], frame
                )

                # Require both appearance and motion agreement.
                if app > 0.32:
                    continue

                if motion > 1.2 + 0.04 * dt:
                    continue

                score = 0.75 * app + 0.25 * motion

                if score < 0.48:
                    candidates.append(
                        (score, identity_id)
                    )

            if candidates:
                _, identity_id = min(candidates)

                # Discard previous raw alias of the recovered ID.
                for old_raw, old_identity in list(
                    raw_to_identity.items()
                ):
                    if old_identity == identity_id:
                        del raw_to_identity[old_raw]

                raw_to_identity[int(raw_id)] = identity_id
                recovery_count += 1
            else:
                proposed = int(raw_id)

                if proposed in memories:
                    proposed = next_id
                    next_id += 1

                raw_to_identity[int(raw_id)] = proposed

        # 2. Mark crossing/occlusion risk.
        for i in range(n):
            if not overlapping[i]:
                continue

            identity_id = raw_to_identity[int(raw_ids[i])]

            if identity_id in memories:
                memories[identity_id].danger_until = max(
                    memories[identity_id].danger_until,
                    frame + 18,
                )

        # 3. Test whether two existing IDs have been exchanged.
        # Use a high improvement margin to avoid unnecessary swaps.
        possible_swaps = []

        for i in range(n):
            for j in range(i + 1, n):
                id_a = raw_to_identity[int(raw_ids[i])]
                id_b = raw_to_identity[int(raw_ids[j])]

                if id_a == id_b:
                    continue

                if id_a not in memories or id_b not in memories:
                    continue

                state_a = memories[id_a]
                state_b = memories[id_b]

                recently_crossed = (
                    frame <= state_a.danger_until
                    or frame <= state_b.danger_until
                    or iou(boxes[i], boxes[j]) > 0.12
                )

                if not recently_crossed:
                    continue

                if any(
                    hist is None
                    for hist in (
                        state_a.appearance,
                        state_b.appearance,
                        histograms[i],
                        histograms[j],
                    )
                ):
                    continue

                normal = (
                    association_cost(
                        state_a, boxes[i],
                        histograms[i], frame
                    )
                    + association_cost(
                        state_b, boxes[j],
                        histograms[j], frame
                    )
                )

                swapped = (
                    association_cost(
                        state_a, boxes[j],
                        histograms[j], frame
                    )
                    + association_cost(
                        state_b, boxes[i],
                        histograms[i], frame
                    )
                )

                normal_app = (
                    appearance_distance(
                        state_a.appearance,
                        histograms[i],
                    )
                    + appearance_distance(
                        state_b.appearance,
                        histograms[j],
                    )
                )

                swapped_app = (
                    appearance_distance(
                        state_a.appearance,
                        histograms[j],
                    )
                    + appearance_distance(
                        state_b.appearance,
                        histograms[i],
                    )
                )

                # Both the total cost and appearance evidence
                # must favor exchanging these two identities.
                if (
                    swapped + 0.22 < normal
                    and swapped_app + 0.18 < normal_app
                ):
                    improvement = normal - swapped
                    possible_swaps.append(
                        (improvement, i, j)
                    )

        used_indices = set()

        for _, i, j in sorted(
            possible_swaps, reverse=True
        ):
            if i in used_indices or j in used_indices:
                continue

            raw_a = int(raw_ids[i])
            raw_b = int(raw_ids[j])

            raw_to_identity[raw_a], raw_to_identity[raw_b] = (
                raw_to_identity[raw_b],
                raw_to_identity[raw_a],
            )

            used_indices.update((i, j))
            swap_count += 1

        # 4. Update stable identity memory.
        current_ids = set()

        for i, row in enumerate(rows):
            raw_id = int(raw_ids[i])
            identity_id = raw_to_identity[raw_id]

            # Prevent duplicate corrected IDs in one frame.
            if identity_id in current_ids:
                identity_id = next_id
                next_id += 1
                raw_to_identity[raw_id] = identity_id

            current_ids.add(identity_id)

            box = boxes[i]
            hist = histograms[i]

            if identity_id not in memories:
                memories[identity_id] = Identity(
                    frame=frame,
                    box=box.copy(),
                    velocity=np.zeros(2),
                    appearance=(
                        None if overlapping[i] else hist
                    ),
                )
            else:
                state = memories[identity_id]
                dt = max(1, frame - state.frame)

                observed_velocity = (
                    center(box) - center(state.box)
                ) / dt

                if np.linalg.norm(observed_velocity) < (
                    2.0 * max(box[3], 20)
                ):
                    state.velocity = (
                        0.6 * state.velocity
                        + 0.4 * observed_velocity
                    )

                # Freeze appearance during and after crossings.
                safe_update = (
                    not overlapping[i]
                    and frame > state.danger_until
                    and hist is not None
                )

                if safe_update:
                    if state.appearance is None:
                        state.appearance = hist
                    else:
                        state.appearance = (
                            0.94 * state.appearance
                            + 0.06 * hist
                        )
                        state.appearance /= max(
                            float(state.appearance.sum()),
                            1e-8,
                        )

                state.frame = frame
                state.box = box.copy()

            corrected = row.copy()
            corrected[1] = identity_id
            repaired.append(corrected)

    return (
        np.asarray(repaired).reshape(-1, 10),
        swap_count,
        recovery_count,
    )


def interpolate(data, max_gap=3):
    if max_gap <= 0 or not len(data):
        return data.copy()

    extra = []
    ids = np.unique(data[:, 1].astype(int))

    for identity_id in ids:
        track = data[data[:, 1] == identity_id]
        track = track[np.argsort(track[:, 0])]

        for before, after in zip(
            track[:-1], track[1:]
        ):
            gap = int(after[0] - before[0]) - 1

            if not 1 <= gap <= max_gap:
                continue

            distance = np.linalg.norm(
                center(before[2:6])
                - center(after[2:6])
            )

            if distance > (
                0.8 * (gap + 1)
                * max(before[5], after[5], 20)
            ):
                continue

            for step in range(1, gap + 1):
                frame = int(before[0]) + step
                fraction = step / (gap + 1)

                box = (
                    (1 - fraction) * before[2:6]
                    + fraction * after[2:6]
                )

                others = data[
                    (data[:, 0] == frame)
                    & (data[:, 1] != identity_id)
                ]

                # Avoid inserting a synthetic box over another ID.
                if any(
                    iou(box, other[2:6]) > 0.45
                    for other in others
                ):
                    continue

                row = before.copy()
                row[0] = frame
                row[2:6] = box
                row[6] = 1.0
                row[7:10] = -1
                extra.append(row)

    if extra:
        return np.vstack(
            [data, np.asarray(extra)]
        )

    return data.copy()


def prepare_stats(dataset, model, sequence):
    source = (
        SOURCE / dataset / "stats"
        / "srstrongsort" / f"{sequence}.json"
    )

    target = (
        OUTPUT / dataset / "stats"
        / model / f"{sequence}.json"
    )

    if not source.is_file():
        raise FileNotFoundError(source)

    target.parent.mkdir(
        parents=True, exist_ok=True
    )
    shutil.copy2(source, target)


def run_dataset(dataset, device):
    original_dir = (
        SOURCE / dataset / "trackers"
        / "srstrongsort" / "data"
    )

    sequences = sorted(
        p.stem for p in original_dir.glob("*.txt")
    )

    expected = 7 if dataset == "MOT17" else 4
    if len(sequences) != expected:
        raise RuntimeError(
            f"{dataset}: expected {expected} sequences, "
            f"found {len(sequences)}"
        )

    guard_dir = (
        OUTPUT / dataset / "trackers"
        / "srguard" / "data"
    )
    gap_dir = (
        OUTPUT / dataset / "trackers"
        / "srguard_gap3" / "data"
    )

    guard_dir.mkdir(parents=True, exist_ok=True)
    gap_dir.mkdir(parents=True, exist_ok=True)

    for sequence in sequences:
        image_dir = (
            ROOT / "datasets" / dataset
            / "train" / sequence / "img1"
        )

        original = read_mot(
            original_dir / f"{sequence}.txt"
        )

        fixed, swaps, recoveries = repair_identities(
            original, image_dir
        )
        write_mot(
            guard_dir / f"{sequence}.txt",
            fixed
        )

        write_mot(
            gap_dir / f"{sequence}.txt",
            interpolate(fixed, max_gap=3)
        )

        for model in ("srguard", "srguard_gap3"):
            prepare_stats(dataset, model, sequence)

        print(
            f"[GUARD] {sequence} "
            f"swaps={swaps} "
            f"recovered_ids={recoveries}"
        )

    # AFLink and GSI are applied without changing SR matching.
    try:
        from baram.run import postprocess

        stage = OUTPUT / "_post_stage" / dataset
        stage_tracks = (
            stage / "trackers"
            / "strongsort" / "data"
        )
        stage_tracks.mkdir(
            parents=True, exist_ok=True
        )

        for sequence in sequences:
            shutil.copy2(
                guard_dir / f"{sequence}.txt",
                stage_tracks / f"{sequence}.txt",
            )

        postprocess(
            ROOT, stage, dataset,
            sequences, device
        )

        post_dir = (
            stage / "trackers"
            / "strongsortpp" / "data"
        )

        final_dir = (
            OUTPUT / dataset / "trackers"
            / "srguardpp" / "data"
        )
        final_dir.mkdir(
            parents=True, exist_ok=True
        )

        for sequence in sequences:
            source = post_dir / f"{sequence}.txt"
            if not source.is_file():
                raise FileNotFoundError(source)

            shutil.copy2(
                source,
                final_dir / f"{sequence}.txt",
            )
            prepare_stats(
                dataset, "srguardpp", sequence
            )

        print(f"[POST COMPLETE] {dataset} AFLink + GSI")

    except Exception as exc:
        print(f"[POST WARNING] {dataset}: {exc}")
        print("srguard and srguard_gap3 remain available")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        choices=["MOT17", "MOT20", "both"],
        default="both",
    )
    parser.add_argument(
        "--device", default="cuda:0"
    )

    args = parser.parse_args()

    datasets = (
        ["MOT17", "MOT20"]
        if args.dataset == "both"
        else [args.dataset]
    )

    for dataset in datasets:
        run_dataset(dataset, args.device)

    print(f"\n[DONE] {OUTPUT}")


if __name__ == "__main__":
    main()
