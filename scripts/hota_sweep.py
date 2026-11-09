from pathlib import Path
import argparse
import csv
import shutil
import subprocess
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs" / "full_3way"
OUTPUT = ROOT / "outputs" / "hota_sweep"

FMT = "%d,%d,%.2f,%.2f,%.2f,%.2f,%.2f,%d,%d,%d"

# (candidate name, base, interpolation gap, min track length, smoothing)
VARIANTS = [
    ("sr_original", "raw", 0, 0, False),
    ("sr_gap2", "raw", 2, 0, False),
    ("sr_gap5", "raw", 5, 0, False),
    ("sr_gap10", "raw", 10, 0, False),
    ("sr_prune2", "raw", 0, 2, False),
    ("sr_prune4", "raw", 0, 4, False),
    ("sr_smooth", "raw", 0, 0, True),
    ("sr_gap5_smooth", "raw", 5, 0, True),
    ("sr_gap5_prune2", "raw", 5, 2, False),
    ("sr_aflink_gsi", "post", 0, 0, False),
    ("sr_aflink_gsi_gap5", "post", 5, 0, False),
    ("sr_aflink_gsi_prune2", "post", 0, 2, False),
]


def load_mot(path):
    if not path.exists():
        raise FileNotFoundError(path)

    if path.stat().st_size == 0:
        return np.empty((0, 10), dtype=float)

    data = np.loadtxt(path, delimiter=",", ndmin=2)
    if data.shape[1] != 10:
        raise ValueError(f"Invalid MOT format: {path}")

    return data.astype(float)


def save_mot(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)

    if len(data) == 0:
        path.write_text("")
        return

    data = data[np.lexsort((data[:, 1], data[:, 0]))]
    np.savetxt(path, data, delimiter=",", fmt=FMT)


def prune_short(data, min_length):
    if not len(data) or min_length <= 1:
        return data.copy()

    ids, counts = np.unique(
        data[:, 1].astype(int), return_counts=True
    )
    keep = ids[counts >= min_length]
    return data[np.isin(data[:, 1], keep)].copy()


def interpolate_short_gaps(data, max_missing):
    if not len(data) or max_missing <= 0:
        return data.copy()

    extra = []

    for track_id in np.unique(data[:, 1]):
        rows = data[data[:, 1] == track_id]
        rows = rows[np.argsort(rows[:, 0])]

        for before, after in zip(rows[:-1], rows[1:]):
            missing = int(after[0] - before[0]) - 1

            if not 1 <= missing <= max_missing:
                continue

            for step in range(1, missing + 1):
                fraction = step / (missing + 1)
                row = before.copy()

                row[0] = before[0] + step
                row[2:6] = (
                    (1 - fraction) * before[2:6]
                    + fraction * after[2:6]
                )
                row[6] = 1.0
                row[7:10] = -1

                if np.all(row[4:6] > 0):
                    extra.append(row)

    if not extra:
        return data.copy()

    return np.vstack([data, np.asarray(extra)])


def smooth_tracks(data):
    result = data.copy()

    for track_id in np.unique(data[:, 1]):
        indices = np.where(data[:, 1] == track_id)[0]
        indices = indices[np.argsort(data[indices, 0])]

        for pos in range(1, len(indices) - 1):
            left, mid, right = indices[pos - 1:pos + 2]

            if (
                data[mid, 0] - data[left, 0] != 1
                or data[right, 0] - data[mid, 0] != 1
            ):
                continue

            result[mid, 2:6] = (
                0.2 * data[left, 2:6]
                + 0.6 * data[mid, 2:6]
                + 0.2 * data[right, 2:6]
            )

    return result


def transform(data, gap, min_length, smooth):
    result = prune_short(data, min_length)
    result = interpolate_short_gaps(result, gap)

    if smooth:
        result = smooth_tracks(result)

    return result


def run_original_aflink_gsi(dataset, sequences, device):
    """Apply ORIGINAL author AFLink/GSI to SR tracks in a staging folder.

    No original StrongSORT/SR files are modified.
    """
    from baram.run import postprocess

    stage = OUTPUT / "_staging" / dataset
    stage_tracks = (
        stage / "trackers" / "strongsort" / "data"
    )
    stage_tracks.mkdir(parents=True, exist_ok=True)

    for seq in sequences:
        src = (
            SOURCE / dataset / "trackers"
            / "srstrongsort" / "data" / f"{seq}.txt"
        )
        shutil.copy2(src, stage_tracks / f"{seq}.txt")

    print(f"\n[POSTPROCESS] {dataset}: SR -> AFLink -> GSI")
    postprocess(ROOT, stage, dataset, sequences, device)

    result_dir = (
        stage / "trackers" / "strongsortpp" / "data"
    )

    for seq in sequences:
        if not (result_dir / f"{seq}.txt").is_file():
            raise FileNotFoundError(
                f"AFLink/GSI output not found: {result_dir / seq}"
            )

    return result_dir


def read_summary(path):
    lines = path.read_text().strip().splitlines()

    if len(lines) < 2:
        raise ValueError(f"Invalid TrackEval summary: {path}")

    return dict(zip(lines[0].split(), lines[1].split()))


def run_dataset(dataset, device):
    raw_dir = (
        SOURCE / dataset / "trackers"
        / "srstrongsort" / "data"
    )

    sequences = sorted(p.stem for p in raw_dir.glob("*.txt"))

    expected = 7 if dataset == "MOT17" else 4
    if len(sequences) != expected:
        raise RuntimeError(
            f"{dataset}: expected {expected} sequences, "
            f"found {len(sequences)}"
        )

    post_dir = None
    try:
        post_dir = run_original_aflink_gsi(
            dataset, sequences, device
        )
    except Exception as exc:
        print(f"[WARNING] AFLink/GSI unavailable: {exc}")
        print("[INFO] Continuing with non-AFLink candidates")

    candidates = []

    for name, base, gap, min_length, smooth in VARIANTS:
        if base == "post" and post_dir is None:
            continue

        src_dir = raw_dir if base == "raw" else post_dir

        target_dir = (
            OUTPUT / dataset / "trackers"
            / name / "data"
        )
        target_dir.mkdir(parents=True, exist_ok=True)

        for seq in sequences:
            data = load_mot(src_dir / f"{seq}.txt")
            improved = transform(
                data, gap, min_length, smooth
            )
            save_mot(
                target_dir / f"{seq}.txt",
                improved
            )

        # TrackEval wrapper requires full-sequence processing metadata.
        # These are inherited from the original SR run.
        # They are NOT new FPS or Re-ID measurements.
        reference = SOURCE / dataset / "stats" / "srstrongsort"
        target = OUTPUT / dataset / "stats" / name
        target.mkdir(parents=True, exist_ok=True)

        for seq in sequences:
            src = reference / f"{seq}.json"
            dst = target / f"{seq}.json"

            if not src.is_file():
                raise FileNotFoundError(
                    f"Original SR metadata missing: {src}"
                )

            shutil.copy2(src, dst)

        candidates.append(name)
        print(f"[PREPARED] {dataset} {name}")

    command = [
        sys.executable,
        str(ROOT / "scripts" / "evaluate.py"),
        "--project", str(ROOT),
        "--output", str(OUTPUT),
        "--benchmark", dataset,
        "--models", *candidates,
    ]

    print(f"\n[TRACK EVAL] {dataset}: {len(candidates)} variants")
    subprocess.run(command, cwd=ROOT, check=True)

    results = []

    for name in candidates:
        summary = (
            OUTPUT / dataset / "trackers"
            / name / "pedestrian_summary.txt"
        )
        metrics = read_summary(summary)

        results.append({
            "dataset": dataset,
            "model": name,
            "HOTA": float(metrics["HOTA"]),
            "MOTA": float(metrics["MOTA"]),
            "IDF1": float(metrics["IDF1"]),
            "IDSW": int(float(metrics["IDSW"])),
            "DetA": float(metrics["DetA"]),
            "AssA": float(metrics["AssA"]),
        })

    results.sort(
        key=lambda r: (r["HOTA"], r["IDF1"]),
        reverse=True,
    )

    print(f"\n===== {dataset} TOP CANDIDATES =====")
    for row in results[:5]:
        print(
            f"{row['model']:28s} "
            f"HOTA={row['HOTA']:.3f} "
            f"IDF1={row['IDF1']:.3f} "
            f"IDSW={row['IDSW']}"
        )

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices=["MOT17", "MOT20", "both"],
        default="both",
    )
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    datasets = (
        ["MOT17", "MOT20"]
        if args.dataset == "both"
        else [args.dataset]
    )

    all_results = []
    for dataset in datasets:
        all_results.extend(
            run_dataset(dataset, args.device)
        )

    output_csv = OUTPUT / "HOTA_SWEEP_RESULTS.csv"
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with output_csv.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "dataset", "model", "HOTA", "MOTA",
                "IDF1", "IDSW", "DetA", "AssA"
            ],
        )
        writer.writeheader()
        writer.writerows(all_results)

    print(f"\n[DONE] {output_csv}")


if __name__ == "__main__":
    main()
