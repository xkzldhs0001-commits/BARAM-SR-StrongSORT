from pathlib import Path
import argparse
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from identity_shield import (
    read_mot,
    write_mot,
    repair_identities,
)
from baram.run import postprocess

SOURCE = ROOT / "outputs/recall_boost"
OUTPUT = ROOT / "outputs/recall_refine"

MODELS = [
    "sr_spawn40",
    "sr_spawn55",
    "sr_recover6",
]


def evaluate(dataset, candidates):
    command = [
        sys.executable,
        str(ROOT / "scripts/evaluate.py"),
        "--project", str(ROOT),
        "--output", str(OUTPUT),
        "--benchmark", dataset,
        "--models", *candidates,
    ]
    subprocess.run(command, cwd=ROOT, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices=["MOT17", "MOT20", "both"],
        default="MOT17",
    )
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    datasets = (
        ["MOT17", "MOT20"]
        if args.dataset == "both"
        else [args.dataset]
    )

    for dataset in datasets:
        candidates = []
        expected = 7 if dataset == "MOT17" else 4

        for model in MODELS:
            source_dir = (
                SOURCE / dataset / "trackers"
                / model / "data"
            )

            sequences = sorted(
                p.stem for p in source_dir.glob("*.txt")
            )

            if len(sequences) != expected:
                print(f"[SKIP] {model}: incomplete inputs")
                continue

            shield = f"{model}_shield"
            shieldpp = f"{model}_shieldpp"

            shield_dir = (
                OUTPUT / dataset / "trackers"
                / shield / "data"
            )

            stage = (
                OUTPUT / "_stage" / dataset / model
            )
            stage_dir = (
                stage / "trackers/strongsort/data"
            )

            shield_dir.mkdir(parents=True, exist_ok=True)
            stage_dir.mkdir(parents=True, exist_ok=True)

            for sequence in sequences:
                images = (
                    ROOT / "datasets" / dataset
                    / "train" / sequence / "img1"
                )

                data = read_mot(
                    source_dir / f"{sequence}.txt"
                )

                corrected, swaps, recovered = (
                    repair_identities(
                        data, images, max_lost=25
                    )
                )

                shield_file = (
                    shield_dir / f"{sequence}.txt"
                )

                write_mot(shield_file, corrected)

                shutil.copy2(
                    shield_file,
                    stage_dir / f"{sequence}.txt",
                )

                original_stats = (
                    ROOT / "outputs/full_3way"
                    / dataset / "stats"
                    / "srstrongsort"
                    / f"{sequence}.json"
                )

                target_stats = (
                    OUTPUT / dataset / "stats"
                    / shield / f"{sequence}.json"
                )
                target_stats.parent.mkdir(
                    parents=True, exist_ok=True
                )
                shutil.copy2(
                    original_stats, target_stats
                )

                print(
                    f"[SHIELD] {dataset} {model} "
                    f"{sequence} swaps={swaps} "
                    f"reconnected={recovered}",
                    flush=True,
                )

            candidates.append(shield)

            # Apply official AFLink and GSI.
            try:
                postprocess(
                    ROOT,
                    stage,
                    dataset,
                    sequences,
                    args.device,
                )

                post_dir = (
                    stage / "trackers"
                    / "strongsortpp" / "data"
                )

                final_dir = (
                    OUTPUT / dataset / "trackers"
                    / shieldpp / "data"
                )
                final_dir.mkdir(
                    parents=True, exist_ok=True
                )

                for sequence in sequences:
                    src = post_dir / f"{sequence}.txt"

                    if not src.is_file():
                        raise FileNotFoundError(src)

                    shutil.copy2(
                        src,
                        final_dir / f"{sequence}.txt",
                    )

                    stats_src = (
                        OUTPUT / dataset / "stats"
                        / shield / f"{sequence}.json"
                    )
                    stats_dst = (
                        OUTPUT / dataset / "stats"
                        / shieldpp / f"{sequence}.json"
                    )
                    stats_dst.parent.mkdir(
                        parents=True, exist_ok=True
                    )
                    shutil.copy2(stats_src, stats_dst)

                candidates.append(shieldpp)
                print(f"[POST OK] {dataset} {shieldpp}")

            except Exception as exc:
                print(f"[POST FAILED] {model}: {exc}")

        if not candidates:
            raise RuntimeError(
                f"No valid candidates for {dataset}"
            )

        evaluate(dataset, candidates)

        results = []

        for name in candidates:
            summary = (
                OUTPUT / dataset / "trackers"
                / name / "pedestrian_summary.txt"
            )

            lines = summary.read_text().splitlines()
            metrics = dict(
                zip(lines[0].split(), lines[1].split())
            )

            results.append((
                float(metrics["HOTA"]),
                float(metrics["IDF1"]),
                int(float(metrics["IDSW"])),
                name,
            ))

        results.sort(reverse=True)

        print(f"\n===== {dataset} HOTA RANKING =====")
        for hota, idf1, idsw, name in results:
            print(
                f"{name:30s} "
                f"HOTA={hota:.3f} "
                f"IDF1={idf1:.3f} "
                f"IDSW={idsw}"
            )


if __name__ == "__main__":
    main()
