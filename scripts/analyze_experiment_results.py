#!/usr/bin/env python3
"""Compare reported benchmark metrics without mixing experiment sources."""

from __future__ import annotations
import argparse
import csv
from pathlib import Path

METRICS = ("hota", "idf1", "mota", "idsw", "fps", "reid_extractions", "peak_vram_mib")


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def compare(rows: list[dict[str, str]], dataset: str, baseline: str,
            method: str) -> dict[str, dict[str, float | None]]:
    matches = {}
    for label in (baseline, method):
        candidates = [row for row in rows if row["dataset"] == dataset and row["method"] == label]
        if len(candidates) != 1:
            raise ValueError(f"Expected exactly one {dataset}/{label} row, found {len(candidates)}")
        matches[label] = candidates[0]

    result = {}
    for name in METRICS:
        original = matches[baseline].get(name, "").strip()
        current = matches[method].get(name, "").strip()
        if not original or not current:
            continue
        base_value, new_value = float(original), float(current)
        result[name] = {
            "baseline": base_value,
            "candidate": new_value,
            "delta": new_value - base_value,
            "relative_percent": None if base_value == 0 else (new_value / base_value - 1) * 100,
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two rows within one benchmark source.")
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--csv", type=Path,
                        default=root / "experiments" / "poster_comparison.csv")
    parser.add_argument("--dataset", choices=("MOT17", "MOT20"), required=True)
    parser.add_argument("--baseline", default="StrongSORT")
    parser.add_argument("--method", default="SR-StrongSORT")
    args = parser.parse_args()
    results = compare(read_rows(args.csv), args.dataset, args.baseline, args.method)
    for name, values in results.items():
        rel = values["relative_percent"]
        pct = "N/A" if rel is None else f"{rel:+.2f}%"
        print(f"{name:20s} {values['baseline']:.3f} -> {values['candidate']:.3f}"
              f" | delta {values['delta']:+.3f} | relative {pct}")


if __name__ == "__main__":
    main()
