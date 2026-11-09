#!/usr/bin/env python3
"""Render existing MOTChallenge tracking outputs as a labeled comparison video."""
from __future__ import annotations

import argparse
import configparser
import csv
from pathlib import Path

import cv2
import numpy as np


def read_tracks(path: Path) -> dict[int, list[tuple[int, float, float, float, float]]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    frames: dict[int, list[tuple[int, float, float, float, float]]] = {}
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.reader(stream):
            if len(row) < 6:
                continue
            frame, track_id = int(float(row[0])), int(float(row[1]))
            x, y, width, height = map(float, row[2:6])
            if width > 0 and height > 0:
                frames.setdefault(frame, []).append((track_id, x, y, width, height))
    return frames


def color_for_id(track_id: int) -> tuple[int, int, int]:
    hue = (track_id * 37) % 180
    pixel = np.uint8([[[hue, 220, 255]]])
    b, g, r = cv2.cvtColor(pixel, cv2.COLOR_HSV2BGR)[0, 0]
    return int(b), int(g), int(r)


def draw_frame(image: np.ndarray, rows: list[tuple[int, float, float, float, float]], label: str) -> np.ndarray:
    canvas = image.copy()
    height, width = canvas.shape[:2]
    for track_id, x, y, box_w, box_h in rows:
        left, top = max(0, int(round(x))), max(0, int(round(y)))
        right = min(width - 1, int(round(x + box_w)))
        bottom = min(height - 1, int(round(y + box_h)))
        if right <= left or bottom <= top:
            continue
        color = color_for_id(track_id)
        cv2.rectangle(canvas, (left, top), (right, bottom), color, 2)
        cv2.putText(canvas, f"ID {track_id}", (left, max(18, top - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
    cv2.rectangle(canvas, (0, 0), (width, 34), (0, 0, 0), -1)    cv2.putText(canvas, label, (12, 24), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (255, 255, 255), 2, cv2.LINE_AA)
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--dataset", choices=["MOT17", "MOT20"], default="MOT17")
    parser.add_argument("--sequence", default="MOT17-02-FRCNN")
    parser.add_argument("--models", nargs="+", default=["strongsort", "strongsortpp", "srstrongsort"])
    parser.add_argument("--results", type=Path, default=Path("outputs/full_3way"))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--panel-width", type=int, default=640)
    args = parser.parse_args()
    if len(args.models) < 2:
        parser.error("provide at least two models for a comparison")

    root = args.project.resolve()
    results = args.results if args.results.is_absolute() else root / args.results
    sequence_dir = root / "datasets" / args.dataset / "train" / args.sequence
    image_dir = sequence_dir / "img1"
    if not image_dir.is_dir():
        raise FileNotFoundError(image_dir)
    trackers = {
        model: read_tracks(results / args.dataset / "trackers" / model / "data" / f"{args.sequence}.txt")
        for model in args.models
    }
    frames = sorted((p for p in image_dir.iterdir()
                     if p.suffix.lower() in {".jpg", ".png"} and p.stem.isdecimal()),
                    key=lambda p: int(p.stem))
    if not frames:
        raise RuntimeError(f"No frames found in {image_dir}")

    info = configparser.ConfigParser()
    info.read(sequence_dir / "seqinfo.ini")
    fps = info.getfloat("Sequence", "frameRate", fallback=30.0)
    first = cv2.imread(str(frames[0]))
    if first is None:
        raise RuntimeError(f"Cannot decode {frames[0]}")
    panel_height = max(2, int(round(first.shape[0] * args.panel_width / first.shape[1])))
    output = args.output or results / "video_comparison" / f"{args.sequence}-{'-vs-'.join(args.models)}.mp4"
    if not output.is_absolute():
        output = root / output
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps,
                             (args.panel_width * len(args.models), panel_height))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open video writer: {output}")
    try:
        for index, path in enumerate(frames, 1):
            image = cv2.imread(str(path))
            if image is None:
                raise RuntimeError(f"Cannot decode {path}")
            frame_number = int(path.stem)
            panels = []
            for model in args.models:
                label = f"{model.upper()} | frame {frame_number}"
                panel = draw_frame(image, trackers[model].get(frame_number, []), label)
                panel = cv2.resize(panel, (args.panel_width, panel_height), interpolation=cv2.INTER_AREA)
                panels.append(panel)
            writer.write(np.hstack(panels))
            if index % 500 == 0 or index == len(frames):
                print(f"[RENDER] {args.sequence}: {index}/{len(frames)}", flush=True)
    finally:
        writer.release()
    print(f"[DONE] {output}")


if __name__ == "__main__":
    main()
