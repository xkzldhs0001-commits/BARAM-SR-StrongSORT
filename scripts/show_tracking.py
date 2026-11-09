from pathlib import Path
from collections import defaultdict
import argparse
import configparser
import csv

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="MOT17")
parser.add_argument("--seq", default="MOT17-04-FRCNN")
parser.add_argument("--frames", type=int, default=450)
args = parser.parse_args()

dataset = args.dataset
seq = args.seq

img_dir = ROOT / "datasets" / dataset / "train" / seq / "img1"

best_model = (
    "sr_spawn40_shield"
    if dataset == "MOT17"
    else "sr_spawn40_shieldpp"
)

sources = [
    (
        "StrongSORT++",
        ROOT / "outputs/full_3way" / dataset
        / "trackers/strongsortpp/data" / f"{seq}.txt",
    ),
    (
        "SR-StrongSORT BEST HOTA",
        ROOT / "outputs/recall_refine" / dataset
        / "trackers" / best_model / "data" / f"{seq}.txt",
    ),
    (
        "SR-StrongSORT Stable",
        ROOT / "outputs/recall_refine" / dataset
        / "trackers/sr_recover6_shieldpp/data" / f"{seq}.txt",
    ),
]

def load_tracks(path):
    tracks = defaultdict(list)

    if not path.is_file():
        raise FileNotFoundError(path)

    with path.open() as f:
        for row in csv.reader(f):
            if len(row) < 6:
                continue

            frame_id = int(float(row[0]))
            track_id = int(float(row[1]))
            x, y, w, h = map(float, row[2:6])

            tracks[frame_id].append(
                (track_id, x, y, w, h)
            )

    return tracks

loaded = [(name, load_tracks(path)) for name, path in sources]

images = sorted(img_dir.glob("*.jpg"))
if not images:
    images = sorted(img_dir.glob("*.png"))

if not images:
    raise FileNotFoundError(f"Dataset images not found: {img_dir}")

images = images[:args.frames]

sample = cv2.imread(str(images[0]))
if sample is None:
    raise RuntimeError("Cannot read dataset image")

original_h, original_w = sample.shape[:2]

panel_w = 640
panel_h = round(original_h * panel_w / original_w)
header_h = 36
scale = panel_w / original_w

config = configparser.ConfigParser()
config.read(img_dir.parent / "seqinfo.ini")
fps = config.getfloat("Sequence", "frameRate", fallback=30)

out_dir = ROOT / "outputs/preview"
out_dir.mkdir(parents=True, exist_ok=True)

output = out_dir / f"{seq}_compare.mp4"

writer = cv2.VideoWriter(
    str(output),
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (panel_w * len(loaded), panel_h + header_h),
)

if not writer.isOpened():
    raise RuntimeError("MP4 video writer failed to open")

for index, image_path in enumerate(images, 1):
    frame = cv2.imread(str(image_path))
    if frame is None:
        continue

    frame_id = int(image_path.stem)
    panels = []

    for name, tracks in loaded:
        panel = cv2.resize(frame, (panel_w, panel_h))

        current_tracks = tracks.get(frame_id, [])

        for tid, x, y, w, h in current_tracks:
            color = (
                40 + (tid * 37) % 210,
                40 + (tid * 73) % 210,
                40 + (tid * 113) % 210,
            )

            x1 = round(x * scale)
            y1 = round(y * scale)
            x2 = round((x + w) * scale)
            y2 = round((y + h) * scale)

            cv2.rectangle(
                panel, (x1, y1), (x2, y2), color, 2
            )

            cv2.putText(
                panel, f"ID {tid}",
                (max(0, x1), max(15, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42, color, 2,
            )

        header = np.zeros(
            (header_h, panel_w, 3), dtype=np.uint8
        )

        cv2.putText(
            header,
            f"{name} | Frame {frame_id} | Tracks {len(current_tracks)}",
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.53, (255, 255, 255), 1,
        )

        panels.append(np.vstack((header, panel)))

    writer.write(np.hstack(panels))

    if index % 100 == 0:
        print(f"[VIDEO] {index}/{len(images)}")

writer.release()

print(f"\n[SUCCESS] {output}")
