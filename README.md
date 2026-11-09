# SR-StrongSORT — Selective Re-ID Tracking

Multi-object tracking project using YOLO pedestrian detections, StrongSORT-based motion prediction, and selective appearance feature extraction.

## Project report and results

- [Korean project report: research purpose, methods, quantitative results, and limitations](PROJECT_REPORT_KO.md)
- [Experiment CSVs and comparison script](experiments/README.md)
- [Source revision inventory and integration status](CODE_STATUS.md)

The later V4.3/V4.4/V4.5 ZIP packages are not included in this source snapshot. The benchmark summaries describe later runs and should not be presented as results reproduced by the older checkout.

## Architecture

1. Detect pedestrians using YOLO26n and cache detections for reproducible comparisons.
2. Predict track locations with the StrongSORT Kalman filter.
3. Estimate association ambiguity from overlap, aspect-ratio similarity, and motion distance.
4. Extract OSNet embeddings selectively for ambiguous associations; otherwise rely on motion and geometry.
5. Match detections to tracks and maintain appearance memory for identity recovery.
6. Export MOT-format tracking results, performance summaries, and optional visualizations.

## Repository layout

- `baram/`: detection, StrongSORT integration, appearance features, association logic, and experiment runner.
- `scripts/`: Ubuntu setup, benchmark evaluation, post-processing, diagnostics, and visualization utilities.
- `tests/`: geometry, matching, and tracker behavior tests.

## Requirements

Ubuntu 22.04, Python 3.10+, a CUDA-capable NVIDIA GPU for the official StrongSORT/BoT-FastReID/StrongSORT++ runs, and the pretrained checkpoints used in the original experiments.

Prepare the project dependencies:

```bash
bash scripts/setup_ubuntu.sh
source .venv/bin/activate
source scripts/env.sh
bash scripts/check_env.sh
```

The setup script clones the upstream StrongSORT, FastReID, TrackEval, and Torchreid code into `third_party/`. You must supply the external pretrained checkpoints and MOT datasets separately. Large models and datasets are deliberately not tracked in Git.

Expected checkpoint names in `weights/`: `yolo26n.pt`, `osnet_x1_0.pth.tar`, `duke_bot_S50.pth` and (for StrongSORT++) `AFLink_epoch20.pth`.

Expected benchmark directories: `datasets/MOT17/train/` and `datasets/MOT20/train/`.

## Run

Quick tracker checks:

```bash
python -m unittest discover -s tests -v
bash scripts/run_smoke.sh
```

Comparison on a short MOT17 segment:

```bash
python -m baram.run --dataset MOT17 --mode both --max-frames 100 --video
```

Full benchmark workflow:

```bash
bash scripts/run_full.sh
```

The full workflow requires prepared models and datasets. Official ECC JSON files are also needed for full StrongSORT camera-motion compensation.

## Evaluation notes

The scripts can run controlled comparisons between the original StrongSORT implementation and the selective Re-ID tracker. Benchmark results must be calculated from completed sequences. The project does not include trained checkpoint binaries, downloaded MOT datasets, or benchmark output videos.

**Source snapshot:** These source files were consolidated from the recoverable October 28, 2026 code snapshot. Subsequent V4.3, V4.4, and V4.5 experiment packages were produced separately; those archived changes are not represented as integrated code in this snapshot.

## Dependencies and attribution

The tracker integrates third-party StrongSORT, FastReID, TrackEval, and Torchreid implementations. Their licenses and attribution requirements remain applicable. This repository contains integration and experiment code, not vendored copies of the upstream implementations.
