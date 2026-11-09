# BARAM — SR-StrongSORT

**Selective Re-Identification for Multi-Person Tracking**

Research project exploring whether unnecessary OSNet appearance-feature extraction can be reduced in a StrongSORT-based pedestrian tracker while retaining identity continuity. The later live experiments combine **YOLO26x**, motion/IoU/Mahalanobis gating, selective OSNet feature extraction, identity memory, live relinking, and occlusion handling.

## Source version: important distinction

- **Published now:** `baram/`, `scripts/`, `tests/` from a recovered **October 2026 source snapshot**, plus documentation and benchmark summaries.
- **Still to be uploaded from the tested Ubuntu working directory:** YOLO26x / **V4.3 live** sources, benchmark workers, runtime-profiler workers.
- The archived source **does not reproduce the later V4.3 results by itself**. See [CODE_STATUS.md](CODE_STATUS.md).

This repository is **independent from** the Jackal/Gazebo multi-human simulation project.

## Research methods

1. YOLO pedestrian detection and Kalman prediction.
2. Match-candidate gating with motion and geometric cues.
3. OSNet appearance embeddings for selected ambiguous detections.
4. Hungarian association, optional ID memory/relinking, and low-confidence occlusion rescue.
5. Controlled StrongSORT vs selective SR-StrongSORT experiments on MOT17 and MOT20.

## Reported quantitative results (later YOLO26x/V4.3 configuration)

MOT17 and MOT20 **training split**, not the official test leaderboard:

| Dataset | Method | HOTA ↑ | IDF1 ↑ | MOTA ↑ | IDSW ↓ | Re-ID crops ↓ | Peak VRAM (MiB) ↓ |
|---|---|---:|---:|---:|---:|---:|---:|
| MOT17 | StrongSORT | 39.74 | 45.83 | 32.67 | 243 | 47,881 | 822 |
| MOT17 | SR-StrongSORT | 37.12 | 43.02 | 31.75 | 359 | 27,865 | 766 |
| MOT20 | StrongSORT | 10.38 | 7.67 | 4.33 | 232 | 59,283 | 1,034 |
| MOT20 | SR-StrongSORT | 12.95 | 11.17 | 7.99 | 381 | 29,478 | 872 |

- Re-ID extractions decreased **41.8% on MOT17** and **50.3% on MOT20**.
- GPU memory decreased **6.8%** and **15.7%**, respectively.
- MOT20 accuracy improved in these comparisons, but MOT17 accuracy fell; **ID switches increased in both datasets**.
- Reproducibility caveat: these numbers reflect reported poster/evaluation summaries from subsequent code revisions. They must not be attributed to a successful run of the older source snapshot. The November 6 TrackEval execution encountered a seqmap-related error, as recorded in the research report.

## Runtime bottleneck study

Mean stage latency (ms/frame) from saved per-frame logs:

| Dataset | Stage | StrongSORT | SR-StrongSORT |
|---|---|---:|---:|
| MOT17 | YOLO26x detection | 31.38 | 30.54 |
| MOT17 | OSNet Re-ID | 17.77 | 11.34 |
| MOT17 | Prediction & gating | 0.26 | 2.22 |
| MOT17 | Data association | 1.35 | 1.96 |
| MOT17 | ID management | 0.00 | 2.59 |
| MOT20 | YOLO26x detection | 34.81 | 32.85 |
| MOT20 | OSNet Re-ID | 14.54 | 9.01 |
| MOT20 | Prediction & gating | 0.27 | 4.40 |
| MOT20 | Data association | 1.29 | 2.97 |
| MOT20 | ID management | 0.00 | 5.00 |

Selective OSNet reduced Re-ID stage latency by approximately **36.2% (MOT17)** and **38.0% (MOT20)**, while additional gating/ID handling consumed time. Detection was the largest measured stage in both methods. Without controlled ablations, latency differences should not be interpreted as definitive causal effects.

## Directory guide

- `baram/` — archived tracker, appearance extraction, association pipeline.
- `scripts/` — archived evaluation, diagnostic and visualization utilities.
- `tests/` — regression/unit tests for the archived source.
- `experiments/` — CSV comparisons and provenance notes.
- [PROJECT_REPORT_KO.md](PROJECT_REPORT_KO.md) — Korean research background, methods, results and limitations.
- [CODE_STATUS.md](CODE_STATUS.md) — exact revision and reproducibility status.

## Archived snapshot setup

```bash
bash scripts/setup_ubuntu.sh
source .venv/bin/activate
source scripts/env.sh
python -m unittest discover -s tests -v
```

The setup depends on separately installed upstream repositories, dataset files, and model checkpoints. The current live V4.3 runtime is **not yet covered** by the archived checkout.

## Data, weights and licenses

MOT images/labels, videos, exported results, checkpoints such as `yolo26x.pt` and `osnet_x1_0.pth.tar`, virtual environments and secrets are intentionally excluded. The project integrates third-party StrongSORT/OSNet/YOLO and TrackEval; their individual licenses and attribution requirements still apply.
