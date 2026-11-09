# Source revision inventory

## Code currently included

- `baram/*.py`, `scripts/*.py`/`*.sh`, `tests/*.py`: source files from the recovered 2026-10-28 SR-StrongSORT code snapshot.
- `scripts/analyze_experiment_results.py`: tabular metric-comparison utility added separately; it does not alter the tracker.
- The experiment results documented here belong to subsequent YOLO26x configurations and **cannot be used to claim that the older snapshot reproduces those scores**.

## Later packages identified

| Archive | Purpose | Original bytes |
|---|---|---|
| `BARAM_V4_3_VSCODE_CODE.zip` | V4.3 code package | Not accessible for source extraction |
| `BARAM_V44_IDSW_GUARD_AB.zip` | V4.4 ID switch guard tests | Not accessible for source extraction |
| `BARAM_V45_DIAGNOSTIC_ABLATION.zip` | V4.5 diagnostic ablation | Not accessible for source extraction |
| `BARAM_LIVE_YOLO26X_OCCLUSION_V4_1.zip` | Live tracking and occlusion handling | Not accessible for source extraction |
| `BARAM_SR_YOLO26X_STABLE_V6.zip` | Stable YOLO26x integration | Not accessible for source extraction |
| `BARAM_YOLO26X_MOT17_MOT20_AB.zip` | Benchmark runner | Not accessible for source extraction |
| `BARAM_MOT_SAME_FRAME_VIS.zip` | Matched-frame visualization | Not accessible for source extraction |

The archive records are present in the project file library, but source extraction is unavailable. **No file from these archives has been passed off as integrated original source.**

## Checklist for integrating a recovered revision

1. Verify the ZIP checksum and inspect its file list.
2. Compare the revision with the prior code, one file at a time.
3. Retain upstream dependency notices and avoid committing pretrained models, datasets, generated videos, or local secrets.
4. Run syntax tests, unit tests, a short MOT sample, and a full version-matched TrackEval before changing the published benchmark claim.
5. Record the exact checkpoint, dataset split, run command, FPS timing boundaries, and revision commit.

## Known benchmark issue

The 2026-11-06 TrackEval run aborted because `SEQMAP_FILE` was interpreted as a list where a path string was expected. The console log preserved performance counters but not valid TrackEval metrics from that failed evaluation. See `PROJECT_REPORT_KO.md`.
