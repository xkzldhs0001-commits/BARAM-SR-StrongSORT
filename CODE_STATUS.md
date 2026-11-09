# Code inventory and reproducibility status

## On GitHub

The tracked `baram/`, `scripts/`, and `tests/` code files were recovered from a 2026-10-28 revision. Research summaries in `experiments/` and `PROJECT_REPORT_KO.md` reference **later YOLO26x and V4.3 selective Re-ID runs**. This older checkout alone is not a version-matched reproduction of those runs.

## Next integration: actual Ubuntu live source

Working-directory names reported during experiments:

- `~/Downloads/BARAM_LIVE_YOLO26X_OCCLUSION_V4_1`
- `~/Downloads/BARAM_SR_StrongSORT`

Files of interest include `run.sh`, `live.py`, `identity_memory.py`, `live_relink.py`, `occlusion_bridge.py`, MOT comparison workers, `baram_profile_worker_v43.py`, `run_baram_runtime_profile.py`, and `official_compare/strongsort_adapter.py`.

Upload the selected **actual current local files**, not a presumed equivalently named old ZIP, to the **BARAM-SR-StrongSORT** repository. Never upload them to the unrelated Jackal project.

## Verification checklist

1. Compare source checksums and existing modifications; preserve source and third-party attribution.
2. Exclude datasets, trained model weights, videos, performance output folders, private tokens, and venvs.
3. Validate Python syntax and run relevant unit tests.
4. Perform short GPU smoke tests with the actual YOLO26x + OSNet checkpoints.
5. State exact MOT split, timing boundaries and TrackEval configuration.
6. Reconcile differing historical FPS results before final scientific publication.

The November 6 TrackEval execution produced an error due to a sequence-map argument type; the reported HOTA/IDF1/MOTA comparisons should be independently matched to successful TrackEval output before being claimed as reproducible results.
