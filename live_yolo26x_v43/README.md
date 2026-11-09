# YOLO26x / V4.3 live tracking source

Selected code copied from the actually used Ubuntu working directories,
rather than reconstructed from a historical source snapshot.

## Components

- `live.py`, `run.sh`: live multi-person tracking and launch script.
- `identity_memory.py`, `live_relink.py`: appearance/ID memory and relinking.
- `occlusion_bridge.py`: weak-confidence occlusion-rescue candidates.
- `run_baram_ab.py`, `baram_compare_worker.py`: StrongSORT vs selective benchmark.
- `run_baram_runtime_profile.py`, `baram_profile_worker_v43.py`: runtime profiling.
- `official_compare/`, `srstrongsort/`: locally maintained bridge/model modules.

Only files found in the user's working directories are copied; missing optional
modules remain missing. Run the code with an external dependency environment,
the original `BARAM_SR_StrongSORT` tree, and external YOLO26x/OSNet weights.
It is not a standalone vendored distribution of StrongSORT.

For representative live runs in the original working directory:

```bash
bash run.sh sr-fast --camera 2 --show-internal-id --record-annotated --record-raw
bash run.sh sr --camera 2 --show-internal-id --record-annotated --record-raw
```

The research compares baseline StrongSORT with V4.3 selective Re-ID on the
MOT17/MOT20 *training* sequences. See the root research report for reported
metrics and important provenance limitations.
