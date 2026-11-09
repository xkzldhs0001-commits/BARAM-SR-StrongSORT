#!/usr/bin/env bash
set -Eeuo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${BARAM_PROJECT_DIR:-$HOME/Downloads/BARAM_SR_StrongSORT}"
PY="$ROOT/.venv/bin/python"
export PYTHONPATH="$ROOT/third_party/fast-reid:$ROOT:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export PYTHONUNBUFFERED=1
CMD="${1:-help}"
shift || true
case "$CMD" in
  test)
    "$PY" "$HERE/selftest.py"
    "$PY" "$HERE/selftest_occlusion.py" ;;
  check)
    "$PY" "$HERE/selftest.py"
    "$PY" "$HERE/selftest_occlusion.py"
    "$PY" "$HERE/live.py" --root "$ROOT" --check "$@" ;;
  install)
    "$PY" -m pip install --upgrade --no-deps 'ultralytics>=8.4.0' ;;
  sr|sr-full)
    # Stability-first full OSNet features, YOLO26x; no frame limit.
    "$PY" "$HERE/live.py" --root "$ROOT" --mode sr_osnet_full "$@" ;;
  sr-fast)
    # Research-oriented selective extractor; may have more ID fragmentation.
    "$PY" "$HERE/live.py" --root "$ROOT" --mode sr_osnet_selective "$@" ;;
  strongsort-osnet)
    "$PY" "$HERE/live.py" --root "$ROOT" --mode strongsort_osnet "$@" ;;
  strongsort)
    "$PY" "$HERE/live.py" --root "$ROOT" --mode strongsort_bot "$@" ;;
  help|--help)
    cat <<'EOF'
BARAM YOLO26x Live Occlusion Rescue V4.1

  bash run.sh check
  bash run.sh sr --camera 0 --show-internal-id        # stability-first (OSNet FULL)
  bash run.sh sr-fast --camera 0 --show-internal-id   # selective Re-ID; faster candidate
  bash run.sh strongsort-osnet --camera 0            # official StrongSORT+OSNet comparison
  bash run.sh sr --input ~/Videos/example.mp4 --record-annotated

Defaults: YOLO26x, conf=.40, tracker max_age=75, n_init=2,
          live motion gate=14.5, full OSNet in 'sr',
          internal second-pass appearance reattach, persistent display-ID bank.
          Unlimited runtime until Q or Ctrl+C.

Options:
  --no-live-relink        Disable second-pass internal tracking recovery
  --no-id-memory          Disable display ID bank (shows raw track ID)
  --id-threshold .59      One-person scene minimum similarity
  --id-multi-threshold .69    Multi-person scene minimum similarity
  --motion-gate 14.5      Live-only Kalman gating threshold
  --tracker-max-age 75
  --tracker-n-init 2
  --conf 0.40 --occlusion-low-conf 0.15 --imgsz 640
  --occlusion-max-missed 12 --predicted-hold 10
  --no-occlusion-rescue    Disable weak-box rescue for A/B
  Weak-score detections cannot create new IDs. PRED boxes are estimates.
  --show-internal-id
  --record-raw            Record same source for GT/fair A/B comparison

This version changes live-only behavior and DOES NOT overwrite MOT results.
EOF
    ;;
  *) echo "[ERROR] Unknown command: $CMD" >&2;exit 2 ;;
esac
