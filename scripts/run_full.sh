#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT";source .venv/bin/activate;source scripts/env.sh
bash scripts/audit_upstream.sh
# For full original StrongSORT (BoT+ECC+NSA+EMA+MC+woC), official ECC JSONs required.
ECC_DIR="${BARAM_ECC_DIR:-$ROOT/weights}"
python -m baram.run --dataset both --mode all --device cuda:0 --conf 0.25 --imgsz 640 \
  --ecc-dir "$ECC_DIR" --require-ecc --output outputs/full_3way "$@"
# TrackEval does not run on partial/smoke frames.
python scripts/evaluate.py --project "$ROOT" --output outputs/full_3way --benchmark both --models strongsort strongsortpp srstrongsort
python scripts/collect.py --output outputs/full_3way
