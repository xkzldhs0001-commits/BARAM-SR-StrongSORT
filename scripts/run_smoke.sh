#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source .venv/bin/activate; source scripts/env.sh
bash scripts/audit_upstream.sh
python -m unittest discover -s tests -v
python -m baram.run --dataset MOT17 --mode both --max-frames 40 --video --output outputs/smoke
