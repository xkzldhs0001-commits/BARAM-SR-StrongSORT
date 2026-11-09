#!/usr/bin/env bash
# Run or source from the project root after creating the virtual environment.
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/third_party/StrongSORT:${PROJECT_ROOT}/third_party/fast-reid:${PROJECT_ROOT}/third_party/deep-person-reid:${PROJECT_ROOT}/third_party/StrongSORT/AFLink:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
