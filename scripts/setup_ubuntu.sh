#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p third_party weights datasets outputs
if [[ ! -e third_party/StrongSORT ]]; then
 git clone https://github.com/dyhBUPT/StrongSORT.git third_party/StrongSORT
 git -C third_party/StrongSORT checkout ee995076da5083e28d0da1f885297df62705ebd7
fi
if [[ ! -e third_party/fast-reid ]]; then
 git clone https://github.com/JDAI-CV/fast-reid.git third_party/fast-reid
fi
if [[ ! -e third_party/TrackEval ]]; then
 git clone https://github.com/JonathonLuiten/TrackEval.git third_party/TrackEval
fi
if [[ ! -e third_party/deep-person-reid ]]; then
 git clone https://github.com/KaiyangZhou/deep-person-reid.git third_party/deep-person-reid
fi
# Existing Ubuntu user has already installed torch + NVIDIA driver + GPU build.
if [[ ! -d .venv ]]; then python3 -m venv .venv;fi
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install 'numpy>=1.24,<2.3' 'scipy>=1.10' 'scikit-learn>=1.3' 'opencv-python>=4.8' 'ultralytics>=8.4' 'PyYAML>=6' 'yacs' 'termcolor' 'tabulate' 'faiss-cpu' 'gdown' 'matplotlib' 'tqdm' 'Pillow' 'filterpy' 'tensorboard' 'flake8'
# Install repository source as importable paths without modifying source files.
python -m pip install --no-build-isolation -e third_party/fast-reid || echo '[INFO] FastReID editable install failed; PYTHONPATH fallback is used (see scripts/env.sh).'
python -m pip install --no-build-isolation -e third_party/deep-person-reid || echo '[INFO] torchreid fallback via PYTHONPATH.'
# Reuse previously trained checkpoints, but never create fake model weights.
LEGACY="$HOME/Downloads/BARAM_SR_StrongSORT"
for f in yolo26n.pt osnet_x1_0.pth.tar duke_bot_S50.pth AFLink_epoch20.pth; do
 if [[ ! -e "weights/$f" ]]; then
  if [[ -s "$LEGACY/weights/$f" ]]; then ln -s "$LEGACY/weights/$f" "weights/$f";
  elif [[ -s "$LEGACY/$f" ]]; then ln -s "$LEGACY/$f" "weights/$f";fi
 fi
done
if [[ ! -e datasets/MOT17 && -d "$LEGACY/datasets/MOT17" ]];then ln -s "$LEGACY/datasets/MOT17" datasets/MOT17;fi
if [[ ! -e datasets/MOT20 && -d "$LEGACY/datasets/MOT20" ]];then ln -s "$LEGACY/datasets/MOT20" datasets/MOT20;fi
bash scripts/audit_upstream.sh
bash scripts/check_env.sh
