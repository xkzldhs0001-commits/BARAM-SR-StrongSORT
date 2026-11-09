#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
if [[ -e .venv/bin/activate ]];then source .venv/bin/activate;fi
source scripts/env.sh
python - <<'PY'
from pathlib import Path
import importlib.util
import torch
root=Path.cwd()
print('PyTorch',torch.__version__,'CUDA?',torch.cuda.is_available())
if torch.cuda.is_available():print('GPU:',torch.cuda.get_device_name(0))
for module in ['ultralytics','fastreid','torchreid','scipy','cv2']:
 print(module, 'OK' if importlib.util.find_spec(module) else 'MISSING')
for s in ['yolo26n.pt','osnet_x1_0.pth.tar','duke_bot_S50.pth','AFLink_epoch20.pth']:
 p=root/'weights'/s
 print('weights/'+s, ('READY' if p.is_file() and p.stat().st_size>0 else 'MISSING'))
for s in ['MOT17','MOT20']:
 d=root/'datasets'/s/'train'
 print(s, 'READY' if d.is_dir() else 'MISSING')
PY
