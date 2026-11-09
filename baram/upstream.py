"""Load unedited original StrongSORT classes; original flags stay enabled."""
from __future__ import annotations
from pathlib import Path
import sys
from .compat import enable


def load(root:Path, dataset='MOT17', ecc_data=None):
    enable()
    official=root/'third_party'/'StrongSORT'
    if not (official/'deep_sort'/'track.py').exists():
        raise RuntimeError('Official source missing: run scripts/setup_ubuntu.sh first')
    if str(official) not in sys.path:sys.path.insert(0,str(official))
    # The original opts.py parses CLI at module import time. Supply correct
    # user-facing StrongSORT options without patching its source tree.
    backup=sys.argv[:]
    # Use MOT17/val for opt bootstrap because original opts.py has NO MOT20/val key.
    # We select MOT20 sequences ourselves later. Do not pass --ECC here because
    # original opts.py immediately opens a hard-coded author-local JSON path.
    sys.argv=['strong_sort.py','MOT17','val','--BoT','--NSA','--EMA','--MC','--woC']
    try:
        from opts import opt
        from deep_sort.detection import Detection
        from deep_sort.nn_matching import NearestNeighborDistanceMetric
        from deep_sort.tracker import Tracker
        from deep_sort.track import Track,TrackState
        from deep_sort.kalman_filter import KalmanFilter
    finally:sys.argv=backup
    opt.dataset=dataset
    opt.BoT=True;opt.NSA=True;opt.EMA=True;opt.MC=True;opt.woC=True
    opt.EMA_alpha=.9;opt.MC_lambda=.98
    opt.max_cosine_distance=.45;opt.nn_budget=1
    opt.ECC=ecc_data is not None
    if ecc_data is not None:opt.ecc=ecc_data
    return {'opt':opt,'Detection':Detection,'Metric':NearestNeighborDistanceMetric,
            'Tracker':Tracker,'Track':Track,'TrackState':TrackState,'KalmanFilter':KalmanFilter}
