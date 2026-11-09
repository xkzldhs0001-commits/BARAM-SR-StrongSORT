"""Optional BoT ResNeSt50 person-ReID via FastReID.

Requires a *trained person-ReID* BoT/ResNeSt50 checkpoint AND matching FastReID
YAML config. This adapter is not a random torchvision ResNeSt classifier.
Preprocessing follows FastReID DefaultPredictor's batched tensor interface.
"""
from pathlib import Path
import numpy as np

class BoTResNeSt50Extractor:
    def __init__(self, config, weights, device='cpu'):
        for name, p in [('config',config),('BoT checkpoint',weights)]:
            if not Path(p).is_file():
                raise FileNotFoundError(f'{name} not found: {p}\n'
                    'Need compatible FastReID BoT/ResNeSt50 person-ReID config & weights. '
                    'Do NOT pass OSNet weights as the baseline.')
        try:
            from fastreid.config import get_cfg
            from fastreid.engine import DefaultPredictor
        except ImportError as exc:
            raise RuntimeError('FastReID is not installed. See README_BOT.md for official setup. '
                               'Use --mode selective for OSNet-only experiments.') from exc
        import torch
        self.torch = torch
        cfg=get_cfg()
        cfg.merge_from_file(str(config))
        cfg.MODEL.WEIGHTS=str(weights)
        cfg.MODEL.DEVICE='cuda:0' if device.startswith('cuda') else 'cpu'
        cfg.freeze()
        name=str(cfg.MODEL.BACKBONE.NAME).lower()
        if 'resnest' not in name:
            raise ValueError('Expected BoT ResNeSt50 config (build_resnest_backbone), '
                             'but got: ' + str(cfg.MODEL.BACKBONE.NAME))
        self.architecture=str(cfg.MODEL.BACKBONE.NAME)
        self.predictor=DefaultPredictor(cfg)
        self.size=tuple(int(x) for x in cfg.INPUT.SIZE_TEST)  # (height,width)

    def __call__(self,crops_bgr):
        import cv2
        if not crops_bgr: return np.empty((0,0),dtype=np.float32)
        h,w=self.size
        batch=np.stack([cv2.cvtColor(cv2.resize(c,(w,h)),cv2.COLOR_BGR2RGB)
                        .transpose(2,0,1) for c in crops_bgr]).copy()
        tensor=self.torch.from_numpy(batch).float()
        output=self.predictor(tensor)
        if not isinstance(output,self.torch.Tensor):
            raise RuntimeError('Unexpected FastReID output; check BoT checkpoint/config compatibility.')
        output=output.detach().cpu().numpy().astype(np.float32)
        if output.ndim != 2 or output.shape[0] != len(crops_bgr):
            raise RuntimeError('BoT embeddings must have shape (B,D)')
        return output
