"""YOLO26n detections and real Torchreid OSNet features."""
from __future__ import annotations
from pathlib import Path
import numpy as np
from .tracker import Detection

class YOLO26PersonDetector:
    def __init__(self,weights='yolo26n.pt',conf=0.3,device='auto',imgsz=640):
        from ultralytics import YOLO
        self.model=YOLO(weights)
        self.conf=conf
        self.device=device
        self.imgsz=imgsz

    def __call__(self,frame):
        kw=dict(conf=self.conf,classes=[0],verbose=False,imgsz=self.imgsz)
        if self.device!='auto':
            kw['device']=self.device
        r=self.model.predict(frame,**kw)[0]
        if r.boxes is None or len(r.boxes)==0:
            return []
        b=r.boxes.xyxy.detach().cpu().numpy()
        c=r.boxes.conf.detach().cpu().numpy()
        return [Detection(box=np.array(box,dtype=float),confidence=float(conf))
                for box,conf in zip(b,c)]

class OSNetExtractor:
    def __init__(self,weights,device='auto'):
        """Requires RE-ID trained weights; never silently run with random weights."""
        if not Path(weights).is_file():
            raise FileNotFoundError(f'OSNet weights missing: {weights}. See README model installation.')
        import torch
        try:
            from torchreid.utils import FeatureExtractor
        except ImportError as e:
            raise RuntimeError('torchreid installation required; see scripts/install_ubuntu.sh') from e
        actual = ('cuda' if torch.cuda.is_available() else 'cpu') if device=='auto' else device
        self.extractor=FeatureExtractor(model_name='osnet_x1_0',model_path=weights,device=actual)

    def __call__(self,crops_bgr):
        import cv2
        # FeatureExtractor's ndarray input is RGB according to Torchreid preprocessing.
        rgb=[cv2.cvtColor(c,cv2.COLOR_BGR2RGB) for c in crops_bgr]
        f=self.extractor(rgb)
        return f.detach().cpu().numpy().astype(np.float32)
