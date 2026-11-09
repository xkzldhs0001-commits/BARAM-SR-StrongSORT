"""Raw COCO YOLO26n detections cached once per sequence for fair input parity."""
from __future__ import annotations
from pathlib import Path
import csv
import hashlib
import json
import time
import numpy as np
import cv2


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def dataset_sequences(root,dataset):
    p=root/dataset/'train'
    if not p.is_dir():raise FileNotFoundError(f'MOT train sequences missing: {p}')
    names=sorted(x.name for x in p.iterdir() if x.is_dir() and (x/'img1').exists())
    if dataset=='MOT17':
        # Avoid duplicate MOT17 image sets for FRCNN/DPM/SDP variants.
        names=[s for s in names if s.endswith('-FRCNN')]
    return names


def list_frames(seq_dir):
    images=seq_dir/'img1'
    if not images.is_dir():raise FileNotFoundError(images)
    result=[]
    for p in images.iterdir():
        if p.suffix.lower() in ('.jpg','.png') and p.stem.isdecimal():result.append((int(p.stem),p))
    result.sort()
    return result


def cache_yolo(seq_dir,out,weights,device='cuda:0',confidence=.25,imgsz=640,max_frames=0,max_det=1000):
    import torch
    from ultralytics import YOLO
    if not weights.is_file():raise FileNotFoundError(weights)
    frames=list_frames(seq_dir)
    if max_frames:frames=frames[:max_frames]
    if not frames:raise RuntimeError('Empty image sequence')
    out.mkdir(parents=True,exist_ok=True)
    cache=out/'detections.npy'
    meta=out/'detections_meta.json'
    fingerprint={'source':str(seq_dir.resolve()),'frames':len(frames),'last_frame':frames[-1][0],
                 'yolo_sha256':sha256(weights),'confidence':confidence,'imgsz':imgsz,'device':device,
                 'classes':[0],'max_det':int(max_det)}
    if cache.is_file() and meta.is_file():
        old=json.loads(meta.read_text())
        if all(old.get(k)==v for k,v in fingerprint.items()):
            print('[CACHE] Identical YOLO detections:',cache,flush=True)
            return np.load(cache),old
    model=YOLO(str(weights))
    if (model.names.get(0) if isinstance(model.names,dict) else model.names[0])!='person':
        raise RuntimeError('Class 0 is not person; wrong detection weights')
    rows=[];elapsed=0.
    for index,(fr,file) in enumerate(frames,1):
        im=cv2.imread(str(file))
        if im is None:raise RuntimeError('Cannot decode '+str(file))
        if device.startswith('cuda'):torch.cuda.synchronize()
        st=time.perf_counter()
        pred=model.predict(im,classes=[0],conf=confidence,imgsz=imgsz,device=device,
                           verbose=False,max_det=max_det)[0]
        if device.startswith('cuda'):torch.cuda.synchronize()
        elapsed+=time.perf_counter()-st
        if pred.boxes is not None:
            xyxy=pred.boxes.xyxy.detach().cpu().numpy()
            probs=pred.boxes.conf.detach().cpu().numpy()
            for b,prob in zip(xyxy,probs):
                x1,y1,x2,y2=map(float,b)
                if x2-x1<2 or y2-y1<2:continue
                rows.append([fr,-1,x1,y1,x2-x1,y2-y1,float(prob),-1,-1,-1])
        if index%100==0 or index==len(frames):
            print(f'[YOLO] {seq_dir.name} {index}/{len(frames)}',flush=True)
    data=np.asarray(rows,dtype=np.float64).reshape(-1,10)
    np.save(cache,data)
    fingerprint['detector_seconds']=elapsed
    fingerprint['detections']=len(data)
    meta.write_text(json.dumps(fingerprint,indent=2))
    return data,fingerprint


def per_frame_detection(detections, frame,source):
    # Retain exact confidence and bounding boxes for all modes.
    return [source['Detection'](r[2:6].copy(),float(r[6]),None)
            for r in detections if int(r[0])==frame]
