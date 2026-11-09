#!/usr/bin/env python3
"""Generate one *computed* ECC JSON for all trackers on MOT Train; not author's precomputed JSON.

Provides comparable camera compensation without modifying the official code.
If exact-paper reproduction is required, replace with authors' supplied ECC JSON.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
from baram.data import dataset_sequences,list_frames


def make_one(seqdir,resize_width=640):
    frames=list_frames(seqdir)
    if not frames:raise RuntimeError('No input images '+str(seqdir))
    matrices={}
    prev=None
    for n,(frame,path) in enumerate(frames,1):
        img=cv2.imread(str(path),cv2.IMREAD_GRAYSCALE)
        if img is None:raise FileNotFoundError(path)
        scale=resize_width/img.shape[1]
        hh=max(100,int(round(img.shape[0]*scale)))
        small=cv2.resize(img,(resize_width,hh),interpolation=cv2.INTER_AREA).astype(np.float32)/255.
        if prev is None:
            warp=np.eye(3,dtype=float)
        else:
            identity=np.eye(3,dtype=np.float32)
            try:
                # Current template vs previous input to estimate camera motion.
                cc,estimate=cv2.findTransformECC(small,prev,identity,cv2.MOTION_HOMOGRAPHY,
                    (cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,40,1e-5))
                # Map output back to original image coordinate system.
                s=np.diag([scale,scale,1.]);warp=np.linalg.inv(s)@estimate@s
                if not np.all(np.isfinite(warp)) or np.linalg.norm(warp-np.eye(3))>50:
                    warp=np.eye(3,dtype=float)
            except cv2.error:
                warp=np.eye(3,dtype=float)
        matrices[str(frame)]=warp.tolist()
        prev=small
        if n%250==0 or n==len(frames):print(f'[ECC] {seqdir.name}: {n}/{len(frames)}',flush=True)
    return matrices


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--project',type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument('--dataset',choices=['MOT17','MOT20','both'],default='both')
    p.add_argument('--output',type=Path,default=None)
    a=p.parse_args();root=a.project.resolve();output=a.output or root/'weights';output.mkdir(exist_ok=True,parents=True)
    for ds in (['MOT17','MOT20'] if a.dataset=='both' else [a.dataset]):
        target=output/f'{ds}_ECC_val.json'
        if target.exists():
            print('[SKIP] Existing ECC file (possibly official) will not be overwritten:',target)
            continue
        data={}
        for seq in dataset_sequences(root/'datasets',ds):
            data[seq]=make_one(root/'datasets'/ds/'train'/seq)
        target.write_text(json.dumps(data))
        print('[DONE] Computed ECC:',target,'NOT the authors supplied ECC, compare equally across models.')

if __name__=='__main__':main()
