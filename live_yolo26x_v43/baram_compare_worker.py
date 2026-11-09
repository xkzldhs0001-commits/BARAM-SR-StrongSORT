#!/usr/bin/env python3
"""BARAM live YOLO26x + optimized SR-StrongSORT, unlimited camera runtime.

This is a NEW detector condition; YOLOX-X benchmark metrics do not carry over.
Only COCO person detections are passed to the unchanged optimized tracker.
"""
from __future__ import annotations
import argparse
import configparser
from collections import deque
import csv
from datetime import datetime
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np

MODES=('sr_osnet_selective','sr_osnet_full','strongsort_osnet','strongsort_bot')
EXPERIMENT='BARAM_LIVE_YOLO26X_OCCLUSION_V4_3_LIGHT_ID'


def parse():
    p=argparse.ArgumentParser(description='BARAM YOLO26x + optimized StrongSORT live demo')
    p.add_argument('--root',type=Path,default=Path.home()/'Downloads'/'BARAM_SR_StrongSORT')
    p.add_argument('--mode',choices=MODES,default='sr_osnet_full')
    p.add_argument('--camera',type=int,default=0)
    p.add_argument('--input',type=Path,help='Optional recorded video; fair repeat tests use identical file')
    p.add_argument('--mot17-seq','--mot-seq',dest='mot17_seq',type=Path,help='MOT17 or MOT20 sequence folder containing img1/ and seqinfo.ini')
    p.add_argument('--detector-floor',type=float,default=None,help='Common detector score floor (e.g. 0.15) for BOTH trackers; normal detections still use --conf')
    p.add_argument('--no-render',action='store_true',help='Skip preview drawing for fair algorithm FPS')
    p.add_argument('--mot-output',action='store_true',help='Write confirmed measured DISPLAY IDs to MOT17 result txt')
    p.add_argument('--width',type=int,default=1280)
    p.add_argument('--height',type=int,default=720)
    p.add_argument('--capture-fps',type=float,default=30.0)
    p.add_argument('--conf',type=float,default=.40)
    p.add_argument('--imgsz',type=int,default=640)
    p.add_argument('--occlusion-low-conf',type=float,default=.15,
                   help='Lower person score accepted ONLY to reconnect existing tracks (default 0.15)')
    p.add_argument('--no-occlusion-rescue',action='store_true',
                   help='Disable low-confidence rescue; for comparison with original V4')
    p.add_argument('--occlusion-max-missed',type=int,default=12,
                   help='Maximum frames since observation to allow low-confidence rescue')
    p.add_argument('--predicted-hold',type=int,default=10,
                   help='Draw dashed, explicitly PREDICTED boxes for N missing frames; 0 disables')
    p.add_argument('--iou',type=float,default=.7,help='Ultralytics NMS IoU (may not apply to NMS-free YOLO26)')
    p.add_argument('--weights',type=Path,default=None,help='YOLO26x weights path (default: project/yolo26x.pt)')
    p.add_argument('--device',default='cuda:0')
    ecc_group=p.add_mutually_exclusive_group()
    ecc_group.add_argument('--disable-ecc',dest='disable_ecc',action='store_true',default=True,
                           help='Default: disable motion-camera compensation for fixed webcams')
    ecc_group.add_argument('--enable-ecc',dest='disable_ecc',action='store_false',
                           help='Enable motion-camera compensation for moving cameras')
    p.add_argument('--tracker-max-age',type=int,default=75,help='Live lost-track persistence in processed frames; MOT benchmark unaffected')
    p.add_argument('--tracker-n-init',type=int,default=2,help='Consecutive matches to confirm a live SR track')
    p.add_argument('--motion-gate',type=float,default=14.5,help='Live-only Kalman Mahalanobis gating threshold (research baseline 9.4877)')
    p.add_argument('--no-live-relink',action='store_true',help='Disable live-only second-pass appearance matching')
    p.add_argument('--live-relink-sim',type=float,default=.88,help='Fallback similarity for close-by unmatched tracks')
    p.add_argument('--id-memory-seconds',type=float,default=180.,help='Remember previously visible IDs for N wall-clock seconds')
    p.add_argument('--id-threshold',type=float,default=.72,help='Appearance threshold if only one identity is enrolled')
    p.add_argument('--id-multi-threshold',type=float,default=.80,help='Appearance threshold when multiple identities are enrolled')
    p.add_argument('--id-max-gap',type=float,default=1.5,help='Short-gap geometry rescue window in seconds')
    p.add_argument('--id-max-distance',type=float,default=1.4,help='Normalized center distance for short-gap rescue')
    p.add_argument('--id-margin',type=float,default=.09,help='Minimum appearance advantage over rivals')
    p.add_argument('--id-confirm-frames',type=int,default=3,help='Consecutive appearance matches needed to restore a dormant ID')
    p.add_argument('--id-swap-confirm-frames',type=int,default=3,help='Consecutive contradictory clean matches before detaching a bound ID')
    p.add_argument('--id-max-verify-frames',type=int,default=9,help='Minimum VERIFY observation frames before considering a demonstrably new person; ambiguous faces remain VERIFY')
    p.add_argument('--id-shape-ratio',type=float,default=4.,help='Max height/width ratio difference for re-entry matching')
    p.add_argument('--no-id-memory',action='store_true',help='Disable live display-ID recovery (for A/B comparison)')
    p.add_argument('--appearance-refresh',type=int,default=12,help='Selective SR: refresh clean tracked appearance every N frames; 0 disables')
    p.add_argument('--show-internal-id',action='store_true',help='Display raw tracker ID alongside persistent display ID')
    p.add_argument('--max-frames',type=int,default=0,help='0 (default): run continuously until Q or Ctrl+C')
    p.add_argument('--read-retries',type=int,default=5,help='Temporary live camera read-failure retries')
    p.add_argument('--no-display',action='store_true')
    p.add_argument('--mirror',action='store_true')
    p.add_argument('--rotate',type=int,choices=(0,180),default=0)
    p.add_argument('--record-annotated',action='store_true',help='Optional demo recording; adds video encoding overhead')
    p.add_argument('--record-raw',action='store_true',help='Save unannotated source for later GT labeling')
    p.add_argument('--check',action='store_true')
    p.add_argument('--out',type=Path)
    args=p.parse_args()
    if args.max_frames<0:p.error('--max-frames must be >= 0')
    if args.input and args.mot17_seq:p.error('Select either --input or --mot17-seq, not both')
    if args.mot_output and not args.mot17_seq:p.error('--mot-output requires --mot17-seq')
    if args.camera<0:p.error('--camera must be >= 0')
    if args.read_retries<0:p.error('--read-retries must be >= 0')
    if not 0 < args.conf < 1:p.error('--conf must be in (0,1)')
    if args.tracker_max_age<1:p.error('--tracker-max-age must be >=1')
    if not 0<args.occlusion_low_conf<args.conf:p.error('--occlusion-low-conf must be between 0 and --conf')
    if args.occlusion_max_missed<1:p.error('--occlusion-max-missed must be >=1')
    if args.predicted_hold<0:p.error('--predicted-hold must be >=0')
    if args.tracker_n_init<1:p.error('--tracker-n-init must be >=1')
    if args.motion_gate<9.4877:p.error('--motion-gate must be >=9.4877')
    if not 0<args.live_relink_sim<1:p.error('--live-relink-sim must be in (0,1)')
    if not 0<args.id_multi_threshold<1:p.error('--id-multi-threshold must be in (0,1)')
    if args.id_max_gap<0 or args.id_max_distance<=0:p.error('invalid ID geometry gap/distance')
    if args.id_memory_seconds<=0:p.error('--id-memory-seconds must be >0')
    if not 0<args.id_threshold<1:p.error('--id-threshold must be in (0,1)')
    if args.id_margin<0:p.error('--id-margin must be >=0')
    if args.id_confirm_frames<2 or args.id_swap_confirm_frames<2:p.error('ID confirmation frames must be >=2')
    if args.id_max_verify_frames<args.id_confirm_frames:p.error('ID max verify frames must be >= confirm frames')
    if args.id_shape_ratio<1:p.error('--id-shape-ratio must be >=1')
    if args.appearance_refresh<0:p.error('--appearance-refresh must be >=0')
    if args.detector_floor is not None and not (0<args.detector_floor<=args.conf):p.error('--detector-floor must be in (0, --conf]')
    if args.no_render and args.record_annotated:p.error('--no-render is incompatible with --record-annotated')
    return args


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for data in iter(lambda:f.read(1<<20),b''):h.update(data)
    return h.hexdigest()


def sync(device):
    if str(device).startswith('cuda'):
        import torch
        torch.cuda.synchronize()


def file_check(root,mode,weights=None):
    """Check existing tracker files without downloading anything."""
    files={
      'Optimized StrongSORT bridge':root/'official_compare/strongsort_adapter.py',
      'Official StrongSORT':root/'third_party/StrongSORT_official/deep_sort/tracker.py',
      'Live ID memory module':Path(__file__).resolve().parent/'identity_memory.py',
      'Live appearance reattach module':Path(__file__).resolve().parent/'live_relink.py',
      'Occlusion weak-box rescue module':Path(__file__).resolve().parent/'occlusion_bridge.py',
    }
    if mode!='strongsort_bot':
        files['OSNet weights']=root/'weights/osnet_x1_0.pth.tar'
        files['OSNet extractor']=root/'srstrongsort/models.py'
    else:
        files['BoT weights']=root/'weights/duke_bot_S50.pth'
        files['BoT config']=root/'third_party/fast-reid/configs/DukeMTMC/bagtricks_S50.yml'
        files['BoT extractor']=root/'srstrongsort/bot.py'
    missing=[]
    for label,path in files.items():
        ok=path.is_file()
        status='PASS' if ok else 'FAIL'
        print(f'[{status}] {label}: {path}',flush=True)
        if not ok:missing.append(str(path))
    bridge=files['Optimized StrongSORT bridge']
    if bridge.is_file():
        optimized='BARAM_VECTOR_GATE_OPT_V1' in bridge.read_text(errors='replace')
        status='PASS' if optimized else 'FAIL'
        print(f'[{status}] optimized SR bridge installed: {optimized}',flush=True)
        if not optimized:missing.append('SR bridge patch missing: apply BARAM_SR_ASSOC_OPT_V1 first')
    try:
        from importlib.metadata import version
        v=version('ultralytics')
        print('[PASS] ultralytics package version:',v,flush=True)
        try:
            from packaging.version import Version
            if Version(v)<Version('8.4.0'):
                missing.append('ultralytics is too old for YOLO26x; run: bash run.sh install')
                print('[FAIL] ultralytics >=8.4.0 is required for YOLO26x',flush=True)
        except ImportError:
            print('[WARN] Could not check Ultralytics minimum version; packaging not installed',flush=True)
    except Exception:
        missing.append('ultralytics package missing; run: bash run.sh install')
        print('[FAIL] ultralytics missing (run: bash run.sh install)',flush=True)
    weights_path=resolve_weights(root,weights)
    if weights_path.is_file():
        print('[PASS] YOLO26x checkpoint:',weights_path,flush=True)
    elif weights is not None:
        missing.append('Specified weights file missing: '+str(weights_path))
        print('[FAIL] specified model checkpoint:',weights_path,flush=True)
    else:
        print('[INFO] YOLO26x checkpoint not present. Ultralytics will download yolo26x.pt on first run (internet needed).',flush=True)
        print('[INFO] Expected path:',weights_path,flush=True)
    if missing:raise RuntimeError('Preflight failed: '+'; '.join(missing))
    print('[PASS] YOLO26x + SR tracking prerequisites ready',flush=True)
    return files,weights_path


def resolve_weights(root,weights):
    if weights is not None:
        p=weights.expanduser()
        return (root/p).resolve() if not p.is_absolute() else p.resolve()
    for p in (root/'yolo26x.pt',root/'weights/yolo26x.pt'):
        if p.is_file():return p.resolve()
    return (root/'yolo26x.pt').resolve()


class YOLO26XDetector:
    """Ultralytics COCO person (class 0) -> list of tlwh boxes and scores."""
    def __init__(self,weights_path,root,device):
        from ultralytics import YOLO
        self.device=device
        weights_path=Path(weights_path)
        if weights_path.is_file():
            print('[DETECTOR] Loading existing YOLO26x:',weights_path,flush=True)
            self.model=YOLO(str(weights_path))
        else:
            # YOLO('yolo26x.pt') downloads to cwd if not installed.
            print('[DETECTOR] Downloading official yolo26x.pt on first use.',flush=True)
            old_cwd=Path.cwd()
            try:
                os.chdir(root)
                self.model=YOLO('yolo26x.pt')
            finally:
                os.chdir(old_cwd)
            if not weights_path.is_file():
                downloaded=Path(str(getattr(self.model,'ckpt_path',''))).expanduser()
                if downloaded.is_file():weights_path=downloaded.resolve()
                else:raise FileNotFoundError(f'Model downloaded but not found at {weights_path}. Find yolo26x.pt and pass --weights /absolute/path.')
        self.weights_path=weights_path
    def detect(self,frame,conf,imgsz,iou):
        res=self.model.predict(source=frame,conf=conf,imgsz=imgsz,
            iou=iou,classes=[0],device=self.device,verbose=False,max_det=300)[0]
        if res.boxes is None or len(res.boxes)==0:return [],[]
        xyxy=res.boxes.xyxy.detach().cpu().numpy()
        scores=res.boxes.conf.detach().cpu().numpy()
        boxes=[]; confs=[]
        for corners,score in zip(xyxy,scores):
            x1,y1,x2,y2=map(float,corners)
            if x2<=x1 or y2<=y1:continue
            boxes.append(np.array([x1,y1,x2-x1,y2-y1],dtype=np.float64))
            confs.append(float(score))
        return boxes,confs


def get_crop(frame,box):
    """Identical cropping convention to the previous YOLOX-X experiments."""
    h,w=frame.shape[:2]
    x,y,bw,bh=map(float,box)
    x1=max(0,min(w-1,int(x)));y1=max(0,min(h-1,int(y)))
    x2=max(x1+1,min(w,int(np.ceil(x+bw))));y2=max(y1+1,min(h,int(np.ceil(y+bh))))
    return frame[y1:y2,x1:x2].copy()


def label_box(im,box,text,color,thin=2):
    import cv2
    h,w=im.shape[:2]
    x,y,bw,bh=[float(t) for t in box]
    x1=max(0,min(w-1,int(x)));y1=max(0,min(h-1,int(y)))
    x2=max(0,min(w-1,int(x+bw)));y2=max(0,min(h-1,int(y+bh)))
    if x2<=x1 or y2<=y1:return
    cv2.rectangle(im,(x1,y1),(x2,y2),color,thin)
    if text:
        (tw,th),bl=cv2.getTextSize(text,cv2.FONT_HERSHEY_SIMPLEX,.55,2)
        by=max(th+bl+2,y1)
        cv2.rectangle(im,(x1,max(0,by-th-bl-5)),(min(w-1,x1+tw+8),min(h-1,by+2)),color,-1)
        cv2.putText(im,text,(x1+4,by-bl),cv2.FONT_HERSHEY_SIMPLEX,.55,(0,0,0),2,cv2.LINE_AA)


def draw(im,mode,trackers,boxes,chosen,fps,frames,total_det,total_reid,input_fps,elapsed,display_ids=None,id_restores=0,show_internal=False,predicted_hold=0,low_rescued=0,provisional_ids=None):
    import cv2
    # Overlay detection boxes first; selected Re-ID boxes in warm yellow.
    selected=set(chosen)
    for k,b in enumerate(boxes):
        label_box(im,b,'',(0,185,255) if k in selected else (95,95,95),1)
    visible=0
    for t in trackers.tracks:
        if not t.is_confirmed() or t.is_deleted():continue
        age=int(t.time_since_update)
        if age>predicted_hold and age!=0:continue
        track_id=int(t.track_id)
        shown_id=display_ids.get(track_id,'VERIFY') if display_ids is not None else track_id
        if isinstance(shown_id, (int, np.integer)):
            color=(int((53*shown_id+75)%190+65),int((91*shown_id+9)%190+65),int((37*shown_id+40)%190+65))
            title=f'ID {shown_id}' if age==0 else f'PRED ID {shown_id}'
        else:
            color=(150,150,150)
            title='VERIFY' if age==0 else 'PRED VERIFY'
        if provisional_ids is not None and track_id in provisional_ids:
            title += ' [OCC?]'  # Provisional during overlap or suspected switch
        if show_internal:title+=f' (track {track_id})'
        if age==0:
            label_box(im,t.to_tlwh(),title,color,2)
            visible+=1
        else:
            # Inference/prediction is NOT an observed detection. Dashed orange.
            bx=t.to_tlwh(); x,y,w0,h0=map(int,bx)
            x=max(0,min(im.shape[1]-1,x));y=max(0,min(im.shape[0]-1,y))
            x2=max(x,min(im.shape[1]-1,x+w0));y2=max(y,min(im.shape[0]-1,y+h0))
            if x2>x and y2>y:
                for a in range(x,x2,16):
                    cv2.line(im,(a,y),(min(a+8,x2),y),(0,180,250),2)
                    cv2.line(im,(a,y2),(min(a+8,x2),y2),(0,180,250),2)
                for a in range(y,y2,16):
                    cv2.line(im,(x,a),(x,min(a+8,y2)),(0,180,250),2)
                    cv2.line(im,(x2,a),(x2,min(a+8,y2)),(0,180,250),2)
                cv2.putText(im,title,(x,max(13,y-5)),cv2.FONT_HERSHEY_SIMPLEX,.45,(0,180,250),1,cv2.LINE_AA)
    h,w=im.shape[:2]
    cv2.rectangle(im,(0,0),(w,126),(18,22,28),-1)
    f=cv2.FONT_HERSHEY_SIMPLEX
    messages=[
      f'BARAM  |  YOLO26x + {mode}  |  LIVE VIDEO',
      f'Processing FPS: {fps:.1f}   |   Camera setting: {input_fps:.0f} fps   |   Frame: {frames}  ({elapsed:.0f}s)',
      f'Person detections: {len(boxes)}   |   Active IDs: {visible}   |   low-score rescues: {low_rescued}   |   Re-ID: {len(chosen)}/{len(boxes)}',
      f'Re-ID total: {total_reid}/{total_det} ({100*total_reid/max(1,total_det):.1f}%)  |  ID restores: {id_restores}  |  Q quit  S snapshot',
    ]
    for i,msg in enumerate(messages):
        cv2.putText(im,msg,(15,27+30*i),f,.62 if i==0 else .54,(220,238,244),2 if i==0 else 1,cv2.LINE_AA)
    return visible


def observed_tracks_with_fresh_features(tracker, boxes, feature_map, identity_bank=None, frame_size=None):
    """Use *current-frame* high-confidence crops, not cached track features.

    Discard geometrically ambiguous feature associations when boxes overlap.
    This avoids learning another person's appearance during a crossing.
    OSNet inference already happened upstream; this function adds none.
    """
    from scipy.optimize import linear_sum_assignment
    from official_compare.strongsort_adapter import iou_xywh
    observed=[t for t in tracker.tracks
              if t.is_confirmed() and not t.is_deleted() and t.time_since_update==0]
    good_indices=sorted(j for j in feature_map if j < len(boxes))
    assigned={}
    if observed and good_indices:
        overlap=np.asarray([[iou_xywh(t.to_tlwh(),boxes[j]) for j in good_indices]
                            for t in observed],dtype=float)
        rows,cols=linear_sum_assignment(-overlap)
        for i,k in zip(rows,cols):
            j=good_indices[k]
            best=float(overlap[i,k])
            row_alt=max((float(overlap[i,l]) for l in range(len(good_indices)) if l!=k),default=0.)
            col_alt=max((float(overlap[l,k]) for l in range(len(observed)) if l!=i),default=0.)
            if best<.5 or best-row_alt<.10 or best-col_alt<.10:
                continue
            # Seeing two bodies overlap often produces a mixed appearance crop.
            # Such embeddings may be inspected only after separation: never
            # enroll them or use them to forcibly change a bound ID.
            crowded=any(iou_xywh(boxes[j],bb)>.25
                        for q,bb in enumerate(boxes) if q!=j)
            x,y,w,h=map(float,boxes[j])
            edge=False
            if frame_size is not None:
                fw,fh=frame_size
                edge=(x<4 or y<4 or x+w>fw-4 or y+h>fh-4)
            clean=not crowded and not edge and w>=20 and h>=50
            assigned[i]=(feature_map[j],clean,crowded)
    output=[]
    # Track-to-track overlap is dangerous even when a unique detector match
    # was found: the tracker can have exchanged its internal IDs mid-crossing.
    track_ambiguity=[any(i!=j and iou_xywh(t.to_tlwh(),other.to_tlwh())>.18
                         for j,other in enumerate(observed)) for i,t in enumerate(observed)]
    for i,t in enumerate(observed):
        iid=int(t.track_id)
        is_low=iid in getattr(tracker,'last_low_update_ids',set())
        prev_id=identity_bank.bindings.get(iid) if identity_bank is not None else None
        prev_rec=identity_bank.records.get(prev_id) if identity_bank is not None else None
        # Weak-score rescues must not contaminate long-term appearance/geometry.
        if is_low and prev_rec is not None:
            output.append({'internal_id':iid,'bbox':prev_rec.bbox,
                           'feature':None,'learnable':False,
                           'occluded':bool(track_ambiguity[i] or is_low)})
            continue
        feature,clean,det_crowded=assigned.get(i,(None,False,False))
        occluded=bool(track_ambiguity[i] or det_crowded or is_low)
        output.append({'internal_id':iid,'bbox':t.to_tlwh(),
                       'feature':None if is_low else feature,
                       'learnable':bool(clean and not is_low and not occluded),
                       'occluded':occluded})
    return output



class MOTImageCapture:
    """Lightweight lossless JPEG-frame reader; no transcoding or frame loss."""
    def __init__(self, sequence):
        import cv2
        self.sequence=Path(sequence).expanduser().resolve()
        ini=self.sequence/'seqinfo.ini'
        folder=self.sequence/'img1'
        if not ini.is_file() or not folder.is_dir():
            raise FileNotFoundError(f'Expected {ini} and {folder}')
        cfg=configparser.ConfigParser()
        cfg.read(ini,encoding='utf-8-sig')
        sec=cfg['Sequence']
        self.fps=float(sec.get('frameRate',30))
        self.width=int(sec.get('imWidth',0))
        self.height=int(sec.get('imHeight',0))
        self.expected=int(sec.get('seqLength',0))
        extension=sec.get('imExt','.jpg')
        self.images=sorted(folder.glob('*'+extension))
        if not self.images:raise FileNotFoundError(f'No {extension} frames in {folder}')
        if self.expected and len(self.images)!=self.expected:
            raise RuntimeError(f'MOT frame count mismatch: seqinfo={self.expected}, img1={len(self.images)}')
        numbers=[]
        for p in self.images:
            if not p.stem.isdigit():raise ValueError(f'Non-numeric frame filename: {p.name}')
            numbers.append(int(p.stem))
        if numbers!=list(range(1,len(numbers)+1)):
            raise RuntimeError('MOT frames must be consecutive and start at 000001')
        self.pos=0
    def isOpened(self):return self.pos<len(self.images)
    def read(self):
        import cv2
        if self.pos>=len(self.images):return False,None
        path=self.images[self.pos]
        frame=cv2.imread(str(path),cv2.IMREAD_COLOR)
        if frame is None:raise RuntimeError(f'Cannot read MOT frame: {path}')
        self.pos+=1
        return True,frame
    def get(self,prop):
        import cv2
        if prop==cv2.CAP_PROP_FPS:return self.fps
        if prop==cv2.CAP_PROP_FRAME_WIDTH:return self.width
        if prop==cv2.CAP_PROP_FRAME_HEIGHT:return self.height
        return 0.0
    def release(self):pass


def save_mot_tracks(writer, tracker, identity_bank, frame_i, frame_size):
    """Only measured confirmed boxes with settled numeric display identities. VERIFY is omitted (counts toward FN in TrackEval)."""
    width,height=frame_size
    seen=set()
    for t in tracker.tracks:
        if not t.is_confirmed() or t.is_deleted() or t.time_since_update!=0:
            continue
        iid=int(t.track_id)
        did=identity_bank.bindings.get(iid) if identity_bank is not None else iid
        if not isinstance(did,(int,np.integer)) or int(did)<=0:continue
        did=int(did)
        if did in seen:
            raise RuntimeError(f'Duplicate DISPLAY ID {did} in frame {frame_i}')
        seen.add(did)
        x,y,w,h=map(float,t.to_tlwh())
        left=max(0.,min(float(width),x))
        top=max(0.,min(float(height),y))
        right=max(0.,min(float(width),x+w))
        bottom=max(0.,min(float(height),y+h))
        if right-left<1 or bottom-top<1:continue
        # MOTChallenge uses 1-based bounding-box top-left coordinates.
        writer.writerow([frame_i,did,round(left+1,3),round(top+1,3),
                         round(right-left,3),round(bottom-top,3),1,-1,-1,-1])

def main():
    args=parse()
    root=args.root.expanduser().resolve()
    req,weights_path=file_check(root,args.mode,args.weights)
    if args.check:return
    for p in (root/'third_party/fast-reid',root):
        if str(p) not in sys.path:sys.path.insert(0,str(p))

    import cv2
    import torch
    from official_compare.strongsort_adapter import (
        prepare_official,make_sr_classes,SRDetection,iou_xywh,make_ecc_matrix)
    from identity_memory import StableIdentityBank
    from occlusion_bridge import choose_low_conf_rescue, patched_live_sr_tracker

    # Keep the inference device string (e.g. 'cuda:0') for YOLO and OSNet,
    # but use an explicit integer GPU index for PyTorch memory statistics.
    # Some torch/CUDA combinations reject a string in the memory-stats API.
    cuda_stats_device = None
    cuda_stats_enabled = False
    if str(args.device).startswith('cuda'):
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is not available. Check nvidia-smi and installed PyTorch.')
        requested = torch.device(args.device)
        cuda_stats_device = requested.index
        if cuda_stats_device is None:
            cuda_stats_device = torch.cuda.current_device()
        if not 0 <= cuda_stats_device < torch.cuda.device_count():
            raise RuntimeError(f'GPU index {cuda_stats_device} does not exist; '
                               f'{torch.cuda.device_count()} CUDA GPU(s) available')
        try:
            # Include model initialization in the allocator peak measurements.
            torch.cuda.reset_peak_memory_stats(device=cuda_stats_device)
            cuda_stats_enabled = True
            print(f'[GPU] CUDA peak stats enabled for device index {cuda_stats_device}', flush=True)
        except (RuntimeError, ValueError, TypeError) as exc:
            # Never abort actual process VRAM sampling due to an optional
            # allocator metric. nvidia-smi runs in the parent benchmark process.
            print(f'[WARN] PyTorch allocator peak reset unavailable: {exc}; '
                  'continuing with nvidia-smi process VRAM sampling', flush=True)
    if args.mot17_seq:
        cap=MOTImageCapture(args.mot17_seq)
        src_label=cap.sequence.name
    elif args.input:
        source=str(args.input.expanduser().resolve())
        if not args.input.expanduser().is_file():raise FileNotFoundError(args.input)
        cap=cv2.VideoCapture(source)
        src_label='file'
    else:
        cap=cv2.VideoCapture(args.camera,cv2.CAP_V4L2)
        if not cap.isOpened():raise RuntimeError('Camera not opened. Check /dev/video0 and close other camera apps.')
        cap.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT,args.height)
        cap.set(cv2.CAP_PROP_FPS,args.capture_fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE,1) # best effort, V4L2 backend may ignore
        src_label=f'camera{args.camera}'
    if not cap.isOpened():raise RuntimeError('Cannot open input source')
    input_fps=float(cap.get(cv2.CAP_PROP_FPS) or args.capture_fps)
    got_w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    got_h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f'[CAMERA] input={src_label} {got_w}x{got_h}, source fps reported={input_fps:.2f} (NOT tracking FPS)',flush=True)
    print('[INFO] Start model initialization. No camera frames are tracked during warmup.',flush=True)
    api=prepare_official(root/'third_party/StrongSORT_official')
    # Only the live Python process receives this threshold; verified MOT code
    # and saved experiment outputs are NOT modified.
    import official_compare.strongsort_adapter as live_bridge
    live_bridge.CHI2_95_4=args.motion_gate
    print(f'[LIVE TRACKER] Mahalanobis gate={args.motion_gate:.2f}; benchmark gate=9.4877',flush=True)
    detector=YOLO26XDetector(weights_path,root,args.device)
    if args.mode=='strongsort_bot':
        from srstrongsort.bot import BoTResNeSt50Extractor
        extractor=BoTResNeSt50Extractor(req['BoT config'],req['BoT weights'],args.device)
    else:
        from srstrongsort.models import OSNetExtractor
        extractor=OSNetExtractor(req['OSNet weights'],args.device)
    if args.mode.startswith('strongsort_'):
        tracker=api.Tracker(api.Metric('cosine',.45,budget=1),max_age=30,n_init=3)
    else:
        SR=make_sr_classes(api,lambda_app=.65,match_threshold=.82,reid_similarity=.72)
        LiveSR=patched_live_sr_tracker(SR,min_similarity=args.live_relink_sim,
                                   enabled=not args.no_live_relink)
        tracker=LiveSR()
        # Live-only changes; existing MOT benchmark and source stay untouched.
        tracker.max_age=args.tracker_max_age
        tracker.n_init=args.tracker_n_init

    enable_id_memory=args.mode.startswith('sr_') and not args.no_id_memory
    identity_bank=(StableIdentityBank(ttl_seconds=args.id_memory_seconds,
       min_similarity=args.id_threshold,multi_similarity=args.id_multi_threshold,
       margin=args.id_margin,max_shape_ratio=args.id_shape_ratio,
       max_gap_seconds=args.id_max_gap,max_center_distance=args.id_max_distance,
       confirm_frames=args.id_confirm_frames,swap_confirm_frames=args.id_swap_confirm_frames,
       max_verify_frames=args.id_max_verify_frames)
       if enable_id_memory else None)
    print('[ID MEMORY]', 'ON' if enable_id_memory else 'OFF',
          '| live tracker max_age:',tracker.max_age,
          '| mode:', 'ROBUST FULL RE-ID' if args.mode=='sr_osnet_full' else args.mode,
          '| refresh interval:',args.appearance_refresh if args.mode=='sr_osnet_selective' else 0,
          flush=True)
    if enable_id_memory:
        print('[NOTE] Persistent DISPLAY ID is appearance-relinked; internal track IDs may still change.',flush=True)

    stamp=datetime.now().strftime('%Y%m%d_%H%M%S')
    out=(args.out or (root/'outputs'/EXPERIMENT)).expanduser().resolve()/f'{src_label}_{args.mode}_{stamp}'
    out.mkdir(parents=True,exist_ok=False)
    api.opt.ecc[src_label]={}
    print(f'[OUTPUT] {out}',flush=True)
    print('[INFO] Camera preview: Q to stop, S to save snapshot; default frame limit=0 (UNLIMITED)',flush=True)

    ret,first=cap.read()
    if not ret:raise RuntimeError('Cannot read first frame')
    first_original=first.copy() if args.mot17_seq else None
    if args.mirror:first=cv2.flip(first,1)
    if args.rotate==180:first=cv2.rotate(first,cv2.ROTATE_180)
    with torch.no_grad():
        for _ in range(2):
            b,c=detector.detect(first,conf=args.conf,imgsz=args.imgsz,iou=args.iou)
            if b:extractor([get_crop(first,b[j]) for j in range(min(len(b),4))])
        sync(args.device)

    # Live camera/video behavior is unchanged; MOT17 reuses warmed-up frame 1
    # so ground-truth and predictions are frame-aligned.
    raw_writer=None
    annotated_writer=None
    prev=None
    fps_window=deque(maxlen=25)
    start=time.perf_counter()
    total_detected=total_reid=0
    total_low_rescue_candidates=0
    zero_detection_frames=0
    predicted_box_frames=0
    frame_i=0
    last_fps=0.0
    exit_reason='end_of_stream'
    failed_reads=0
    mot_file=(out/'mot_results.txt').open('w',newline='',encoding='utf-8') if args.mot_output else None
    mot_writer=csv.writer(mot_file) if mot_file is not None else None
    metrics_file=(out/'frames.csv').open('w',newline='',encoding='utf-8')
    metrics_writer=csv.DictWriter(metrics_file,fieldnames=['frame','timestamp_seconds','detections','reid_features','visible_confirmed_ids','id_restores','id_new','low_conf_rescue_candidates','low_conf_rescues_matched','predicted_boxes','detector_ms','ecc_ms','predict_select_ms','reid_ms','association_ms','id_memory_ms','wall_ms','rolling_fps'])
    metrics_writer.writeheader()
    events_file=(out/'identity_events.csv').open('w',newline='',encoding='utf-8')
    events_writer=csv.DictWriter(events_file,fieldnames=['event','frame','seconds','internal_id','display_id','similarity','absent_seconds','method'])
    events_writer.writeheader()
    try:
        while True:
            t_frame=time.perf_counter()
            if first_original is not None:
                ok,frame=True,first_original
                first_original=None
            else:
                ok,frame=cap.read()
            if not ok:
                if not args.input and not args.mot17_seq and failed_reads<args.read_retries:
                    failed_reads+=1
                    print(f'[WARN] camera read failed; retry {failed_reads}/{args.read_retries}',flush=True)
                    time.sleep(.1)
                    continue
                exit_reason='camera_read_failure' if not (args.input or args.mot17_seq) else 'end_of_file'
                print('[STOP] Camera/video frame read failed or source ended',flush=True)
                break
            failed_reads=0
            if args.mirror:frame=cv2.flip(frame,1)
            if args.rotate==180:frame=cv2.rotate(frame,cv2.ROTATE_180)
            frame_i+=1
            height,width=frame.shape[:2]
            if (args.record_raw or args.record_annotated) and (raw_writer is None and annotated_writer is None):
                fourcc=cv2.VideoWriter_fourcc(*'mp4v')
                if args.record_raw:
                    raw_writer=cv2.VideoWriter(str(out/'raw_camera.mp4'),fourcc,max(1.,min(input_fps,30)),(width,height))
                    if not raw_writer.isOpened():raise RuntimeError('Cannot open raw video writer')
                if args.record_annotated:
                    annotated_writer=cv2.VideoWriter(str(out/'annotated.mp4'),fourcc,max(1.,min(input_fps,30)),(width,height))
                    if not annotated_writer.isOpened():raise RuntimeError('Cannot open annotated video writer')
            if raw_writer is not None:raw_writer.write(frame)
            sync(args.device);t=time.perf_counter()
            with torch.no_grad():
                # A low score is used ONLY to rescue already tracked people.
                # Normal confirmed detections must still pass --conf.
                detector_conf=(args.detector_floor if args.detector_floor is not None else
                               (args.occlusion_low_conf if args.mode.startswith('sr_') and not args.no_occlusion_rescue
                               else args.conf))
                raw_boxes,raw_confs=detector.detect(frame,conf=detector_conf,imgsz=args.imgsz,iou=args.iou)
            boxes=[b for b,c in zip(raw_boxes,raw_confs) if c>=args.conf]
            confs=[c for c in raw_confs if c>=args.conf]
            high_count=len(boxes)
            lows=[b for b,c in zip(raw_boxes,raw_confs) if c<args.conf]
            low_scores=[c for c in raw_confs if c<args.conf]
            sync(args.device);det_sec=time.perf_counter()-t
            t=time.perf_counter()
            matrix=np.eye(3) if args.disable_ecc else make_ecc_matrix(prev,frame)
            prev=frame
            api.opt.ecc[src_label][str(frame_i)]=matrix
            # Keep ECC cache bounded for arbitrarily long camera sessions.
            if frame_i>5:api.opt.ecc[src_label].pop(str(frame_i-5),None)
            ecc_sec=time.perf_counter()-t
            sync(args.device);t=time.perf_counter()
            if not args.disable_ecc:tracker.camera_update(src_label,frame_i)
            tracker.predict()
            selected_low=[]
            low_target_track_ids=[]
            if args.mode.startswith('sr_') and not args.no_occlusion_rescue and lows:
                low_pairs=choose_low_conf_rescue(
                    tracker,boxes,lows,max_missed=args.occlusion_max_missed,return_pairs=True)
                selected_low=[j for j,_ in low_pairs]
                low_target_track_ids=[tid for _,tid in low_pairs]
                boxes.extend(lows[j] for j in selected_low)
                confs.extend(low_scores[j] for j in selected_low)
            total_low_rescue_candidates+=len(selected_low)
            if args.mode=='sr_osnet_selective':
                chosen=[j for j in tracker.select_reid(boxes,confs) if j<high_count]
                # New / re-entering people require an appearance sample, even
                # when normal selective scheduling skips other Re-ID crops.
                need=set(chosen)
                for j in range(high_count):
                    near_tracks=[t for t in tracker.tracks
                                 if t.is_confirmed() and not t.is_deleted()
                                 and t.time_since_update<=1
                                 and iou_xywh(t.to_tlwh(),boxes[j])>.40]
                    if not near_tracks:
                        need.add(j)
                        continue
                    # 2-3 extra crops ONLY for people being verified or
                    # suspected of an ID switch. Never re-encode everyone.
                    if identity_bank is not None and any(
                        int(t.track_id) in identity_bank.pending or
                        identity_bank.suspect.get(int(t.track_id),0)>0
                        for t in near_tracks):
                        need.add(j)
                chosen=sorted(need)
                # Live-only periodic appearance refresh for changes in pose/light.
                # Not part of the original research ablation and counted in Re-ID.
                if args.appearance_refresh and frame_i%args.appearance_refresh==0:
                    need=set(chosen)
                    for j,b in enumerate(boxes):
                        if confs[j]<max(args.conf,.35):continue
                        if any(t.is_confirmed() and t.time_since_update<=1
                               and iou_xywh(t.to_tlwh(),b)>.35 for t in tracker.tracks):
                            need.add(j)
                    chosen=sorted(need)
            else:
                chosen=list(range(high_count)) if args.mode.startswith('sr_') else list(range(len(boxes)))
            select_predict_sec=time.perf_counter()-t
            sync(args.device);t=time.perf_counter()
            with torch.no_grad():
                if chosen:
                    crops=[get_crop(frame,boxes[j]) for j in chosen]
                    feats=extractor(crops)
                    if len(feats)!=len(chosen):raise RuntimeError('Feature count mismatch')
                    feature_map=dict(zip(chosen,feats))
                else:feature_map={}
            sync(args.device);feat_sec=time.perf_counter()-t
            t=time.perf_counter()
            if args.mode.startswith('strongsort_'):
                det_objs=[api.Detection(box,conf,feature_map[j]) for j,(box,conf) in enumerate(zip(boxes,confs))]
            else:
                occlusions=[max((iou_xywh(b,bb) for j,bb in enumerate(boxes) if i!=j),default=0.)
                           for i,b in enumerate(boxes)]
                det_objs=[]
                for j,(box,conf) in enumerate(zip(boxes,confs)):
                    det=SRDetection(box,conf,feature_map.get(j),occlusions[j])
                    det.rescue_low=(j>=high_count)
                    if det.rescue_low:det.rescue_track_id=low_target_track_ids[j-high_count]
                    det_objs.append(det)
            tracker.update(det_objs)
            sync(args.device);assoc_sec=time.perf_counter()-t
            total_detected+=len(boxes);total_reid+=len(chosen)
            zero_detection_frames+=int(not boxes)
            predicted_box_frames+=int(any(t.is_confirmed() and not t.is_deleted() and 0<t.time_since_update<=args.predicted_hold for t in tracker.tracks))
            duration=time.perf_counter()-start
            display_ids={}
            id_events=[]
            t_id_memory=time.perf_counter()
            if identity_bank is not None:
                # Store only actually observed, confirmed tracks (avoid false
                # positive tentative memory); prefer fresh detector crop embeddings.
                observed=observed_tracks_with_fresh_features(tracker,boxes,feature_map,identity_bank,(width,height))
                identity_seconds=(frame_i-1)/input_fps if args.mot17_seq else duration
                display_ids,id_events=identity_bank.update(observed,identity_seconds,frame_i)
                for event in id_events:
                    events_writer.writerow({**event,'seconds':round(duration,3)})
                    if event['event']=='restored':
                        print(f"[ID-RESTORED] display ID {event['display_id']} "
                              f"<- internal track {event['internal_id']}; "
                              f"method={event['method']}, "
                              f"OSNet={event['similarity'] if event['similarity'] is not None else 'N/A'}, "
                              f"gap={event['absent_seconds']:.1f}s",flush=True)
            id_memory_sec=time.perf_counter()-t_id_memory
            if mot_writer is not None:
                save_mot_tracks(mot_writer,tracker,identity_bank,frame_i,(width,height))
            key=-1
            if args.no_render:
                # Benchmark-only mode; excludes box drawing, display, video encoding.
                # Both competitors execute the same measured operations.
                visible=sum(1 for tr in tracker.tracks if tr.is_confirmed() and not tr.is_deleted()
                            and int(tr.time_since_update)==0 and
                            (identity_bank is None or
                             isinstance(identity_bank.bindings.get(int(tr.track_id)),(int,np.integer))))
            else:
                preview=frame.copy()
                visible=draw(preview,args.mode,tracker,boxes,chosen,last_fps,frame_i,total_detected,total_reid,input_fps,duration,
                             display_ids={**identity_bank.bindings,**display_ids} if identity_bank else None,
                             id_restores=identity_bank.total_restores if identity_bank else 0,
                             show_internal=args.show_internal_id,
                             predicted_hold=args.predicted_hold,
                             low_rescued=len(getattr(tracker,'last_low_update_ids',set())),
                             provisional_ids=identity_bank.provisional_internals if identity_bank else None)
                if annotated_writer is not None:annotated_writer.write(preview)
                if not args.no_display:
                    cv2.imshow('BARAM | Live SR-StrongSORT',preview)
                    key=cv2.waitKey(1)&0xFF
            sync(args.device)
            frame_wall=time.perf_counter()-t_frame
            fps_window.append(frame_wall)
            last_fps=len(fps_window)/sum(fps_window)
            metrics_writer.writerow({'frame':frame_i,'timestamp_seconds':round(duration,4),
                'detections':len(boxes),'reid_features':len(chosen),'visible_confirmed_ids':visible,
                'id_restores':identity_bank.total_restores if identity_bank else 0,
                'id_new':identity_bank.total_new if identity_bank else 0,
                'low_conf_rescue_candidates':len(selected_low),
                'low_conf_rescues_matched':len(getattr(tracker,'last_low_update_ids',set())),
                'predicted_boxes':sum(1 for q in tracker.tracks if q.is_confirmed() and not q.is_deleted()
                                      and 0<q.time_since_update<=args.predicted_hold),
                'detector_ms':round(det_sec*1000,3),'ecc_ms':round(ecc_sec*1000,3),
                'predict_select_ms':round(select_predict_sec*1000,3),
                'reid_ms':round(feat_sec*1000,3),'association_ms':round(assoc_sec*1000,3),
                'id_memory_ms':round(id_memory_sec*1000,3),
                'wall_ms':round(frame_wall*1000,3),'rolling_fps':round(last_fps,3)})
            if frame_i%30==0:
                metrics_file.flush()
                events_file.flush()
                print(f'[LIVE] frame={frame_i}, fps={last_fps:.2f}, ids={visible}, det={len(boxes)}, '
                      f'ReID extracted={total_reid}/{total_detected}, low-rescue={len(getattr(tracker,"last_low_update_ids",set()))}',flush=True)
            if key==ord('s'):
                path=out/f'snapshot_{frame_i:06d}.jpg'
                cv2.imwrite(str(path),preview)
                print('[SNAPSHOT]',path,flush=True)
            if key==ord('q'):
                exit_reason='user_q'
                break
            if args.max_frames and frame_i>=args.max_frames:
                exit_reason='max_frames'
                break
    except KeyboardInterrupt:
        exit_reason='keyboard_interrupt'
        print('\n[STOP] Ctrl+C',flush=True)
    finally:
        wall=time.perf_counter()-start
        cap.release()
        if raw_writer is not None:raw_writer.release()
        if annotated_writer is not None:annotated_writer.release()
        if not args.no_display:
            cv2.destroyAllWindows()
        metrics_file.close()
        events_file.close()
        if mot_file is not None:mot_file.close()
        # These numbers are a demo diagnostic; recording/GUI/capture impact the FPS.
        cuda_allocated_peak_mib = None
        cuda_reserved_peak_mib = None
        if cuda_stats_enabled:
            try:
                cuda_allocated_peak_mib = round(
                    torch.cuda.max_memory_allocated(device=cuda_stats_device)/2**20, 2)
                cuda_reserved_peak_mib = round(
                    torch.cuda.max_memory_reserved(device=cuda_stats_device)/2**20, 2)
            except (RuntimeError, ValueError, TypeError) as exc:
                print(f'[WARN] PyTorch allocator peak read unavailable: {exc}', flush=True)

        summary={
            'source':src_label,'mode':args.mode,'detector':'YOLO26x COCO person','weights_yolo26x_path':str(detector.weights_path),
            'weights_yolo26x_sha256':sha(detector.weights_path),
            'adapter_sha256':sha(req['Optimized StrongSORT bridge']),
            'input_video':str(args.input) if args.input else None,
            'mot17_sequence':str(args.mot17_seq) if args.mot17_seq else None,
            'mot17_predictions':'mot_results.txt' if args.mot_output else None,
            'mot17_timebase':'frame_timestamps' if args.mot17_seq else 'wall_clock',
            'reported_input_fps':input_fps,'frames_processed':frame_i,
            'total_person_detections':total_detected,'reid_features_extracted':total_reid,
            'reid_features_saved_vs_full':total_detected-total_reid,
            'reid_reduction_percent':100*(1-total_reid/max(1,total_detected)) if total_detected else 0.,
            'wall_seconds':wall,'mean_processing_fps_including_capture_display':frame_i/max(1e-9,wall),
            'disable_ecc':args.disable_ecc,'mirror':args.mirror,'rotate':args.rotate,
            'id_memory_enabled':enable_id_memory,
            'id_memory_seconds':args.id_memory_seconds,'id_threshold':args.id_threshold,
            'id_multi_threshold':args.id_multi_threshold,'id_max_gap_seconds':args.id_max_gap,
            'id_max_distance':args.id_max_distance,
            'live_motion_gate':args.motion_gate,'live_n_init':args.tracker_n_init,
            'live_relink_enabled':not args.no_live_relink,'live_relink_similarity':args.live_relink_sim,
            'live_internal_matches_recovered':getattr(tracker,'live_reattach_count',0),
            'id_margin':args.id_margin,'tracker_max_age':args.tracker_max_age,
            'id_confirm_frames':args.id_confirm_frames,'id_swap_confirm_frames':args.id_swap_confirm_frames,
            'id_max_verify_frames':args.id_max_verify_frames,
            'id_pending_display_frames':identity_bank.total_pending_frames if identity_bank else 0,
            'id_overlap_provisional_frames':identity_bank.total_occluded_frames if identity_bank else 0,
            'new_ids_suppressed_ambiguous':identity_bank.total_new_suppressed if identity_bank else 0,
            'id_restore_policy':'V4.3 clean appearance only, overlap holds, novel ID quarantine, atomic swap',
            'appearance_refresh_frames':args.appearance_refresh if args.mode=='sr_osnet_selective' else 0,
            'occlusion_low_conf':args.occlusion_low_conf,'occlusion_rescue_enabled':not args.no_occlusion_rescue,
            'occlusion_max_missed':args.occlusion_max_missed,'predicted_hold_frames':args.predicted_hold,
            'low_conf_candidates_admitted':total_low_rescue_candidates,
            'low_conf_rescues_matched':getattr(tracker,'low_conf_rescues',0),
            'frames_zero_person_detection':zero_detection_frames,
            'frames_with_predicted_hold_box':predicted_box_frames,
            'recovered_display_id_links':identity_bank.total_restores if identity_bank else 0,
            'short_gap_spatial_restores':identity_bank.total_spatial if identity_bank else 0,
            'appearance_corrected_display_swaps':identity_bank.total_corrected_swaps if identity_bank else 0,
            'new_display_ids':identity_bank.total_new if identity_bank else 0,
            'identity_events_csv':'identity_events.csv',
            'raw_recording':bool(args.record_raw),'annotated_recording':bool(args.record_annotated),
            'torch_cuda_peak_allocated_mib':cuda_allocated_peak_mib,
            'torch_cuda_peak_reserved_mib':cuda_reserved_peak_mib,
            'exit_reason':exit_reason,'max_frames_setting':args.max_frames,
            'conf':args.conf,'imgsz':args.imgsz,'iou':args.iou,
            'detector_floor':args.detector_floor,'no_render':args.no_render,
            'limitations':'V4.3 cannot guarantee identity under complete occlusion or identical clothing. No true IDSW measurement without GT. Appearance-only display-ID relinking can remain uncertain for look-alike clothing, changes in pose, lighting and full exits. VERIFY means identity withheld until sufficient evidence, not tracking failure. No guaranteed IDSW=0. PRED boxes are estimates, not detections. Live FPS includes camera capture, display and optional video encode; no GT so HOTA/IDF1/IDSW unavailable. Do not mix with MOT benchmark.'
        }
        (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
        print('[SUMMARY] frames',frame_i,'processed FPS',f'{summary["mean_processing_fps_including_capture_display"]:.3f}',
              'ReID',f'{total_reid}/{total_detected}',
              'display ID restores',identity_bank.total_restores if identity_bank else 0,
              'internal reattach',getattr(tracker,'live_reattach_count',0),
              'display swaps corrected',identity_bank.total_corrected_swaps if identity_bank else 0,flush=True)
        print('[SAVED]',out/'summary.json',flush=True)
        if args.mot_output:print('[SAVED]',out/'mot_results.txt',flush=True)
        print('[NOTE] This script exports MOT predictions, but does not compute official TrackEval metrics.',flush=True)


if __name__=='__main__':
    try:main()
    except Exception as e:
        print('[ERROR]',str(e),file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)
