"""Run reproducible official StrongSORT vs proposed selective OSNet tracker on MOT train.

Usage: python -m baram.run --dataset MOT17 --mode all --max-frames 100
Never publishes synthetic metrics or calls outputs official paper reproductions.
"""
from __future__ import annotations
import argparse
import csv
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import cv2
from .upstream import load
from .data import dataset_sequences,list_frames,cache_yolo,sha256
from .extractors import OSNet, BoTFastReID
from .selective import SelectiveTracker,SRConfig
from .official import OfficialStrongSORT


def sync(device):
    if device.startswith('cuda'):
        import torch
        torch.cuda.synchronize()


def init_tracker(mode,src,root,device,sr_config=None):
    if mode=='strongsort':
        extractor=BoTFastReID(root,device)
        tracker=OfficialStrongSORT(src)
    else:
        extractor=OSNet(root,device)
        tracker=SelectiveTracker(src,extractor,sr_config or SRConfig())
    return tracker,extractor


def write_tracks(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',encoding='utf-8') as f:
        for fr,tid,x,y,w,h in rows:
            f.write(f'{fr:d},{tid:d},{x:.2f},{y:.2f},{w:.2f},{h:.2f},1,-1,-1,-1\n')


def run_one(seqdir,dets,det_meta,mode,src,root,out,device,frames_to_run,max_frames,video=False,ecc=False,
            tracking_fingerprint=None,sr_config=None):
    output=out/'trackers'/mode/'data'/(seqdir.name+'.txt')
    statsfile=out/'stats'/mode/(seqdir.name+'.json')
    if statsfile.is_file() and output.is_file():
        old=json.loads(statsfile.read_text())
        if (old.get('frames')==len(frames_to_run)
                and old.get('detection_fingerprint')==det_meta
                and old.get('tracking_fingerprint')==tracking_fingerprint):
            print('[REUSE]',mode,seqdir.name,flush=True)
            return old
    tracker,extractor=init_tracker(mode,src,root,device,sr_config)
    fps_video=30.; h,w=0,0;writer=None
    if video:
        info=seqdir/'seqinfo.ini'
        if info.exists():
            import configparser
            c=configparser.ConfigParser();c.read(info)
            fps_video=c.getfloat('Sequence','frameRate',fallback=30.)
        sample=cv2.imread(str(frames_to_run[0][1]));h,w=sample.shape[:2]
        clip=out/'video'/mode/(seqdir.name+'.mp4');clip.parent.mkdir(parents=True,exist_ok=True)
        writer=cv2.VideoWriter(str(clip),cv2.VideoWriter_fourcc(*'mp4v'),fps_video,(w,h))
        if not writer.isOpened():raise RuntimeError('MP4 writer unavailable: '+str(clip))
    row_by_frame={}
    for d in dets:row_by_frame.setdefault(int(d[0]),[]).append(d)
    rows=[];reid_elapsed=0.;track_elapsed=0.;feature_total=0; batches=0
    try:
        for k,(frame,file) in enumerate(frames_to_run,1):
            im=cv2.imread(str(file))
            if im is None:raise RuntimeError('Bad image '+str(file))
            detections=[src['Detection'](np.array(r[2:6],dtype=float),float(r[6]),None)
                        for r in row_by_frame.get(frame,[])]
            sync(device);t=time.perf_counter()
            if mode=='strongsort':
                features=extractor.extract(im,[d.tlwh for d in detections])
                sync(device);reid_elapsed+=time.perf_counter()-t
                feature_total+=len(features);batches+=bool(len(features))
                t=time.perf_counter()
                current=tracker.process(features,detections,seqdir.name,frame,ecc)
            else:
                # Upstream NSA/EMA track API kept in SelectiveTracker;
                # extractor is invoked only for proposed selected detections.
                current=tracker.process(im,detections,seqdir.name,ecc)
            sync(device);track_elapsed+=time.perf_counter()-t
            for tid,xywh in current:
                x,y,tw,th=map(float,xywh)
                if tw>0 and th>0:rows.append((frame,int(tid),x,y,tw,th))
            if writer is not None:
                vis=im.copy()
                for tid,(x,y,bw,bh) in current:
                    cv2.rectangle(vis,(int(x),int(y)),(int(x+bw),int(y+bh)),(40,210,50),2)
                    cv2.putText(vis,f'ID {tid}',(int(x),max(20,int(y)-5)),cv2.FONT_HERSHEY_SIMPLEX,.65,(255,255,255),2)
                cv2.putText(vis,f'{mode} | frame {frame}',(12,28),cv2.FONT_HERSHEY_SIMPLEX,.7,(30,210,255),2)
                writer.write(vis)
            if k%100==0 or k==len(frames_to_run):
                print(f'[{mode} {seqdir.name}] {k}/{len(frames_to_run)}',flush=True)
    finally:
        if writer:writer.release()
    write_tracks(output,rows)
    if mode!='strongsort':
        feature_total=int(tracker.stats['reid_features'])
        batches=int(tracker.stats['reid_batches'])
        reid_elapsed=float(tracker.stats['reid_seconds'])
        # track_elapsed already includes SR Re-ID time; do not double count.
        association_elapsed=float(tracker.stats['association_seconds'])
        additional=dict(tracker.stats)
    else:
        association_elapsed=track_elapsed
        additional={}
    total_tracker=track_elapsed if mode!='strongsort' else track_elapsed+reid_elapsed
    detector_s=float(det_meta['detector_seconds'])
    per_frame_detector=detector_s/max(1,int(det_meta['frames']))
    comparable_seconds=total_tracker+len(frames_to_run)*per_frame_detector
    result={'model':mode,'sequence':seqdir.name,'frames':len(frames_to_run),'detections':sum(len(row_by_frame.get(fr,[])) for fr,_ in frames_to_run),
            'reid_features':feature_total,'reid_batches':batches,'reid_seconds':reid_elapsed,
            'association_seconds':association_elapsed,'tracker_seconds':total_tracker,
            'shared_yolo_seconds':len(frames_to_run)*per_frame_detector,
            'estimated_pipeline_fps':len(frames_to_run)/max(comparable_seconds,1e-10),
            'timing_note':'detector_seconds measured once, then added to this mode; video decode/drawing/writing excluded; not an isolated end-to-end live FPS',
            'detection_fingerprint':det_meta,'tracking_fingerprint':tracking_fingerprint,
            'ecc':bool(ecc),'upstream_tracker_source':mode=='strongsort',
            'mode_stats':additional}
    statsfile.parent.mkdir(parents=True,exist_ok=True)
    statsfile.write_text(json.dumps(result,indent=2))
    return result


def pp_one(input_file,output_file,model,dataset_transform):
    """Native AFLink and GSI from original repo; numerical shape compatibility only."""
    from AFLink.AppFreeLink import AFLink
    import GSI
    import numpy as np
    if not input_file.is_file():raise FileNotFoundError(input_file)
    if input_file.stat().st_size==0:
        output_file.write_text('');return
    data=np.loadtxt(input_file,delimiter=',',ndmin=2)
    if len(data)<2:
        output_file.write_text(input_file.read_text());return
    temp=output_file.with_name(output_file.stem+'_aflink.txt')
    AFLink(path_in=str(input_file),path_out=str(temp),model=model,dataset=dataset_transform,
           thrT=(0,30),thrS=75,thrP=.05).link()
    # Original GSI.py assumes sklearn.GPR.predict(N,1) returns an Nx1 matrix;
    # sklearn newer versions can return N-vector. Adapt only *shape* in memory.
    OriginalGPR=GSI.GPR
    class CompatibleGPR(OriginalGPR):
        def predict(self,X,*a,**kw):
            pred=super().predict(X,*a,**kw)
            if isinstance(pred,tuple):return pred
            return pred.reshape(-1,1) if np.asarray(pred).ndim==1 else pred
    GSI.GPR=CompatibleGPR
    try:GSI.GSInterpolation(str(temp),str(output_file),interval=20,tau=10)
    finally:GSI.GPR=OriginalGPR


def postprocess(root,out,dataset,names,device):
    import torch
    if not device.startswith('cuda'):
        raise RuntimeError('Original AFLink hardcodes .cuda(): StrongSORT++ needs CUDA.')
    from AFLink.model import PostLinker
    from AFLink.dataset import LinkData
    weights=root/'weights'/'AFLink_epoch20.pth'
    if not weights.is_file():raise FileNotFoundError(weights)
    model=PostLinker()
    # Only load explicitly trusted model files.
    obj=torch.load(str(weights),map_location='cpu',weights_only=True)
    model.load_state_dict(obj)
    transform=LinkData('','')
    results=[]
    for seq in names:
        inp=out/'trackers'/'strongsort'/'data'/f'{seq}.txt'
        dest=out/'trackers'/'strongsortpp'/'data'/f'{seq}.txt'
        dest.parent.mkdir(parents=True,exist_ok=True)
        before=time.perf_counter()
        pp_one(inp,dest,model,transform)
        seconds=time.perf_counter()-before
        print('[POST] original AFLink -> GSI',seq,f'{seconds:.2f}s')
        results.append({'sequence':seq,'postprocess_seconds':seconds})
    (out / f'postprocess_{dataset}.json').write_text(json.dumps(results,indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project',type=Path,default=Path(__file__).resolve().parents[1]);
    p.add_argument('--dataset',choices=['MOT17','MOT20','both'],default='MOT17')
    p.add_argument('--mode',choices=['strongsort','srstrongsort','both','all'],default='both')
    p.add_argument('--device',default='cuda:0');p.add_argument('--conf',type=float,default=.25)
    p.add_argument('--imgsz',type=int,default=640);p.add_argument('--max-det',type=int,default=1000)
    p.add_argument('--max-frames',type=int,default=0)
    p.add_argument('--video',action='store_true')
    p.add_argument('--ecc-dir',type=Path,default=None,help='Directory containing official MOT17_ECC_val.json and MOT20_ECC_val.json')
    p.add_argument('--require-ecc',action='store_true',help='Fail rather than silently disabling official camera compensation')
    p.add_argument('--sr-config',type=Path,default=None,
                   help='JSON object overriding SRConfig fields for reproducible MOT train tuning')
    p.add_argument('--output',type=Path,default=None)
    a=p.parse_args()
    if not 0<a.conf<1 or a.max_frames<0 or a.max_det<1:p.error('Invalid confidence, max-det, or max-frames')
    root=a.project.resolve();out=(a.output or root/'outputs'/'study_v1').resolve()
    sr_config=SRConfig()
    if a.sr_config:
        config_path=a.sr_config if a.sr_config.is_absolute() else root/a.sr_config
        raw=json.loads(config_path.read_text())
        if not isinstance(raw,dict):p.error('--sr-config must contain a JSON object')
        try:sr_config=SRConfig(**raw)
        except TypeError as exc:p.error(f'Invalid --sr-config: {exc}')
        if not (0<=sr_config.iou_gate<=1 and 0<=sr_config.ars_gate<=1
                and sr_config.maximum_cost>0 and 0<=sr_config.motion_weight<=1
                and 0<=sr_config.appearance_weight<=1 and sr_config.max_age>=1
                and sr_config.min_hits>=1 and 0<=sr_config.new_track_conf<=1
                and 0<=sr_config.recovery_similarity<=1 and sr_config.recovery_window>=1
                and sr_config.recovery_max_distance>0 and sr_config.appearance_refresh>=0):
            p.error('SRConfig thresholds and lifecycle values are outside valid ranges')
    datasets=['MOT17','MOT20'] if a.dataset=='both' else [a.dataset]
    modes=['strongsort','srstrongsort'] if a.mode in ('both','all') else [a.mode]
    weight=root/'weights'/'yolo26n.pt'
    if not weight.is_file():raise FileNotFoundError('Expected trained COCO YOLO26n at '+str(weight))
    out.mkdir(parents=True,exist_ok=True)
    all_stats=[]
    for ds in datasets:
        ecc_path=(a.ecc_dir/f'{ds}_ECC_val.json') if a.ecc_dir else None
        ecc_data=json.loads(ecc_path.read_text()) if ecc_path and ecc_path.is_file() else None
        if a.require_ecc and ecc_data is None:
            raise FileNotFoundError(f'Official ECC JSON required: {ecc_path} (see README)')
        if ecc_data is None:
            print(f'[WARNING] {ds} ECC disabled; this is NOT the full official StrongSORT flag configuration',flush=True)
        src=load(root,ds,ecc_data)
        ecc_fingerprint=(json.dumps(ecc_data,sort_keys=True,separators=(',',':'))
                         if ecc_data is not None else None)
        seqroot=root/'datasets';seqs=dataset_sequences(seqroot,ds)
        if not seqs:raise RuntimeError('No '+ds+' train sequences found')
        for seq in seqs:
            directory=seqroot/ds/'train'/seq
            subset=list_frames(directory)
            if a.max_frames:subset=subset[:a.max_frames]
            if not subset:continue
            cache_dir=out/'yolo_cache'/ds/seq
            detections,meta=cache_yolo(directory,cache_dir,weight,a.device,a.conf,a.imgsz,
                                      a.max_frames,a.max_det)
            for mode in modes:
                reid_weights=(root/'weights'/('duke_bot_S50.pth' if mode=='strongsort'
                                               else 'osnet_x1_0.pth.tar'))
                tracking_fingerprint={
                    'schema':2,
                    'mode':mode,
                    'video_output':bool(a.video),
                    'tracker_config':None if mode=='strongsort' else asdict(sr_config),
                    'reid_weights_sha256':sha256(reid_weights),
                    'ecc_json_sha256':hashlib.sha256(ecc_fingerprint.encode()).hexdigest()
                                        if ecc_fingerprint is not None else None,
                    'frame_ids_sha256':hashlib.sha256(
                        ','.join(str(fr) for fr,_ in subset).encode()).hexdigest(),
                    'tracker_code_sha256':hashlib.sha256(''.join(
                        sha256(root/'baram'/name) for name in (
                            ['run.py','official.py','upstream.py','compat.py','extractors.py']+
                            (['selective.py','geometry.py'] if mode=='srstrongsort' else []))
                    ).encode()).hexdigest(),
                    'strongsort_revision':'ee995076da5083e28d0da1f885297df62705ebd7',
                }
                result=run_one(directory,detections,meta,mode,src,root,out/ds,a.device,subset,
                               a.max_frames,a.video,ecc=bool(ecc_data),
                               tracking_fingerprint=tracking_fingerprint,
                               sr_config=sr_config)
                result.update({'dataset':ds})
                all_stats.append(result)
        if a.mode=='all':
            if 'strongsort' not in modes:raise RuntimeError('StrongSORT needed for StrongSORT++')
            postprocess(root,out/ds,ds,seqs,a.device)
    fields=['dataset','model','sequence','frames','detections','reid_features','reid_batches','reid_seconds',
            'tracker_seconds','shared_yolo_seconds','estimated_pipeline_fps','ecc']
    with (out/'performance.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(all_stats)
    print('[COMPLETE] Tracker output: ',out,flush=True)
    print('For *actual* HOTA/MOTA/IDF1 call scripts/evaluate.py on full train sequences only.',flush=True)

if __name__=='__main__':main()
