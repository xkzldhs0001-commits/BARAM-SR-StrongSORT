#!/usr/bin/env python3
"""Fair same-detector MOT17/MOT20 A/B: StrongSORT+YOLO26x vs BARAM V4.3+YOLO26x.

Requires existing BARAM project and MOTChallenge *train* images/GT. No fake scores.
TrackEval is the reference evaluator for HOTA / CLEAR MOTA+IDSW / IDF1.
"""
from __future__ import annotations
import argparse
import csv
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import threading
import time

HERE=Path(__file__).resolve().parent
ROOT_DEFAULT=Path.home()/'Downloads'/'BARAM_SR_StrongSORT'
OUT_DEFAULT=ROOT_DEFAULT/'outputs'/'BARAM_YOLO26X_STRONGSORT_AB'
PREFERRED={'MOT17':'FRCNN'}
MODES={'baseline':'strongsort_osnet','v43_full':'sr_osnet_full','v43_selective':'sr_osnet_selective'}
SCANNED_BLACKLIST={'outputs','venv','.venv','.git','third_party','site-packages','.cache','runs','wandb',
                   '__pycache__','node_modules','images','build','dist'}
SEQUENCE_PATTERN=re.compile(r'^MOT(17|20)-(\d\d)(?:-(FRCNN|DPM|SDP))?$')


def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()


def collect_sequences(search_dir):
    """Discover available MOT17/20 train sequences, without duplicate MOT17 image streams."""
    candidates={}
    for directory, subdirs, files in os.walk(str(search_dir)):
        subdirs[:]=[d for d in subdirs if d not in SCANNED_BLACKLIST]
        folder=Path(directory)
        match=SEQUENCE_PATTERN.fullmatch(folder.name)
        if match and 'seqinfo.ini' in files and (folder/'img1').is_dir():
            dataset='MOT'+match.group(1)
            seqnum=match.group(2)
            if (folder/'gt'/'gt.txt').is_file():
                candidates.setdefault((dataset,seqnum),[]).append(folder.resolve())
            subdirs[:]=[]
    result={'MOT17':[],'MOT20':[]}
    for (ds,_),paths in sorted(candidates.items()):
        if ds=='MOT17':
            rank={'FRCNN':0,'SDP':1,'DPM':2}
            paths.sort(key=lambda p:(rank.get(p.name.rsplit('-',1)[-1],3),str(p)))
        else:paths.sort()
        result[ds].append(paths[0])
    return result


def process_vram_mib(pid):
    try:
        res=subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_gpu_memory',
                            '--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=3)
        if res.returncode: return None
        for line in res.stdout.splitlines():
            a=[x.strip() for x in line.split(',')]
            if len(a)>=2 and a[0].isdigit() and int(a[0])==pid:
                m=re.search(r'\d+(?:\.\d+)?',a[1])
                return float(m.group(0)) if m else None
    except (OSError,subprocess.TimeoutExpired):pass
    return None


def run_with_log(cmd, log_path, cwd, env=None, sample_vram=False):
    print('\n[COMMAND]', ' '.join(str(x) for x in cmd),flush=True)
    stop=threading.Event(); vram=[]
    proc=subprocess.Popen([str(x) for x in cmd],cwd=str(cwd),env=env,
                          stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,
                          encoding='utf-8',errors='replace',bufsize=1)
    def sample():
        while not stop.is_set():
            n=process_vram_mib(proc.pid)
            if n is not None:vram.append(n)
            stop.wait(.5)
    watcher=None
    if sample_vram and shutil.which('nvidia-smi'):
        watcher=threading.Thread(target=sample,daemon=True)
        watcher.start()
    outdir=None
    try:
        with log_path.open('w',encoding='utf-8') as stream:
            for line in proc.stdout:
                print(line,end='',flush=True)
                stream.write(line)
                if line.startswith('[OUTPUT] '):outdir=Path(line.split('[OUTPUT] ',1)[1].strip())
        code=proc.wait()
    except KeyboardInterrupt:
        proc.terminate()
        try:proc.wait(timeout=5)
        except subprocess.TimeoutExpired:proc.kill();proc.wait()
        raise
    finally:
        stop.set()
        if watcher:watcher.join(timeout=3)
    return code,outdir,max(vram) if vram else None


def read_metric_summary(base):
    files=list(base.rglob('pedestrian_summary.txt'))
    if len(files)!=1:
        raise RuntimeError(f'Expected one TrackEval pedestrian_summary.txt under {base}, found {len(files)}')
    lines=[line.strip() for line in files[0].read_text(encoding='utf-8').splitlines() if line.strip()]
    if len(lines)<2:raise RuntimeError(f'Invalid TrackEval summary: {files[0]}')
    headers=lines[0].split()
    vals=lines[1].split()
    d=dict(zip(headers,vals))
    wanted=['HOTA','IDF1','MOTA','IDSW']
    missing=[x for x in wanted if x not in d]
    if missing:raise RuntimeError(f'Unexpected TrackEval summary (missing {missing}): {headers}')
    return {k:float(d[k]) for k in wanted}


def read_metric_details(base):
    files=list(base.rglob('pedestrian_detailed.csv'))
    if len(files)!=1:return {}
    with files[0].open('r',encoding='utf-8-sig',newline='') as f:
        rows=list(csv.DictReader(f))
    out={}
    for row in rows:
        seq=(row.get('seq') or row.get('sequence') or row.get('Sequence') or '').strip()
        if not seq:continue
        try:out[seq]={k:float(row[k]) for k in ('HOTA','IDF1','MOTA','IDSW')}
        except (KeyError,ValueError):continue
    return out


def report_rows(out,run_summaries,mode_names,metric_summaries,metric_details,results):
    per=[]; aggregate=[]
    for ds, seqs in results.items():
        for mode in mode_names:
            summaries=[run_summaries[(ds,s.name,mode)] for s in seqs]
            frames=sum(int(x['frames_processed']) for x in summaries)
            seconds=sum(float(x['wall_seconds']) for x in summaries)
            features=sum(int(x['reid_features_extracted']) for x in summaries)
            detections=sum(int(x['total_person_detections']) for x in summaries)
            frames_per_sec=frames/seconds if seconds else 0.
            metrics=metric_summaries.get((ds,mode),{})
            row={'dataset':ds,'method':mode,'sequences':len(seqs),'frames':frames,
                 'HOTA':metrics.get('HOTA'),'IDF1':metrics.get('IDF1'),
                 'MOTA':metrics.get('MOTA'),'IDSW':metrics.get('IDSW'),
                 'FPS':round(frames_per_sec,3),'ReID_extractions':features,
                 'detections':detections,'ReID_per_detection_pct':round(100*features/max(1,detections),2),
                 'peak_process_VRAM_MiB':max((x.get('_vram_peak_mib') for x in summaries
                                               if x.get('_vram_peak_mib') is not None),default=None)}
            aggregate.append(row)
            details=metric_details.get((ds,mode),{})
            for s in seqs:
                x=run_summaries[(ds,s.name,mode)]
                m=details.get(s.name,{})
                per.append({'dataset':ds,'sequence':s.name,'method':mode,
                            'HOTA':m.get('HOTA'),'IDF1':m.get('IDF1'),
                            'MOTA':m.get('MOTA'),'IDSW':m.get('IDSW'),
                            'FPS':round(x['frames_processed']/max(1e-9,x['wall_seconds']),3),
                            'frames':x['frames_processed'],
                            'ReID_extractions':x['reid_features_extracted'],
                            'detections':x['total_person_detections'],
                            'peak_process_VRAM_MiB':x.get('_vram_peak_mib')})
    for name, rows in [('comparison.csv',aggregate),('per_sequence.csv',per)]:
        with (out/name).open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]) if rows else [])
            if rows:writer.writeheader();writer.writerows(rows)
    def fmt(v,dec=2):return 'N/A' if v is None else (f'{v:,.{dec}f}' if isinstance(v,float) else f'{v:,}')
    lines=['# BARAM YOLO26x: StrongSORT baseline vs V4.3','',
           'MOTChallenge **train** GT 기반 TrackEval 점수. 이 실험은 공식 MOT17/20 test leaderboard 점수가 아닙니다.','',
           'FPS: GPU + 이미지 디코딩 + YOLO26x + OSNet + 트래커 + ID Memory + MOT 결과 저장(영상 생성/렌더링 제외).','',
           '| Dataset | Method | HOTA ↑ | IDF1 ↑ | MOTA ↑ | IDSW ↓ | FPS ↑ | Re-ID 횟수 ↓ | Peak VRAM MiB ↓ |',
           '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in aggregate:
        lines.append('| '+ ' | '.join([r['dataset'],r['method'],fmt(r['HOTA']),fmt(r['IDF1']),fmt(r['MOTA']),
                                        fmt(r['IDSW'],0),fmt(r['FPS'],2),fmt(r['ReID_extractions'],0),
                                        fmt(r['peak_process_VRAM_MiB'],0)])+' |')
    lines+=['','Re-ID 수는 실제 OSNet feature extraction 호출에서 반환받은 **검출 크롭 수**입니다.',
            '서로 다른 영상의 IDSW는 무조건 절대값을 직접 비교하면 안 됩니다. 같은 데이터셋/시퀀스의 동일 조건끼리 비교하세요.',
            'VERIFY 미확정 ID는 MOT 예측에 넣지 않으므로 FN·MOTA와 IDSW 모두 영향을 받을 수 있습니다.',
            'Vanilla baseline은 본 프로젝트의 공식 StrongSORT tracker core + OSNet으로, AFLink/GSI를 포함한 StrongSORT++가 아닙니다.','']
    (out/'comparison.md').write_text('\n'.join(lines),encoding='utf-8')
    print('\n[FINAL TABLE]\n'+'\n'.join(lines[:len(aggregate)+8]),flush=True)
    return aggregate


def main():
    p=argparse.ArgumentParser(description='MOT17 + MOT20 fair YOLO26x StrongSORT-vs-BARAM V4.3 benchmark')
    p.add_argument('--root',type=Path,default=ROOT_DEFAULT,help='Existing BARAM_SR_StrongSORT project')
    p.add_argument('--dataset-root',type=Path,default=Path.home()/'Downloads',help='Search directory containing MOT17/20 train sequences')
    p.add_argument('--trackeval',type=Path,default=Path.home()/'Downloads'/'TrackEval')
    p.add_argument('--datasets',nargs='+',choices=('MOT17','MOT20'),default=['MOT17','MOT20'])
    p.add_argument('--only',nargs='*',default=None,help='Optional sequence folder names for a smaller full-sequence run')
    p.add_argument('--include-selective',action='store_true',help='Add lightweight V4.3 selective Re-ID as THIRD arm')
    p.add_argument('--out',type=Path,default=None,help='New output location (default timestamped project output)')
    p.add_argument('--quick-frames',type=int,default=0,help='Smoke test N frames; do NOT compute HOTA etc')
    p.add_argument('--skip-eval',action='store_true',help='Only produce MOT files and timing/feature counts')
    p.add_argument('--video',action='store_true',help='Record annotated videos (FPS includes encoder overhead)')
    p.add_argument('--conf',type=float,default=.40)
    p.add_argument('--detector-floor',type=float,default=.15)
    p.add_argument('--imgsz',type=int,default=640)
    p.add_argument('--device',default='cuda:0')
    a=p.parse_args()
    if a.quick_frames<0:p.error('--quick-frames must be >=0')
    if a.quick_frames and not a.skip_eval:
        print('[INFO] Quick-frame runs cannot yield valid full-sequence MOT metrics; forcing --skip-eval.')
        a.skip_eval=True
    if a.video:print('[WARN] Video encoding is enabled; FPS includes VideoWriter. Omit --video for fair inference FPS.')
    if not (0<a.detector_floor<=a.conf<1):p.error('Need 0 < detector-floor <= conf < 1')
    root=a.root.expanduser().resolve();py=root/'.venv/bin/python'
    if not py.is_file():p.error(f'Missing project Python: {py}')
    if not (HERE/'baram_compare_worker.py').is_file():p.error('baram_compare_worker.py missing next to launcher')
    track_script=a.trackeval.expanduser().resolve()/'scripts/run_mot_challenge.py'
    if not a.skip_eval and not track_script.is_file():
        p.error(f'TrackEval not found: {track_script}\nInstall: git clone --depth 1 https://github.com/JonathonLuiten/TrackEval.git ~/Downloads/TrackEval')
    selected=collect_sequences(a.dataset_root.expanduser().resolve())
    for ds in a.datasets:
        found=selected[ds]
        if a.only is not None:found=[x for x in found if x.name in a.only]
        if not found:p.error(f'No GT-labeled {ds} training sequences under {a.dataset_root}; use --dataset-root path or --datasets MOT17')
        selected[ds]=found
    selected={ds:selected[ds] for ds in a.datasets}
    for ds, seqs in selected.items():
        print(f'[DATASET] {ds}: {len(seqs)} unique sequences:',', '.join(s.name for s in seqs),flush=True)
        if ds=='MOT17' and any(not x.name.endswith('-FRCNN') for x in seqs):
            print('[NOTE] At least one MOT17 sequence uses SDP/DPM fallback (only one detector variant per scene).')
    modes=['baseline','v43_full']+(['v43_selective'] if a.include_selective else [])
    out=(a.out.expanduser().resolve() if a.out else (root/'outputs'/'BARAM_YOLO26X_STRONGSORT_AB'/datetime.now().strftime('%Y%m%d_%H%M%S')).resolve())
    if out.exists() and any(out.iterdir()):p.error(f'Output directory already contains files: {out}. Use a new --out path.')
    out.mkdir(parents=True,exist_ok=True)
    manifest={'created_at':datetime.now().isoformat(),'root':str(root),'selected':{ds:[str(p) for p in paths] for ds,paths in selected.items()},
              'modes':modes,'flags':vars(a),'worker_sha256':sha256(HERE/'baram_compare_worker.py'),
              'device':a.device,'base_weights':str(root/'yolo26x.pt'),
              'ground_truth_split':'train','metric_engine':'TrackEval (HOTA, CLEAR, Identity)',
              'shared_detector_confidence_floor':a.detector_floor,'normal_tracking_confidence':a.conf}
    manifest['flags']={k:str(v) if isinstance(v,Path) else v for k,v in manifest['flags'].items()}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')
    gtroot=out/'gt'; trackerroot=out/'trackers'; resultroot=out/'metric_reports'
    env=os.environ.copy(); env['PYTHONPATH']=os.pathsep.join([str(root/'third_party/fast-reid'),str(root),env.get('PYTHONPATH','')]);env['PYTHONUNBUFFERED']='1'
    env['OMP_NUM_THREADS']=env.get('OMP_NUM_THREADS','4');env['MKL_NUM_THREADS']=env.get('MKL_NUM_THREADS','4')
    summaries={};score_summaries={};score_details={}
    runlog=out/'all_runs.csv'
    with runlog.open('w',encoding='utf-8-sig',newline='') as r:
        run_writer=csv.DictWriter(r,fieldnames=['dataset','sequence','method','frames','seconds','fps','reid_features','detections','vram_peak_mib','predictions','summary'])
        run_writer.writeheader()
        for ds,seqs in selected.items():
            for seq in seqs:
                gtlink=gtroot/(ds+'-train')/seq.name
                gtlink.parent.mkdir(parents=True,exist_ok=True)
                if not gtlink.exists():gtlink.symlink_to(seq,target_is_directory=True)
                for mode in modes:
                    outdir=out/'runs'/ds/seq.name/mode
                    outdir.mkdir(parents=True,exist_ok=True)
                    cmd=[str(py),str(HERE/'baram_compare_worker.py'),
                         '--root',str(root),'--mot-seq',str(seq),'--mode',MODES[mode],
                         '--mot-output','--no-display','--max-frames',str(a.quick_frames),
                         '--imgsz',str(a.imgsz),'--conf',str(a.conf),'--detector-floor',str(a.detector_floor),
                         '--device',a.device,'--out',str(outdir)]
                    if a.video:cmd.append('--record-annotated')
                    else:cmd.append('--no-render')
                    code,output,peak=run_with_log(cmd,outdir/'terminal.log',HERE,env,sample_vram=True)
                    if code!=0 or output is None or not (output/'summary.json').is_file():
                        raise RuntimeError(f'{ds}/{seq.name}/{mode} FAILED (exit={code}). See {outdir}/terminal.log')
                    summary=json.loads((output/'summary.json').read_text(encoding='utf-8'))
                    if summary['frames_processed'] != (a.quick_frames or len(list((seq/'img1').iterdir()))):
                        raise RuntimeError(f'Frame count mismatch for {seq.name}: processed={summary["frames_processed"]}')
                    predictions=output/'mot_results.txt'
                    if not predictions.exists():raise RuntimeError(f'Missing prediction file: {predictions}')
                    dest=trackerroot/(ds+'-train')/mode/'data'/(seq.name+'.txt')
                    dest.parent.mkdir(parents=True,exist_ok=True)
                    shutil.copyfile(predictions,dest)
                    summary['_vram_peak_mib']=peak
                    summaries[ds,seq.name,mode]=summary
                    per_frame=summary['frames_processed']/max(1e-9,summary['wall_seconds'])
                    run_writer.writerow({'dataset':ds,'sequence':seq.name,'method':mode,
                        'frames':summary['frames_processed'],'seconds':summary['wall_seconds'],
                        'fps':round(per_frame,3),'reid_features':summary['reid_features_extracted'],
                        'detections':summary['total_person_detections'],'vram_peak_mib':peak,
                        'predictions':str(dest),'summary':str(output/'summary.json')})
                    r.flush()
                    print(f'[AB RUN OK] {ds} {seq.name} {mode}: FPS={per_frame:.2f} | '
                          f'ReID={summary["reid_features_extracted"]} | VRAM={peak} MiB')
    if not a.skip_eval:
        for ds,seqs in selected.items():
            seqmap=out/(ds+'_seqmap.txt')
            seqmap.write_text('name\n'+'\n'.join(s.name for s in seqs)+'\n',encoding='utf-8')
            for mode in modes:
                metric_out=resultroot/ds/mode
                metric_out.mkdir(parents=True,exist_ok=True)
                cmd=[str(py),str(HERE/'baram_trackeval_compat.py'),str(track_script),
                     '--GT_FOLDER',str(gtroot),'--TRACKERS_FOLDER',str(trackerroot),
                     '--OUTPUT_FOLDER',str(metric_out),
                     '--BENCHMARK',ds,'--SPLIT_TO_EVAL','train',
                     '--TRACKERS_TO_EVAL',mode,'--SEQMAP_FILE',str(seqmap),
                     '--METRICS','HOTA','CLEAR','Identity',
                     '--USE_PARALLEL','False','--PRINT_CONFIG','False','--PRINT_RESULTS','True']
                code,_,_=run_with_log(cmd,metric_out/'trackeval.log',HERE,env,False)
                if code:
                    print(f'[ERROR] TrackEval failed for {ds}/{mode}; see {metric_out}/trackeval.log',file=sys.stderr)
                    continue
                try:
                    score_summaries[ds,mode]=read_metric_summary(metric_out)
                    score_details[ds,mode]=read_metric_details(metric_out)
                except Exception as e:
                    print('[WARN] TrackEval completed but could not parse summary:',e,file=sys.stderr)
    report_rows(out,summaries,modes,score_summaries,score_details,selected)
    print('\n[SAVED]',out/'comparison.csv',flush=True)
    print('[SAVED]',out/'comparison.md',flush=True)
    print('[SAVED]',out/'per_sequence.csv',flush=True)
    print('[SAVED]',out/'all_runs.csv',flush=True)
    if not a.skip_eval and len(score_summaries)!=len(modes)*len(selected):
        print('[WARN] Some TrackEval metrics missing; FPS / Re-ID / MOT txt were still saved.',flush=True)
        return 3
    return 0

if __name__=='__main__':
    try:sys.exit(main())
    except KeyboardInterrupt:print('\n[CANCELLED] Run interrupted by user.',file=sys.stderr);sys.exit(130)
    except Exception as e:print('[ERROR]',e,file=sys.stderr);sys.exit(1)
