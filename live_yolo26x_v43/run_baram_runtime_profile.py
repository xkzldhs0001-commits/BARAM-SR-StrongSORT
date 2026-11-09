#!/usr/bin/env python3
"""Visual Studio Code runner for archived or fresh StrongSORT v BARAM V4.3 timing."""
from __future__ import annotations
import argparse
from datetime import datetime
import hashlib
import os
from pathlib import Path
import subprocess
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from baram_profile_analysis import analyze,locate_archive,locate_new_run

DEFAULT_ROOT=Path.home()/'Downloads'/'BARAM_SR_StrongSORT'
DEFAULT_ARCHIVE=DEFAULT_ROOT/'outputs/BARAM_YOLO26X_STRONGSORT_AB/20261106_173209'
MODES={'baseline':'strongsort_osnet','v43_selective':'sr_osnet_selective'}


def sha256(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''):h.update(chunk)
    return h.hexdigest()


def parse():
    p=argparse.ArgumentParser(description='BARAM stage-wise latency and crowd-density profiler')
    exclusive=p.add_mutually_exclusive_group(required=True)
    exclusive.add_argument('--from-ab',type=Path,nargs='?',const=DEFAULT_ARCHIVE,
         help='Analyze existing A/B frames.csv; no GPU use. If path omitted, use original 20261106_173209 run.')
    exclusive.add_argument('--measure',action='store_true',help='Re-run same MOT scene(s) for new measurements')
    p.add_argument('--root',type=Path,default=DEFAULT_ROOT)
    p.add_argument('--only',nargs='+',help='Optional names, e.g. MOT17-10-FRCNN MOT20-02; archive: default ALL')
    p.add_argument('--max-frames',type=int,default=600,help='Only --measure; default 600 per seq, 0=all')
    p.add_argument('--repeats',type=int,default=1,help='Only --measure; 2 adds reversed run order per sequence')
    p.add_argument('--skip-frames',type=int,default=30,
         help='Exclude first 30 frames of each run to reduce model startup effects (default 30)')
    p.add_argument('--output',type=Path,default=None,help='New output report directory')
    p.add_argument('--conf',type=float,default=.40)
    p.add_argument('--detector-floor',type=float,default=.15)
    p.add_argument('--imgsz',type=int,default=640)
    p.add_argument('--device',default='cuda:0')
    a=p.parse_args()
    if a.max_frames<0:p.error('--max-frames must be >= 0')
    if a.repeats<1 or a.repeats>5:p.error('--repeats must be 1..5')
    if a.skip_frames<0:p.error('--skip-frames must be >= 0')
    if not (0<a.detector_floor<=a.conf<1):p.error('Need 0 < detector-floor <= conf < 1')
    return a


def discover(root, names):
    candidates=[]
    for name in names:
        ds=name.split('-',1)[0]
        seq=root/'datasets'/ds/'train'/name
        if not (seq/'seqinfo.ini').is_file() or not (seq/'img1').is_dir():
            raise FileNotFoundError(f'Sequence not found: {seq} (need img1/ and seqinfo.ini)')
        candidates.append((ds,name,seq))
    return candidates


def main():
    args=parse()
    timestamp=datetime.now().strftime('%Y%m%d_%H%M%S')
    root=args.root.expanduser().resolve()
    default_out=root/'outputs/BARAM_RUNTIME_PROFILING'/timestamp
    out=args.output.expanduser().resolve() if args.output else default_out
    if out.exists() and any(out.iterdir()):
        raise RuntimeError(f'Output must be a fresh/empty directory: {out}')
    out.mkdir(parents=True,exist_ok=True)
    if args.from_ab is not None:
        arch=args.from_ab.expanduser().resolve()
        print('[REUSE] Existing saved A/B per-frame timings:',arch,flush=True)
        print('[INFO] No CUDA, YOLO26x, OSNet, or TrackEval rerun is needed.',flush=True)
        files=locate_archive(arch,only=set(args.only) if args.only else None)
        analyze(files,out, warmup=args.skip_frames,source=f'archived_A_B:{arch}')
        return

    py=root/'.venv/bin/python'
    worker=HERE/'baram_profile_worker_v43.py'
    if not py.is_file():raise FileNotFoundError(f'Venv Python missing: {py}')
    if not worker.is_file():raise FileNotFoundError(f'Worker missing: {worker}')
    seqs=discover(root,args.only if args.only else ['MOT17-10-FRCNN','MOT20-02'])
    print('[WORKER] V4.3 benchmark snapshot:',worker,flush=True)
    print('[WORKER SHA256]',sha256(worker),flush=True)
    print('[INFO] --no-display --no-render; no video encodes; synchronized CUDA-stage timing.',flush=True)
    print('[NOTE] Profiling and CUDA synchronization have overhead; use this for bottleneck diagnosis, not headline FPS.',flush=True)
    env=os.environ.copy()
    env['PYTHONPATH']=os.pathsep.join((str(root/'third_party/fast-reid'),str(root),str(HERE),
                                      env.get('PYTHONPATH','')))
    env.setdefault('OMP_NUM_THREADS','4')
    env.setdefault('MKL_NUM_THREADS','4')
    for ds,name,seq in seqs:
        for rep in range(args.repeats):
            order=list(MODES)
            if rep%2:order.reverse()
            for method in order:
                folder=out/'runs'/ds/name/method
                folder.mkdir(parents=True,exist_ok=True)
                cmd=[str(py),str(worker),'--root',str(root),'--mot-seq',str(seq),
                     '--mode',MODES[method], '--no-display', '--no-render',
                     '--max-frames',str(args.max_frames), '--imgsz',str(args.imgsz),
                     '--conf',str(args.conf),'--detector-floor',str(args.detector_floor),
                     '--device',args.device,'--out',str(folder)]
                print('\n[RUN]',ds,name,method,'repeat',rep+1,'/',args.repeats,flush=True)
                print('[COMMAND]',' '.join(cmd),flush=True)
                log=folder/f'profile_repeat{rep+1}.log'
                with log.open('w',encoding='utf-8') as fout:
                    process=subprocess.Popen(cmd,cwd=str(HERE),env=env,
                                stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                                text=True,errors='replace',bufsize=1)
                    try:
                        for line in process.stdout:
                            print(line,end='',flush=True)
                            fout.write(line)
                        code=process.wait()
                    except KeyboardInterrupt:
                        process.terminate()
                        try:process.wait(timeout=5)
                        except subprocess.TimeoutExpired:process.kill();process.wait()
                        raise
                if code:
                    raise RuntimeError(f'Worker failed for {ds} {method} (exit={code}). See {log}')
    # Repeat 2 should not overwrite earlier outputs. For aggregation of repeated
    # runs, copy series into distinct logical sequence keys (seq-repeatN).
    files={}
    for f in (out/'runs').rglob('frames.csv'):
        rel=f.relative_to(out/'runs').parts
        if len(rel)!=5:continue
        ds, seq, method, sub, _=rel
        if method not in MODES:continue
        # recover repeat index by ordering same-method child directories; matches
        # worker output names with seconds. More robust: map timestamp-sorted dirs.
        siblings=sorted((out/'runs'/ds/seq/method).glob('*/frames.csv'))
        repeat_i=siblings.index(f)+1
        logical=seq if args.repeats==1 else f'{seq}_repeat{repeat_i}'
        files[(ds,logical,method)]=f
    expect=2*len(seqs)*args.repeats
    if len(files)!=expect:
        raise RuntimeError(f'Expected {expect} run frame.csv files but found {len(files)}')
    analyze(files,out/'report',warmup=args.skip_frames,source=f'new_inference:{out}')
    print('[DONE] Fresh GPU profiling completed.',flush=True)

if __name__=='__main__':
    try:main()
    except (RuntimeError,ValueError,FileNotFoundError) as exc:
        print('[ERROR]',exc,file=sys.stderr,flush=True)
        sys.exit(1)
