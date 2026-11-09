#!/usr/bin/env python3
"""Official TrackEval runner; full MOT17/MOT20 train only, no fabricated GT metrics."""
import argparse
import csv
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import subprocess
import sys


def main():
 p=argparse.ArgumentParser()
 p.add_argument('--project',type=Path,default=Path(__file__).resolve().parents[1])
 p.add_argument('--output',type=Path,default=Path('outputs/full_3way'))
 p.add_argument('--benchmark',choices=['MOT17','MOT20','both'],default='both')
 p.add_argument('--models',nargs='+',default=['strongsort','srstrongsort'])
 a=p.parse_args();project=a.project.resolve()
 root=a.output if a.output.is_absolute() else project/a.output
 runner=project/'third_party'/'TrackEval'/'scripts'/'run_mot_challenge.py'
 if not runner.exists():raise FileNotFoundError('TrackEval missing, run scripts/setup_ubuntu.sh')
 from baram.data import dataset_sequences
 for ds in (['MOT17','MOT20'] if a.benchmark=='both' else [a.benchmark]):
  seqs=dataset_sequences(project/'datasets',ds)
  # Validate FULL sequence length from seqinfo.ini so partial runs aren't called official.
  import configparser
  for seq in seqs:
   info=project/'datasets'/ds/'train'/seq/'seqinfo.ini'
   ini=configparser.ConfigParser();ini.read(info)
   length=ini.getint('Sequence','seqLength')
   for model in a.models:
    trackfile=root/ds/'trackers'/model/'data'/(seq+'.txt')    stats=root/ds/'stats'/model/(seq+'.json')
    if not trackfile.exists():raise FileNotFoundError(trackfile)
    if model!='strongsortpp':
     if not stats.exists():raise FileNotFoundError(stats)
     import json
     if json.loads(stats.read_text())['frames']!=length:
      raise RuntimeError(f'Incomplete sequence: {seq} {model}. Must run full sequence before GT evaluation.')
  seqmap=root/ds/'seqmap.txt';seqmap.parent.mkdir(parents=True,exist_ok=True)
  seqmap.write_text('name\n'+'\n'.join(seqs)+'\n')
  cmd=[sys.executable,str(runner),'--BENCHMARK',ds,'--SPLIT_TO_EVAL','train',
       '--GT_FOLDER',str(project/'datasets'/ds/'train'), '--SKIP_SPLIT_FOL','True','--SEQMAP_FILE',str(seqmap),
       '--TRACKERS_FOLDER',str(root/ds/'trackers'),'--TRACKERS_TO_EVAL',*a.models,
       '--TRACKER_SUB_FOLDER','data','--METRICS','HOTA','CLEAR','Identity',
       '--USE_PARALLEL','False','--PRINT_CONFIG','False','--DO_PREPROC','True']
  print('[OFFICIAL TRACK EVAL]',ds,flush=True)
  log=root/ds/'trackeval.log'
  with log.open('w') as f:
   p=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,cwd=project)
  if p.returncode:raise RuntimeError(f'Official TrackEval failed, see {log}')
  print('TrackEval finished: ',log)

if __name__=='__main__':main()
