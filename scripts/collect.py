#!/usr/bin/env python3
"""Convert official TrackEval summary outputs into one honest 3-model report."""
import argparse
import csv
from pathlib import Path

METRICS=['HOTA','MOTA','IDF1','IDSW','CLR_FN','CLR_FP','DetA','AssA']

def parse(path):
    if not path.exists():
        raise FileNotFoundError('Official TrackEval summary not found: '+str(path))
    lines=[l.split() for l in path.read_text().splitlines() if l.strip()]
    if len(lines)<2:raise RuntimeError('Invalid TrackEval summary '+str(path))
    cols=lines[0]
    row=dict(zip(cols,lines[1]))
    return {key:row.get(key,'') for key in METRICS}

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,default=Path('outputs/full_3way'))
    p.add_argument('--models',nargs='+',default=['strongsort','strongsortpp','srstrongsort'])
    a=p.parse_args()
    data=[]
    for dataset in ('MOT17','MOT20'):
        for model in a.models:
            summary=a.output/dataset/'trackers'/model/'pedestrian_summary.txt'
            line=dict(dataset=dataset,model=model);line.update(parse(summary))
            data.append(line)
    dest=a.output/'ALL_3MODELS.csv'
    with dest.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
    print('[OFFICIAL TrackEval CSV]',dest)

if __name__=='__main__':main()
