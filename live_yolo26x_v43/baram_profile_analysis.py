#!/usr/bin/env python3
"""Analyze BARAM frame timings; no model execution, no fabricated values.

Columns are sourced from frame-level timers in baram_compare_worker.py. The
'Other' category is a residual, NOT an independently instrumented kernel.
"""
from __future__ import annotations
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

STAGES = [
    ('YOLO26x Detection', 'detector'),
    ('OSNet Re-ID', 'reid'),
    ('Prediction & Gating', 'gating'),
    ('Data Association', 'association'),
    ('ID Management', 'id_management'),
    ('I/O & Other', 'other'),
]
MODES = {'baseline':'StrongSORT', 'v43_selective':'SR-StrongSORT (OURS)'}
BINS = ('0', '1–5', '6–15', '16+')


def crowd_bin(n):
    if n == 0: return '0'
    if n <= 5: return '1–5'
    if n <= 15: return '6–15'
    return '16+'


def percentile(values, p):
    if not values: return None
    a = sorted(values)
    k = (len(a)-1)*p/100
    lo, hi = math.floor(k), math.ceil(k)
    return a[lo]*(hi-k) + a[hi]*(k-lo) if lo != hi else a[lo]


def load_frames(path:Path, warmup:int):
    required = ['wall_ms', 'detector_ms', 'reid_ms', 'predict_select_ms',
                'ecc_ms', 'association_ms', 'id_memory_ms', 'detections',
                'reid_features']
    rows = []
    with path.open('r', newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        missing=[c for c in required if c not in (reader.fieldnames or [])]
        if missing: raise ValueError(f'Missing columns in {path}: {missing}')
        count=0
        for item in reader:
            count+=1
            if count<=warmup:continue
            try:
                vals={k:float(item[k]) for k in required}
            except (ValueError, TypeError, KeyError) as exc:
                raise ValueError(f'Bad numeric frame {count} in {path}: {exc}') from exc
            if not all(math.isfinite(v) and v>=0 for v in vals.values()):
                raise ValueError(f'Negative/nonfinite metric at {path}, row {count}')
            det=vals['detector_ms']
            reid=vals['reid_ms']
            gating=vals['ecc_ms']+vals['predict_select_ms']
            assoc=vals['association_ms']
            memory=vals['id_memory_ms']
            total=det+reid+gating+assoc+memory
            wall=vals['wall_ms']
            if total>wall+2.0:
                raise ValueError(f'Stage timing sum exceeds frame time by >2ms at {path}:{count}; check instrumentation')
            other=max(0.0, wall-total)
            # Archived worker recorded 'detections' incl. low-confidence rescue
            # candidates. New profile worker writes high_conf_detections separately.
            high=item.get('high_conf_detections','')
            high_available=(high is not None and str(high).strip()!='')
            n=int(float(high)) if high_available else int(vals['detections'])
            if n<0:raise ValueError(f'Bad detection count in {path}:{count}')
            rows.append({'wall_ms':wall,'detector':det,'reid':reid,'gating':gating,
                         'association':assoc,'id_management':memory,'other':other,
                         'crowd':crowd_bin(n),'detections':n,
                         'reid_features':int(vals['reid_features']),
                         'high_conf_available':high_available})
    if not rows:raise ValueError(f'No frames after warmup={warmup}: {path}')
    return rows, count


def locate_archive(archive, only=None):
    root=Path(archive).expanduser().resolve()
    folder=root/'runs'
    if not folder.is_dir():raise FileNotFoundError(f'Archive does not contain runs/: {folder}')
    found={}
    for csv_file in folder.rglob('frames.csv'):
        # runs/DATASET/SEQ/METHOD/SEQ_MODE_TIME/frames.csv
        try:
            rel=csv_file.relative_to(folder).parts
        except ValueError:continue
        if len(rel)!=5:continue
        ds,seq,method,subdir,_=rel
        if ds not in ('MOT17','MOT20') or method not in MODES:continue
        if only and seq not in only:continue
        key=(ds,seq,method)
        if key in found:raise ValueError(f'Duplicate run {key} in archive')
        found[key]=csv_file
    if not found:raise FileNotFoundError(f'No baseline/V4.3 Selective frames.csv found under {folder}')
    return found


def locate_new_run(root,only=None):
    found={}
    for file in (Path(root)/'runs').rglob('frames.csv'):
        rel=file.relative_to(Path(root)/'runs').parts
        if len(rel)!=5:continue
        ds, seq, method, subdir, _=rel
        if method not in MODES or ds not in ('MOT17','MOT20'):continue
        if only and seq not in only:continue
        key=(ds,seq,method)
        if key in found:raise ValueError(f'Duplicate run {key}')
        found[key]=file
    return found


def analyze(files, output, warmup=30, source='archive'):
    output=Path(output)
    output.mkdir(parents=True,exist_ok=True)
    groups=defaultdict(list)
    byseq=defaultdict(list)
    metadata=[]
    proxies=set()
    # Only accept paired sequences and equal number of processed frames.
    sequences=sorted({(ds,seq) for ds,seq,method in files})
    for ds,seq in sequences:
        pair=[(ds,seq,m) for m in MODES]
        if any(k not in files for k in pair):
            raise RuntimeError(f'Missing baseline or v43_selective for {ds}/{seq} — cannot compare')
        lengths={}
        for key in pair:
            path=files[key]
            rows,raw_count=load_frames(path,warmup)
            lengths[key[2]]=raw_count
            groups[(ds,key[2])].extend(rows)
            byseq[(ds,seq,key[2])].extend(rows)
            proxies.add(not rows[0]['high_conf_available'])
            metadata.append({'dataset':ds,'sequence':seq,'method':key[2],
                             'frames_original':raw_count,'frames_analyzed':len(rows),
                             'frame_csv':str(path),'crowd_source':
                             'high_conf_detections' if rows[0]['high_conf_available'] else 'detections_including_rescue_candidates'})
        if lengths['baseline'] != lengths['v43_selective']:
            raise RuntimeError(f'Mismatched frame counts for {ds}/{seq}: {lengths}')
    if not groups:raise RuntimeError('No dataset groups to analyze')

    # Stage mean rows: one per dataset/method/stage; the weighted aggregation uses
    # *all* frames, not arithmetic averaging of per-sequence means.
    stage_rows=[]; crowd_rows=[]; summary_rows=[]; byseq_rows=[]
    for (ds,method), rows in sorted(groups.items()):
        total_wall=sum(r['wall_ms'] for r in rows)
        duration=total_wall/1000.
        for label,key in STAGES:
            mean=sum(r[key] for r in rows)/len(rows)
            stage_rows.append({'dataset':ds,'method':method,'stage':label,
                               'mean_ms_per_frame':round(mean,4),
                               'share_of_total_pct':round(100*sum(r[key] for r in rows)/total_wall,3)})
        summary_rows.append({'dataset':ds,'method':method,'frames':len(rows),
            'mean_wall_ms':round(total_wall/len(rows),4),
            'p50_wall_ms':round(percentile([r['wall_ms'] for r in rows],50),4),
            'p95_wall_ms':round(percentile([r['wall_ms'] for r in rows],95),4),
            'profiled_fps':round(len(rows)/duration,3),
            'reid_crops':sum(r['reid_features'] for r in rows),
            'mean_detected_boxes':round(sum(r['detections'] for r in rows)/len(rows),2)})
        for band in BINS:
            subset=[r for r in rows if r['crowd']==band]
            if subset:
                vals=[r['wall_ms'] for r in subset]
                crowd_rows.append({'dataset':ds,'method':method,'density_bin':band,
                    'frames':len(subset),'mean_wall_ms':round(statistics.mean(vals),4),
                    'p95_wall_ms':round(percentile(vals,95),4)})
    for (ds,seq,method), rows in sorted(byseq.items()):
        vals=[r['wall_ms'] for r in rows]
        byseq_rows.append({'dataset':ds,'sequence':seq,'method':method,'frames':len(rows),
                           'mean_wall_ms':round(statistics.mean(vals),4),
                           'p95_wall_ms':round(percentile(vals,95),4),
                           'reid_crops':sum(r['reid_features'] for r in rows)})

    def savecsv(name, data):
        with (output/name).open('w',newline='',encoding='utf-8-sig') as out:
            w=csv.DictWriter(out,fieldnames=list(data[0]))
            w.writeheader();w.writerows(data)
    savecsv('stage_latency.csv',stage_rows)
    savecsv('crowd_latency.csv',crowd_rows)
    savecsv('runtime_summary.csv',summary_rows)
    savecsv('per_sequence_latency.csv',byseq_rows)
    savecsv('input_provenance.csv',metadata)
    (output/'measurement_protocol.json').write_text(json.dumps({
        'source':source,
        'warmup_excluded_per_sequence':warmup,
        'stage_definitions':dict((label,key) for label,key in STAGES),
        'stages_aggregate_from':['detector_ms','reid_ms','predict_select_ms','ecc_ms',
                                 'association_ms','id_memory_ms','wall_ms'],
        'other_ms':'wall_ms minus the five named stage groups; includes image reading and uninstrumented overhead',
        'crowd_definition':'high_conf_detections (current worker) else detections including rescue (archive)',
        'crowd_archived_proxy_used':True in proxies,
        'fps_definition':'number of analyzed frames / sum(frame wall_ms); profile timing includes synchronization and instrumentation overhead',
        'caveat':'This diagnoses time distribution only, NOT HOTA/IDF1/IDSW. Some stage boundaries contain Python+GPU sync cost. No cached detector inputs. Not an isolated causal measurement.',
        'runs':metadata
    },ensure_ascii=False,indent=2),encoding='utf-8')

    datasets=sorted({ds for ds,_ in groups})
    lines=['# Runtime Bottleneck & Efficiency Analysis','',
           f'**Source:** {source}  |  **Warmup frames excluded per sequence:** {warmup}',
           '', 'This report analyzes *measured* per-frame timings; no values have been fabricated.',
           'FPS here reflects synchronized profiled-frame latency and may differ from earlier unprofiled benchmark FPS.',
           '', '| Dataset | Method | Frames | Mean latency (ms) | P95 latency (ms) | Profiled FPS | OSNet crops |',
           '|---|---|---:|---:|---:|---:|---:|']
    for r in summary_rows:
        lines.append(f"| {r['dataset']} | {MODES[r['method']]} | {r['frames']:,} | {r['mean_wall_ms']:.2f} | {r['p95_wall_ms']:.2f} | {r['profiled_fps']:.2f} | {r['reid_crops']:,} |")
    lines+=['','## Stage-wise mean latency (ms/frame)','',
             '| Dataset | Method | ' + ' | '.join(x[0] for x in STAGES) + ' |',
             '|---|---|' + '|'.join(['---:']*len(STAGES)) + '|']
    lookup={(r['dataset'],r['method'],r['stage']):r['mean_ms_per_frame'] for r in stage_rows}
    for ds in datasets:
        for m in MODES:
            lines.append('| '+ ' | '.join([ds,MODES[m]] +
              [f'{lookup[(ds,m,label)]:.2f}' for label,_ in STAGES])+' |')
    lines+=['','## How to interpret','',
      '- YOLO26x Detection includes detector inference plus detection-score filtering.',
      '- OSNet Re-ID includes image cropping and feature extraction; a batch with no selected crops may be near zero.',
      '- Prediction & Gating includes tracker prediction and selective selection/occlusion candidate processing.',
      '- Data Association includes assignment and tracker updates.',
      '- ID Management includes stable display-ID recovery; absent in baseline.',
      '- I/O & Other is a computed residual, not a directly timed model submodule.',
      '- Crowd groups are per-frame detected box counts, not ground-truth person counts.',
      '- Historical frames.csv lacks high_conf_detections: crowd counts include low-confidence rescue candidates.',
      '- Compare differences descriptively; detector output variability and run-order/GPU contention can affect timing.',
      '- Do not label the largest measured stage as the proven *cause* of lower FPS without controlled ablation.',
      '']
    (output/'PROFILE_REPORT.md').write_text('\n'.join(lines),encoding='utf-8')

    # LaTeX values for the earlier Overleaf 5-stage pgfplots poster template.
    # Collapse Prediction & Gating + Data Association into the existing
    # poster row 'Gating + Matching'; no measured values are fabricated.
    def ceil_axis(val):
        return max(10,int(math.ceil(val*1.2/10.)*10))
    for ds in datasets:
        plotvals={}
        for method in MODES:
            x=lambda label:lookup[(ds,method,label)]
            plotvals[method]=[
                x('YOLO26x Detection'),x('OSNet Re-ID'),
                x('Prediction & Gating')+x('Data Association'),
                x('ID Management'),x('I/O & Other')]
        maxstage=max(v for a in plotvals.values() for v in a)
        crowd_by={(r['method'],r['density_bin']):r['mean_wall_ms']
                  for r in crowd_rows if r['dataset']==ds}
        cm=max((v for v in crowd_by.values()),default=0)
        def fmt_stage(method):
            return ' '.join(f'({v:.3f},{i})' for i,v in zip(range(5,0,-1),plotvals[method]))
        def fmt_crowd(method):
            return ' '.join(f'({x},{crowd_by[(method,b)]:.3f})' for b,x in
                            [('1–5',3),('6–15',10),('16+',20)] if (method,b) in crowd_by)
        latex = ['% Paste these macros in place of the placeholder definitions',
                 '% in BARAM_Runtime_Bottleneck_Overleaf.tex.',
                 '% Stage 3 is Gating + Matching = Prediction & Gating + Data Association.',
                 '% Crowd = detected boxes, NOT verified unique human IDs.',
                 '% Archived data include low-confidence rescue candidates in density counts.',
                 r'\profilingreadytrue',
                 r'\def\baseStage{' + fmt_stage('baseline') + '}',
                 r'\def\oursStage{' + fmt_stage('v43_selective') + '}',
                 r'\def\stageAxisMax{' + str(ceil_axis(maxstage)) + '}',
                 r'\def\baseCrowd{' + fmt_crowd('baseline') + '}',
                 r'\def\oursCrowd{' + fmt_crowd('v43_selective') + '}',
                 r'\def\crowdAxisMax{' + str(ceil_axis(cm)) + '}', '']
        (output/f'overleaf_values_{ds}.tex').write_text('\n'.join(latex),encoding='utf-8')

    plot_warning=None
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        baseline_color='#B3C4DA'; ours_color='#355F9A'
        colors={'baseline':baseline_color,'v43_selective':ours_color}
        for ds in datasets:
            # Stage: side-by-side vertical bars, simple and printable.
            labels=[s[0] for s in STAGES]
            x=list(range(len(labels)))
            fig,ax=plt.subplots(figsize=(12,5.0))
            for j,m in enumerate(MODES):
                heights=[lookup[(ds,m,label)] for label in labels]
                offs=[xx+(-.19 if j==0 else .19) for xx in x]
                ax.bar(offs,heights,width=.36,color=colors[m],label=MODES[m])
            ax.set_xticks(x,labels,rotation=16,ha='right')
            ax.set_ylabel('Mean latency (ms/frame)')
            ax.set_title(f'{ds}  |  Stage-wise Latency Breakdown')
            ax.legend(frameon=False)
            ax.grid(axis='y',alpha=.15)
            fig.tight_layout()
            fig.savefig(output/f'{ds}_stage_latency.png',dpi=200)
            plt.close(fig)
            # Density latency: plot only bins with actual samples.
            fig,ax=plt.subplots(figsize=(10,5.0))
            for m in MODES:
                vals=[r for r in crowd_rows if r['dataset']==ds and r['method']==m]
                bybin={r['density_bin']:r for r in vals}
                inds=[i for i,b in enumerate(BINS) if b in bybin]
                if inds:
                    ax.plot(inds,[bybin[BINS[i]]['mean_wall_ms'] for i in inds],
                            marker='o',label=MODES[m],color=colors[m],linewidth=2)
                    for i in inds:
                        rr=bybin[BINS[i]]
                        ax.annotate('n='+str(rr['frames']),
                          (i,rr['mean_wall_ms']),xytext=(4,7),textcoords='offset points',fontsize=8)
            ax.set_xticks(range(len(BINS)),BINS)
            ax.set_xlabel('Detected boxes per frame (density proxy)')
            ax.set_ylabel('Mean end-to-end latency (ms/frame)')
            ax.set_title(f'{ds}  |  Crowd Density vs Processing Latency')
            ax.legend(frameon=False)
            ax.grid(alpha=.15)
            fig.tight_layout()
            fig.savefig(output/f'{ds}_crowd_latency.png',dpi=200)
            plt.close(fig)
    except ImportError:
        plot_warning='matplotlib is not installed; CSV and Markdown reports were still generated.'
        print('[WARN]',plot_warning,flush=True)
    print('[PROFILE COMPLETE]',output,flush=True)
    print('[REPORT]',output/'PROFILE_REPORT.md',flush=True)
    if plot_warning:print('[PLOTS]',plot_warning,flush=True)
    else:
        for ds in datasets:
            print('[PLOTS]',output/f'{ds}_stage_latency.png',flush=True)
            print('[PLOTS]',output/f'{ds}_crowd_latency.png',flush=True)
    return summary_rows,stage_rows,crowd_rows
