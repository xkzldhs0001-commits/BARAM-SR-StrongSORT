#!/usr/bin/env python3
"""Fail-closed audit + TrackEval scoring for saved BARAM 2026 MOT17/MOT20 predictions.

Does NOT run YOLO or modify GT/predictions or prior comparison.csv.
Exit != 0 if any test/evaluation/consistency requirement fails.
"""
from __future__ import annotations
import argparse
import collections
import configparser
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

EXPECTED = {'MOT17': {'02', '04', '05', '09', '10', '11', '13'},
            'MOT20': {'01', '02', '03', '05'}}
MODELS = ('baseline', 'v43_full', 'v43_selective')
METRICS = ('HOTA', 'IDF1', 'MOTA', 'IDSW')
RE_MOT = re.compile(r'^MOT(17|20)-(\d\d)(?:-(?:FRCNN|DPM|SDP))?$')

class AuditFailure(Exception):
    pass


def require(condition, msg):
    if not condition:
        raise AuditFailure(msg)


def digest(file: Path):
    h = hashlib.sha256()
    with file.open('rb') as f:
        for data in iter(lambda: f.read(1024 * 1024), b''):
            h.update(data)
    return h.hexdigest()


def load_json(path: Path):
    require(path.is_file(), f'Missing JSON: {path}')
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except Exception as exc:
        raise AuditFailure(f'Cannot parse {path}: {exc}') from exc


def csv_rows(path: Path):
    require(path.is_file(), f'Missing CSV: {path}')
    with path.open('r', newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        return list(reader), reader.fieldnames or []


def seq_config(path: Path):
    require(path.is_file(), f'Missing seqinfo: {path}')
    parser = configparser.ConfigParser()
    parser.read(path, encoding='utf-8-sig')
    require(parser.has_section('Sequence'), f'Missing [Sequence] in {path}')
    s = parser['Sequence']
    result = {'length': int(s['seqLength']), 'width': int(s['imWidth']),
              'height': int(s['imHeight']), 'ext': s.get('imExt', '.jpg')}
    require(all(result[k] > 0 for k in ('length', 'width', 'height')),
            f'Bad sequence dimensions or length in {path}')
    return result


def ensure_images(seq: Path, cfg: dict):
    imgs = sorted((seq/'img1').glob('*' + cfg['ext']))
    require(len(imgs) == cfg['length'], f'{seq.name}: expected {cfg["length"]} images; found {len(imgs)}')
    for i, p in enumerate(imgs, start=1):
        require(p.stem.isdigit() and int(p.stem) == i,
                f'{seq.name}: image numbering unexpected around {p.name}')
    return imgs


def parse_mot_prediction(path: Path, frames: int, width: int, height: int):
    require(path.is_file(), f'Missing prediction: {path}')
    require(path.stat().st_size > 0, f'Empty prediction: {path}')
    n = 0; frames_seen = set(); ids_by_frame = set(); last = 0
    with path.open('r', newline='', encoding='utf-8-sig') as f:
        for lineno, row in enumerate(csv.reader(f), 1):
            if not row or all(not c.strip() for c in row):
                continue
            require(len(row) == 10, f'{path.name} line {lineno}: require 10 columns, got {len(row)}')
            try:
                numbers = [float(v.strip()) for v in row]
            except ValueError as exc:
                raise AuditFailure(f'{path.name}:{lineno} non-numeric value: {exc}') from exc
            require(all(math.isfinite(x) for x in numbers), f'{path.name}:{lineno} NaN/inf')
            frame_float, id_float, x, y, w, h, conf = numbers[:7]
            require(frame_float.is_integer() and 1 <= frame_float <= frames,
                    f'{path.name}:{lineno} frame out of 1..{frames}')
            require(id_float.is_integer() and id_float >= 1,
                    f'{path.name}:{lineno} invalid ID {id_float}')
            frame, track_id = int(frame_float), int(id_float)
            require(frame >= last, f'{path.name}:{lineno} frame order is not monotonic')
            last = frame
            require((frame, track_id) not in ids_by_frame,
                    f'{path.name}:{lineno} duplicate ID {track_id} at frame {frame}')
            ids_by_frame.add((frame, track_id))
            # BARAM export is 1-based xywh, clipped to image edges.
            require(x >= 1-0.002 and y >= 1-0.002 and w > 0 and h > 0,
                    f'{path.name}:{lineno} invalid MOT 1-based bbox')
            require(x + w <= width + 1 + .02 and y + h <= height + 1 + .02,
                    f'{path.name}:{lineno} bbox exceeds frame size {width}x{height}')
            require(conf >= 0, f'{path.name}:{lineno} invalid confidence')
            frames_seen.add(frame); n += 1
    return {'predictions': n, 'frames_with_predictions': len(frames_seen), 'unique_frame_ids': len(ids_by_frame)}


def parse_mot_gt(path: Path, frames: int):
    require(path.is_file(), f'Missing ground truth: {path}')
    count = 0; ped = 0
    with path.open('r', newline='', encoding='utf-8-sig') as f:
        for lineno, row in enumerate(csv.reader(f), 1):
            if not row or all(not x.strip() for x in row):
                continue
            require(len(row) >= 9, f'GT {path}:{lineno} needs >=9 columns')
            try:
                v = [float(x.strip()) for x in row[:9]]
            except ValueError as exc:
                raise AuditFailure(f'GT {path}:{lineno} malformed: {exc}') from exc
            require(all(math.isfinite(x) for x in v), f'GT {path}:{lineno} contains NaN/Inf')
            require(v[0].is_integer() and 1 <= v[0] <= frames,
                    f'GT {path}:{lineno} invalid frame')
            require(v[4] > 0 and v[5] > 0,
                    f'GT {path}:{lineno} nonpositive bbox')
            count += 1
            if int(v[7]) == 1 and v[6] > 0: ped += 1
    require(count > 0 and ped > 0, f'No usable pedestrian ground truth: {path}')
    return {'all_gt_rows': count, 'positive_ped_gt_rows': ped}


def trackeval_metrics(folder: Path, expected_sequences):
    """Parse and cross-check official TrackEval summary/detailed outputs.

    Official TrackEval detailed.csv uses HOTA___AUC (not HOTA), with
    HOTA___AUC, MOTA and IDF1 on 0..1 scales. Summary.txt displays the
    corresponding values on the 0..100 scale. IDSW remains a count.
    See TrackEval metrics/_base_metric.py and utils.py.
    """
    summary_files = list(folder.rglob('pedestrian_summary.txt'))
    detailed_files = list(folder.rglob('pedestrian_detailed.csv'))
    require(len(summary_files) == 1 and len(detailed_files) == 1,
            f'TrackEval expected one summary and one detailed CSV in {folder}; '
            f'found {len(summary_files)} and {len(detailed_files)}')
    raw = [x.strip() for x in summary_files[0].read_text(encoding='utf-8').splitlines() if x.strip()]
    require(len(raw) == 2, f'Unexpected TrackEval summary rows: {summary_files[0]}')
    headers, values = raw[0].split(), raw[1].split()
    require(len(headers) == len(values) and len(set(headers)) == len(headers),
            f'Misaligned or duplicate TrackEval summary headers: {summary_files[0]}')
    combined = dict(zip(headers, values))
    require(all(k in combined for k in METRICS), f'Missing summary metrics: {summary_files[0]}')
    try:
        summary = {k: float(combined[k]) for k in METRICS}
    except ValueError as exc:
        raise AuditFailure(f'Non-numeric summary metric in {summary_files[0]}: {exc}') from exc
    check_metric_values(summary, str(summary_files[0]))

    details, fieldnames = csv_rows(detailed_files[0])
    # These are the actual official TrackEval detailed CSV headers.
    official_columns = {'HOTA':'HOTA___AUC', 'IDF1':'IDF1', 'MOTA':'MOTA', 'IDSW':'IDSW'}
    required_columns = ['seq', *official_columns.values()]
    require(all(fieldnames.count(k) == 1 for k in required_columns),
            f'TrackEval detailed CSV required columns missing/duplicated: '
            f'{detailed_files[0]}; columns={fieldnames}')
    expected = set(expected_sequences)
    require(len(expected) == len(expected_sequences), 'Duplicated expected sequence name')
    require(len(details) == len(expected) + 1,
            f'Expected exactly {len(expected)} detailed sequence rows + COMBINED; '
            f'found {len(details)} in {detailed_files[0]}')
    by_seq = {}
    combined_detail = None

    for row in details:
        seq = (row.get('seq') or '').strip()
        require(seq in expected or seq == 'COMBINED',
                f'Unexpected/blank detailed sequence {seq!r} in {detailed_files[0]}')
        require(seq not in by_seq and not (seq == 'COMBINED' and combined_detail is not None),
                f'Duplicate TrackEval detailed sequence {seq}')
        require(None not in row, f'Extra CSV columns in detailed sequence {seq}')
        try:
            measured = {}
            for metric, col in official_columns.items():
                val = row.get(col)
                require(val is not None and val.strip() != '',
                        f'Missing {metric} detailed value ({col}) in {seq}')
                n = float(val)
                if metric != 'IDSW':
                    # TrackEval detailed float fields / array AUC are fractions, not percentages.
                    require(math.isfinite(n) and (n <= 1.000001) and
                            (n >= -1e-9 if metric in ('HOTA', 'IDF1') else True),
                            f'Detailed raw {metric} outside expected TrackEval scale in {seq}: {n}')
                    measured[metric] = 100.0 * n
                else:
                    measured[metric] = n
        except ValueError as exc:
            raise AuditFailure(f'Non-numeric TrackEval detailed metric in {seq}: {exc}') from exc
        check_metric_values(measured, seq)
        if seq == 'COMBINED':
            combined_detail = measured
        else:
            by_seq[seq] = measured

    require(set(by_seq) == expected and combined_detail is not None,
            f'Per-sequence report mismatch: missing={expected-set(by_seq)}; '
            f'combined_present={combined_detail is not None}')

    # Official summary values are printed to 5 significant digits, so allow
    # a small rounding difference against unrounded detailed fractions.
    for k in METRICS:
        tolerance = 0.0 if k == 'IDSW' else 0.011
        require(abs(combined_detail[k] - summary[k]) <= tolerance,
                f'TrackEval summary vs COMBINED mismatch {k}: '
                f'{summary[k]} != {combined_detail[k]}')
    require(sum(x['IDSW'] for x in by_seq.values()) == summary['IDSW'],
            f'TrackEval combined IDSW != sum(per-sequence IDSW) in {folder}')
    return summary, by_seq, {'summary_sha256': digest(summary_files[0]),
                              'detailed_sha256': digest(detailed_files[0])}


def check_metric_values(v, context):
    require(all(math.isfinite(v[k]) for k in METRICS), f'{context}: NaN or infinite score')
    require(all(0 <= v[k] <= 100 for k in ('HOTA', 'IDF1')), f'{context}: HOTA/IDF1 range')
    require(v['MOTA'] <= 100.0001, f'{context}: MOTA >100')
    require(v['IDSW'] >= 0 and abs(v['IDSW'] - round(v['IDSW'])) < .001,
            f'{context}: IDSW must be nonnegative integer')


def verify(run: Path, allow_subset: bool, worker_file=None):
    manifest = load_json(run/'manifest.json')
    selected = manifest.get('selected')
    modes = manifest.get('modes')
    require(isinstance(selected, dict) and selected, 'manifest selected missing')
    require(isinstance(modes, list) and set(modes) == set(MODELS) and len(modes) == 3,
            f'Expected all three methods {MODELS}; found {modes}')
    require(set(selected).issubset(EXPECTED), f'Unexpected datasets: {list(selected)}')
    if not allow_subset:
        require(set(selected) == set(EXPECTED), f'Need both MOT17 and MOT20, found {list(selected)}')
    require(manifest.get('ground_truth_split') == 'train', 'Only GT-supported train split is accepted')
    flags = manifest.get('flags', {})
    require(not flags.get('video') and not flags.get('quick_frames') and not flags.get('skip_eval'),
            'This run is not a complete non-video, evaluated benchmark')
    require(float(flags.get('conf', .4)) == .4 and float(flags.get('detector_floor', .15)) == .15,
            'Unexpected confidence settings; check intended protocol')
    require(int(flags.get('imgsz', 640)) == 640, 'Unexpected image size; check intended protocol')
    runrows, _ = csv_rows(run/'all_runs.csv')
    index = {}
    for row in runrows:
        k = (row.get('dataset'), row.get('sequence'), row.get('method'))
        require(k not in index, f'Duplicate all_runs entry: {k}')
        index[k] = row
    evidence = {'protocol': {
        'split': 'train', 'IoU_threshold': .5, 'DO_PREPROC': True,
        'classes': ['pedestrian'], 'fps_basis': 'frames_processed / summary.wall_seconds',
        'MOT17_selection': 'one FRCNN variant per distinct scene',
        'critical_limitations': ['no cached common detector outputs in this prior run',
                                 'GPU FPS is a single measurement per model/sequence',
                                 'not a held-out test-set result'],
        'baseline': 'StrongSORT tracker core, OSNet, YOLO26x; not AFLink/GSI StrongSORT++',
    }, 'run': str(run), 'sequences': {}, 'files': {}, 'runs': {}}
    totals = {}
    run_dir = run/'trackers'
    common_weights = set(); common_adapter = set()
    baseline_feature_counts = collections.defaultdict(int)
    all_model_rows = []
    for ds, paths in selected.items():
        require(isinstance(paths, list) and paths, f'{ds} selected list empty')
        scene_ids = set();names=[]
        for seq_str in paths:
            seq = Path(seq_str).expanduser().resolve()
            name = seq.name
            m = RE_MOT.fullmatch(name)
            require(m and ds == 'MOT' + m.group(1), f'Invalid sequence {seq}')
            require(m.group(2) not in scene_ids, f'{ds}: duplicate scene {m.group(2)}')
            scene_ids.add(m.group(2))
            if ds == 'MOT17':
                require(name.endswith('-FRCNN'), f'MOT17 requires consistent FRCNN folder choice: {name}')
            gt = seq/'gt'/'gt.txt'; ini = seq/'seqinfo.ini'
            cfg = seq_config(ini)
            imgs = ensure_images(seq, cfg)
            gt_stats = parse_mot_gt(gt, cfg['length'])
            gtlink = run/'gt'/f'{ds}-train'/name
            require(gtlink.exists() and gtlink.resolve() == seq,
                    f'GT link does not point at original dataset: {gtlink}')
            evidence['sequences'][name] = {
                'dataset': ds, **cfg, **gt_stats,
                'gt_sha256': digest(gt), 'seqinfo_sha256': digest(ini),
                'first_image_sha256': digest(imgs[0]), 'last_image_sha256': digest(imgs[-1]),
            }
            names.append(name)
            for mode in modes:
                k = (ds, name, mode)
                require(k in index, f'Missing all_runs.csv entry {k}')
                row = index[k]
                summary_path = Path(row['summary']).resolve()
                result_path = Path(row['predictions']).resolve()
                expected_pred = (run_dir/f'{ds}-train'/mode/'data'/f'{name}.txt').resolve()
                require(result_path == expected_pred,
                        f'{k}: prediction path differs from evaluation input: {result_path}')
                require(result_path.is_file() and summary_path.is_file(), f'{k}: missing files')
                require(summary_path.is_relative_to(run.resolve()), f'{k}: summary path outside run')
                s = load_json(summary_path)
                require(int(s.get('frames_processed', -1)) == cfg['length'],
                        f'{k}: processed frames != seqLength')
                expected_modes={'baseline':'strongsort_osnet', 'v43_full':'sr_osnet_full',
                                'v43_selective':'sr_osnet_selective'}
                require(s.get('mode') == expected_modes[mode], f'{k}: wrong tracker mode')
                require(str(s.get('mot17_timebase')) == 'frame_timestamps',
                        f'{k}: frame timebase is not frame_timestamps')
                require(str(s.get('mot17_sequence')) == str(seq),
                        f'{k}: summary sequence mismatch')
                require(s.get('mot17_predictions') == 'mot_results.txt', f'{k}: not MOT output')
                require(s.get('detector') == 'YOLO26x COCO person', f'{k}: detector unexpected')
                require(s.get('no_render') is True, f'{k}: benchmark used rendering')
                require(s.get('annotated_recording') is False and s.get('raw_recording') is False,
                        f'{k}: recording affected FPS')
                require(float(s.get('conf')) == .4 and float(s.get('detector_floor')) == .15,
                        f'{k}: confidence settings mismatch')
                require(int(s.get('imgsz')) == 640, f'{k}: image size mismatch')
                require(float(s.get('iou')) == .7, f'{k}: YOLO NMS IoU differs from protocol')
                require(not s.get('mirror') and int(s.get('rotate',0)) == 0,
                        f'{k}: changed image orientation')
                require(int(s.get('max_frames_setting', -1)) == 0, f'{k}: max frames not zero')
                require(s.get('exit_reason') == 'end_of_file', f'{k}: ended abnormally: {s.get("exit_reason")}')
                require(s.get('weights_yolo26x_sha256') and s.get('adapter_sha256'),
                        f'{k}: missing weight/adapter hash')
                common_weights.add(s['weights_yolo26x_sha256'])
                common_adapter.add(s['adapter_sha256'])
                if mode == 'baseline':
                    require(s.get('id_memory_enabled') is False, f'{k}: baseline ID Memory unexpectedly on')
                else:
                    require(s.get('id_memory_enabled') is True, f'{k}: proposed ID Memory off')
                require(math.isclose(float(s['wall_seconds']), float(row['seconds']), abs_tol=.05),
                        f'{k}: timing discrepancy')
                require(int(row['frames']) == cfg['length'], f'{k}: all_runs frame count mismatch')
                require(math.isclose(float(row['fps']),cfg['length']/float(s['wall_seconds']),
                                     rel_tol=.005, abs_tol=.03), f'{k}: FPS mismatch')
                require(int(s['reid_features_extracted']) == int(row['reid_features']),
                        f'{k}: ReID count discrepancy')
                require(int(s['total_person_detections']) == int(row['detections']),
                        f'{k}: detection count discrepancy')
                require(float(s['wall_seconds']) > 0, f'{k}: wall_seconds invalid')
                features = int(s['reid_features_extracted']); detected = int(s['total_person_detections'])
                require(features >= 0 and detected >= 0 and features <= detected,
                        f'{k}: ReID features impossible count')
                require(row.get('vram_peak_mib') not in (None, ''), f'{k}: unmeasured VRAM')
                peak = float(row['vram_peak_mib'])
                require(math.isfinite(peak) and peak > 0, f'{k}: invalid VRAM measurement')
                predinfo = parse_mot_prediction(result_path, cfg['length'], cfg['width'], cfg['height'])
                all_model_rows.append({'dataset': ds, 'sequence': name, 'method': mode,
                                       'frames': cfg['length'], 'wall_seconds': float(s['wall_seconds']),
                                       'features': features, 'detections': detected, 'vram_peak_mib': peak,
                                       'prediction_sha256': digest(result_path),
                                       'summary_sha256': digest(summary_path), **predinfo})
                evidence['runs']['/'.join(k)] = all_model_rows[-1]
                baseline_feature_counts[ds, mode] += features
        if not allow_subset:
            require(scene_ids == EXPECTED[ds], f'{ds}: incomplete or extra scene set: {scene_ids}')
        seqmap = run/f'{ds}_seqmap.txt'
        require(seqmap.is_file(), f'Missing saved sequence map: {seqmap}')
        seqmap_names = [x.strip() for x in seqmap.read_text(encoding='utf-8-sig').splitlines() if x.strip()]
        require(seqmap_names[:1] == ['name'] and seqmap_names[1:] == names,
                f'{ds} seqmap is different from original selected set')
        evidence['files'][str(seqmap.relative_to(run))] = digest(seqmap)
        totals[ds] = names
    require(len(index) == len(all_model_rows), f'Extraneous/missing all_runs entries {len(index)} vs {len(all_model_rows)}')
    require(len(common_weights) == 1 and len(common_adapter) == 1,
            'YOLO checkpoint or adapter SHA256 differs between models')
    # Compare with the run manifest; source modified since the run fails closed.
    worker_path = (Path(worker_file).resolve() if worker_file else
                   Path(__file__).resolve().parent/'baram_compare_worker.py')
    require(worker_path.is_file() and digest(worker_path) == manifest.get('worker_sha256'),
            f'Original benchmark worker changed or missing: {worker_path}; cannot certify run code identity')
    evidence['original_worker_sha256'] = digest(worker_path)
    weights = Path(manifest['base_weights']).expanduser()
    require(weights.is_file() and digest(weights) == next(iter(common_weights)),
            'YOLO26x checkpoint changed since run; original weights not reproducible')
    evidence['weights_sha256'] = next(iter(common_weights))
    evidence['adapter_sha256'] = next(iter(common_adapter))
    evidence['manifest_sha256'] = digest(run/'manifest.json')
    # Confirm we evaluate exactly the expected 33 outputs for full three-model run.
    if not allow_subset:
        require(len(all_model_rows) == 33, f'Expected 33 model-sequence runs; found {len(all_model_rows)}')
    return evidence, totals, all_model_rows


def score_from_files(run, totals, python, script, trackeval, audit_dir, repeats):
    scores = {}; details = {}; hashes = {}
    for ds, seqs in totals.items():
        for mode in MODELS:
            reference = None
            for rep in range(1, repeats+1):
                out = audit_dir/'trackeval'/f'pass{rep}'/ds/mode
                out.mkdir(parents=True, exist_ok=False)
                args = [str(python), str(script), '--trackeval', str(trackeval),
                        '--gt', str(run/'gt'), '--trackers', str(run/'trackers'),
                        '--output', str(out), '--seqmap', str(run/f'{ds}_seqmap.txt'),
                        '--dataset', ds, '--method', mode]
                print(f'[EVALUATE] {ds}/{mode} repeat {rep}/{repeats}', flush=True)
                with (out/'execution.log').open('w',encoding='utf-8') as f:
                    done = subprocess.run(args, stdout=f, stderr=subprocess.STDOUT)
                require(done.returncode == 0, f'TrackEval failed: {out}/execution.log')
                summary, detail, artifact_hashes = trackeval_metrics(out, seqs)
                if reference is not None:
                    require(all(abs(summary[k]-reference[0][k]) < 1e-9 for k in METRICS),
                            f'TrackEval summary not reproducible for {ds}/{mode}')
                    require(all(abs(detail[s][k]-reference[1][s][k]) < 1e-9
                                for s in seqs for k in METRICS),
                            f'TrackEval detailed scores not reproducible for {ds}/{mode}')
                else:
                    reference = (summary, detail)
                hashes[f'{ds}/{mode}/repeat{rep}'] = artifact_hashes
            scores[ds, mode], details[ds, mode] = reference
            print('[PASS]', ds, mode,
                  ' '.join(f'{k}={scores[ds,mode][k]:.4f}' for k in METRICS), flush=True)
    return scores, details, hashes


def create_report(audit_dir, run, evidence, rows, totals, scores, details, hashes, repeats, te):
    # Check code fingerprints before emitting any user-facing VERIFIED_* results.
    te_dataset_digest = digest(te/'trackeval'/'datasets'/'mot_challenge_2d_box.py')
    te_evaluator_digest = digest(te/'trackeval'/'eval.py')
    scoring_digest = digest(Path(__file__).resolve().parent/'baram_trackeval_direct.py')
    stamp = datetime.now(timezone.utc).isoformat()
    summary = []
    per_seq = []
    for ds, seqs in totals.items():
        for mode in MODELS:
            relevant = [r for r in rows if r['dataset']==ds and r['method']==mode]
            frames = sum(r['frames'] for r in relevant)
            duration = sum(r['wall_seconds'] for r in relevant)
            features = sum(r['features'] for r in relevant)
            dets = sum(r['detections'] for r in relevant)
            require(duration > 0 and len(relevant)==len(seqs), 'Missing FPS input or sequence')
            m = scores[ds,mode]
            summary.append({'dataset':ds, 'method':mode, 'sequences':len(seqs),
                            **{k: round(m[k], 6) for k in METRICS},
                            'FPS': round(frames/duration, 5),
                            'ReID_extractions': features, 'detections': dets,
                            'ReID_per_detection_pct':round(100*features/max(dets,1),4),
                            'Peak_VRAM_MiB':max(r['vram_peak_mib'] for r in relevant)})
            for r in relevant:
                m = details[ds,mode][r['sequence']]
                per_seq.append({'dataset':ds, 'method':mode, 'sequence':r['sequence'],
                                **{k:round(m[k],6) for k in METRICS},
                                'FPS':round(r['frames']/r['wall_seconds'],5),
                                'ReID_extractions':r['features'], 'detections':r['detections'],
                                'Peak_VRAM_MiB':r['vram_peak_mib']})
    for filename, values in (('VERIFIED_comparison.csv', summary),
                             ('VERIFIED_per_sequence.csv', per_seq)):
        with (audit_dir/filename).open('w',newline='',encoding='utf-8-sig') as f:
            w=csv.DictWriter(f,fieldnames=list(values[0]));w.writeheader();w.writerows(values)
    def fmt(x, digits=2):return f'{x:.{digits}f}'
    md=['# BARAM YOLO26x: validated MOT17/MOT20 train A/B benchmark','','**Status: PASS — format, completeness, code/weight consistency, TrackEval per-sequence checks and numerical repeatability.**',
        'Not an MOTChallenge hidden-test leaderboard score; this evaluates annotated train sequences.',
        'StrongSORT here means project baseline using official tracker core, not StrongSORT++ AFLink/GSI.',
        'IDSW alone is not evidence of improved tracking: VERIFY can suppress output and increase FN.',
        '', '| Dataset | Model | HOTA ↑ | IDF1 ↑ | MOTA ↑ | IDSW ↓ | FPS ↑ | Re-ID extracts ↓ | Peak VRAM MiB ↓ |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in summary:
        md.append(f'| {r["dataset"]} | {r["method"]} | {fmt(r["HOTA"])} | {fmt(r["IDF1"])} | {fmt(r["MOTA"])} | {int(r["IDSW"])} | {fmt(r["FPS"])} | {r["ReID_extractions"]:,} | {r["Peak_VRAM_MiB"]:,.0f} |')
    md += ['', '## Protocol','- Input: original MOT17/MOT20 train sequences; MOT17 FRCNN-named unique scenes, **YOLO26x** is detector for all arms.',
           '- Evaluation: TrackEval, MOTChallenge 2D pedestrian, standard distractor preprocessing ON, IoU threshold 0.5.',
           '- All frames evaluated; no rendered video while timing; data loading, YOLO, OSNet, tracking and result writing included.',
           '- FPS = sum(frames) / sum(sequence wall_seconds), **not** arithmetic mean of per-sequence FPS.',
           '- Feature extractions = number of OSNet-encoded detector crops, **not** total number of forward calls.',
           '- Peak VRAM = highest nvidia-smi sampled per-process MiB, approximately 0.5 second sampling; can miss shorter peaks.',
           f'- TrackEval evaluated {repeats} independent processes; compared combined and per-sequence metrics for exact numerical agreement.',
           '- This report does not prove generalization or isolate the individual contribution of each V4.3 feature; use a held-out test set and ablation experiments.',
           '- Device/driver, uncontrolled hardware contention and run-order can affect timing. Repeat full inference with order reversal for publication-grade FPS comparisons.',
           '',f'Original run: `{run}`',f'TrackEval checkout: `{te}`',f'UTC report date: {stamp}',
           '', 'Technical evidence: `VERIFICATION.json`, `VERIFIED_per_sequence.csv`, and `trackeval/pass*/.../execution.log`.']
    (audit_dir/'VERIFIED_comparison.md').write_text('\n'.join(md)+'\n',encoding='utf-8')
    evidence['results'] = {'summary':summary, 'per_sequence':per_seq,
                           'repeat_count':repeats,'trackeval_output_hashes':hashes,
                           'trackeval_code_sha256':te_dataset_digest,
                           'trackeval_eval_py_sha256':te_evaluator_digest,
                           'scoring_script_sha256':scoring_digest,
                           'generated_at_utc':stamp}
    evidence['status']='PASS'
    (audit_dir/'VERIFICATION.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('\n'.join(md[:len(summary)+10]),flush=True)
    print('[ALL VALIDATIONS PASSED] Report:',audit_dir/'VERIFIED_comparison.md',flush=True)


def main():
    ap = argparse.ArgumentParser(description='Strict MOT metrics audit V2 (official detailed CSV compatible; no YOLO rerun)')
    ap.add_argument('--run',type=Path,required=True,help='Original timestamped A/B experiment directory')
    ap.add_argument('--trackeval',type=Path,default=Path.home()/'Downloads/TrackEval')
    ap.add_argument('--python',type=Path,default=Path.home()/'Downloads/BARAM_SR_StrongSORT/.venv/bin/python')
    ap.add_argument('--allow-subset',action='store_true',help='Explicitly accept less than the 7+4 train sequences')
    ap.add_argument('--verify-only',action='store_true',help='Only validate inputs; no TrackEval scoring')
    ap.add_argument('--repeat',type=int,default=2,help='Independent TrackEval runs; default 2')
    a=ap.parse_args()
    if not (1 <= a.repeat <= 4):ap.error('--repeat must be 1..4')
    run=a.run.expanduser().resolve()
    te=a.trackeval.expanduser().resolve()
    py=a.python.expanduser().resolve()
    script=Path(__file__).resolve().parent/'baram_trackeval_direct.py'
    audit_dir=run/'metric_audit_v2'
    try:
        require(run.is_dir(), f'No experiment: {run}')
        require(py.is_file(), f'Python not found: {py}')
        evidence, totals, rows = verify(run, a.allow_subset)
        print('[PRECHECK PASS]',len(rows),'model-sequence outputs, dataset GT, frame counts, ID/bboxes, config, SHA256',flush=True)
        if a.verify_only:
            print('[VERIFICATION ONLY] No evaluation executed. Not a complete metric report.')
            return 0
        require((te/'trackeval/datasets/mot_challenge_2d_box.py').is_file(),
                f'Official TrackEval not installed: {te}')
        require(not audit_dir.exists(), f'Audit folder already exists: {audit_dir}; preserve it; choose a new result directory or move this folder aside')
        audit_dir.mkdir(parents=True)
        # Freeze input fingerprints before and after evaluation to detect accidental changes.
        original_hashes={key: val['prediction_sha256'] for key, val in evidence['runs'].items()}
        scores,details,hashes=score_from_files(run,totals,py,script,te,audit_dir,a.repeat)
        for key,sha_before in original_hashes.items():
            ds,seq,mode=key.split('/')
            p=run/'trackers'/f'{ds}-train'/mode/'data'/f'{seq}.txt'
            require(digest(p)==sha_before, f'Input prediction modified during evaluation: {p}')
        for seq, info in evidence['sequences'].items():
            # Recheck original GT fingerprint in the dataset symlink.
            ds=info['dataset']
            p=run/'gt'/f'{ds}-train'/seq/'gt/gt.txt'
            require(digest(p)==info['gt_sha256'], f'Ground truth changed while scoring: {p}')
        create_report(audit_dir,run,evidence,rows,totals,scores,details,hashes,a.repeat,te)
        return 0
    except (AuditFailure, OSError, ValueError, KeyError, TypeError) as exc:
        if audit_dir.is_dir():
            (audit_dir/'AUDIT_FAILED.txt').write_text(str(exc)+'\n',encoding='utf-8')
        print('[AUDIT FAILED — DO NOT REPORT METRICS]',exc,file=sys.stderr)
        return 1

if __name__=='__main__':
    raise SystemExit(main())
