#!/usr/bin/env python3
"""Render *actual saved* MOT17/MOT20 same-frame StrongSORT vs V4.3 Selective visual comparisons.

No inference or video synthesis: reads the exact source images and MOTChallenge prediction TXT
files produced by a completed A/B run.  Requires OpenCV from the existing BARAM venv.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

HOME = Path.home()
DEFAULT_ROOT = HOME / 'Downloads' / 'BARAM_SR_StrongSORT'
DEFAULT_RUN = DEFAULT_ROOT / 'outputs' / 'BARAM_V44_IDSW_AB' / '20261106_195403'
DEFAULT_SEQUENCES = ('MOT17-10-FRCNN', 'MOT20-02')


def dataset_from_seq(seq: str) -> str:
    return 'MOT17' if seq.startswith('MOT17-') else 'MOT20' if seq.startswith('MOT20-') else ''


def get_seq_path(run: Path, seq: str, dataset_root: Path) -> Path:
    ds = dataset_from_seq(seq)
    if not ds:
        raise ValueError(f'Unknown MOT sequence: {seq}')
    candidates = [run / 'gt' / (ds + '-train') / seq,
                  dataset_root / ds / 'train' / seq,
                  dataset_root / 'datasets' / ds / 'train' / seq,
                  DEFAULT_ROOT / 'datasets' / ds / 'train' / seq]
    manifest = run / 'manifest.json'
    if manifest.is_file():
        try:
            data = json.loads(manifest.read_text(encoding='utf-8'))
            candidates += [Path(x) for x in data.get('selected', {}).get(ds, [])
                           if Path(x).name == seq]
        except (ValueError, OSError, TypeError):
            pass
    for p in candidates:
        if p.name == seq and (p / 'img1').is_dir() and (p / 'seqinfo.ini').is_file():
            return p.resolve()
    raise FileNotFoundError(
        f'Original image sequence missing: {seq}\n'
        f'Look for img1/ and seqinfo.ini under the MOT dataset directory.\n'
        f'Tried: ' + ', '.join(str(p) for p in candidates))


def images_in_seq(seqdir: Path):
    index = {}
    for p in sorted((seqdir / 'img1').iterdir()):
        if p.is_file() and p.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp') and p.stem.isdigit():
            index[int(p.stem)] = p
    if not index:
        raise FileNotFoundError(f'No numeric frame images under {seqdir / "img1"}')
    return index


def get_pred_path(run: Path, ds: str, seq: str, method: str) -> Path:
    a = run / 'trackers' / (ds + '-train') / method / 'data' / (seq + '.txt')
    if a.is_file():
        return a
    run_dir = run / 'runs' / ds / seq / method
    files = list(run_dir.glob('*/mot_results.txt')) + list(run_dir.glob('mot_results.txt'))
    if files:
        return files[0]
    raise FileNotFoundError(f'MOT results missing for {ds}/{seq}/{method}; checked {a}')


def load_mot(path: Path):
    rows = defaultdict(list)
    with path.open('r', encoding='utf-8-sig', newline='') as f:
        for i, parts in enumerate(csv.reader(f), start=1):
            if not parts or not parts[0].strip() or parts[0].lstrip().startswith('#'):
                continue
            if len(parts) < 6:
                raise ValueError(f'{path}:{i}: expected >=6 CSV columns, got {len(parts)}')
            try:
                frame = int(float(parts[0]))
                track = int(float(parts[1]))
                x, y, w, h = [float(x) for x in parts[2:6]]
            except (ValueError, OverflowError) as exc:
                raise ValueError(f'{path}:{i}: invalid MOT values') from exc
            if frame < 1 or track < 1 or w <= 0 or h <= 0:
                continue
            # MOTChallenge annotations use 1-based x,y; image pixel coords use 0-based.
            rows[frame].append((track, x - 1., y - 1., w, h))
    return rows


def color_for(track: int):
    digest = hashlib.sha256(str(track).encode()).digest()
    return (int(65 + digest[0] * .7), int(65 + digest[1] * .7), int(65 + digest[2] * .7))  # BGR


def render_panel(image, boxes, panel_width: int, heading: str):
    height, width = image.shape[:2]
    out_width = min(int(panel_width), width) if panel_width > 0 else width
    scale = out_width / float(width)
    out_height = max(1, round(height * scale))
    canvas = cv2.resize(image, (out_width, out_height), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    line_thick = max(1, round(out_width / 650))
    font_scale = max(.38, out_width / 1900)
    for track, x, y, w, h in boxes:
        x1, y1 = int(x * scale), int(y * scale)
        x2, y2 = int((x + w) * scale), int((y + h) * scale)
        x1, y1 = min(out_width-1, max(0, x1)), min(out_height-1, max(0, y1))
        x2, y2 = min(out_width-1, max(0, x2)), min(out_height-1, max(0, y2))
        if x2 <= x1 or y2 <= y1:
            continue
        c = color_for(track)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), c, line_thick)
        lbl = f'ID {track}'
        size, base = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        bx2 = min(out_width - 1, x1 + size[0] + 6)
        by1 = max(0, y1 - size[1] - base - 7)
        by2 = min(out_height-1, by1 + size[1] + base + 7)
        cv2.rectangle(canvas, (x1, by1), (bx2, by2), c, -1)
        cv2.putText(canvas, lbl, (x1+3, min(out_height-1, by1+size[1]+2)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 1, cv2.LINE_AA)
    header_h, footer_h = 64, 25
    header = cv2.copyMakeBorder(canvas, header_h, footer_h, 0, 0, cv2.BORDER_CONSTANT, value=(244,244,244))
    cv2.rectangle(header, (0, 0), (out_width, header_h-1), (24, 28, 34), -1)
    cv2.putText(header, heading, (14, 28), cv2.FONT_HERSHEY_SIMPLEX,
                min(.7, max(.4, out_width/1450)), (255,255,255), 2, cv2.LINE_AA)
    cv2.putText(header, f'{len(boxes)} visible track boxes', (14, 53),
                cv2.FONT_HERSHEY_SIMPLEX, min(.55, max(.35, out_width/1850)), (220,220,220), 1, cv2.LINE_AA)
    cv2.putText(header, 'Same source frame  |  ID numbers are independent for each model',
                (12, header_h+out_height+18), cv2.FONT_HERSHEY_SIMPLEX,
                min(.45, max(.29, out_width/2350)), (55,55,55), 1, cv2.LINE_AA)
    return header


def compare_frame(frame_image: Path, frame: int, seq: str, labels, out_file: Path, panel_width: int):
    img = cv2.imread(str(frame_image), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f'Cannot decode image: {frame_image}')
    ds = dataset_from_seq(seq)
    lhs = render_panel(img, labels['baseline'].get(frame, []), panel_width,
                       f'{ds} / frame {frame:06d}  |  StrongSORT')
    rhs = render_panel(img, labels['v43_selective'].get(frame, []), panel_width,
                       f'{ds} / frame {frame:06d}  |  SR-StrongSORT (V4.3)')
    separator = np.full((lhs.shape[0], 8, 3), 255, dtype=np.uint8)
    pair = cv2.hconcat([lhs, separator, rhs])
    out_file.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(out_file), pair):
        raise RuntimeError(f'Failed to save {out_file}')
    return pair


def pick_frames(image_index, boxes_baseline, boxes_selective, how_many=3):
    all_frames = sorted(image_index)
    first = all_frames[0]
    last = all_frames[-1]
    # Avoid first/last 10% of sequence when possible.
    pool = [f for f in all_frames if first + .10*(last-first) <= f <= first + .90*(last-first)] or all_frames
    counts = {f: (len(boxes_baseline.get(f, [])), len(boxes_selective.get(f, []))) for f in pool}
    def crowded(f):
        a, b = counts[f]
        return min(a,b)*3 + max(a,b)
    def divergent(f):
        a, b = counts[f]
        return abs(a-b)*2 + min(a,b)*.25
    center = .5*(first+last)
    targets = [max(pool,key=lambda f:(crowded(f),-abs(f-center))),
               max(pool,key=lambda f:(divergent(f),-abs(f-center))),
               min(pool,key=lambda f:(abs(f-center),-crowded(f)))]
    gap = max(10, int((last-first)*.05))
    chosen = []
    for target in targets:
        if len(chosen) >= how_many:
            break
        if all(abs(target-f) >= gap for f in chosen):
            chosen.append(target)
        else:
            alternate = [f for f in pool if all(abs(f-v) >= gap for v in chosen)]
            if alternate:
                chosen.append(max(alternate,key=lambda f:(crowded(f),-abs(f-target))))
    while len(chosen)<how_many:
        remaining = [f for f in all_frames if f not in chosen]
        if not remaining:break
        chosen.append(min(remaining,key=lambda f: min(abs(f-v) for v in chosen) if chosen else 0) if chosen else remaining[0])
    return sorted(chosen)


def render(args):
    run = args.run.expanduser().resolve()
    if not run.is_dir():
        raise FileNotFoundError(f'No completed A/B run: {run}')
    out = (args.out.expanduser() if args.out else
           (DEFAULT_ROOT / 'outputs' / 'MOT_SAME_FRAME_COMPARISONS' / run.name)).resolve()
    out.mkdir(parents=True, exist_ok=True)
    records = []
    for seq in args.sequences:
        ds = dataset_from_seq(seq)
        src = get_seq_path(run, seq, args.dataset_root.expanduser())
        img_index = images_in_seq(src)
        source_paths = {mode: get_pred_path(run, ds, seq, mode) for mode in ('baseline','v43_selective')}
        labels = {mode: load_mot(path) for mode,path in source_paths.items()}
        if args.frames:
            frames = [f for f in args.frames if f in img_index]
            if not frames:
                raise ValueError(f'None of the requested frame numbers exists for {seq}')
        else:
            frames = pick_frames(img_index, labels['baseline'], labels['v43_selective'], args.num_frames)
        pairs = []
        for frame in frames:
            fpath = out / f'{seq}_frame{frame:06d}_StrongSORT_vs_SR-StrongSORT.png'
            image = compare_frame(img_index[frame], frame, seq, labels, fpath, args.panel_width)
            pairs.append(image)
            records.append({'dataset':ds, 'sequence':seq, 'frame':frame,
                            'source_image':str(img_index[frame]),
                            'baseline_result':str(source_paths['baseline']),
                            'selective_result':str(source_paths['v43_selective']),
                            'baseline_boxes':len(labels['baseline'].get(frame, [])),
                            'selective_boxes':len(labels['v43_selective'].get(frame, [])),
                            'comparison_png':str(fpath)})
            print(f'[SAVED] {fpath}')
        if len(pairs)>1:
            sep = 14
            sheet_width = pairs[0].shape[1]
            sheet_parts = []
            for i, pair in enumerate(pairs):
                if i:
                    sheet_parts.append(np.full((sep, sheet_width, 3), 255, dtype=np.uint8))
                sheet_parts.append(pair)
            sheet = cv2.vconcat(sheet_parts)
            sheet_path = out / f'{seq}_SAME_FRAME_CONTACT_SHEET.png'
            cv2.imwrite(str(sheet_path), sheet)
            print(f'[SHEET] {sheet_path}')
    (out/'source_manifest.json').write_text(json.dumps({'run':str(run), 'note':'Rendered using original MOT img1 images and saved MOT tracker predictions; no re-inference.',
                                                         'comparisons':records},ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'\n[DONE] Comparison images folder: {out}')
    print('Open with: xdg-open "' + str(out) + '"')


def main():
    p = argparse.ArgumentParser(description='Same-frame comparison: actual MOT17/MOT20 images, StrongSORT vs V4.3 Selective')
    p.add_argument('--run',type=Path,default=DEFAULT_RUN, help='A/B run containing trackers/{MOT17,MOT20}-train/<method>/data/*.txt')
    p.add_argument('--dataset-root',type=Path,default=DEFAULT_ROOT/'datasets',help='Fallback MOT dataset root')
    p.add_argument('--sequences',nargs='+',default=list(DEFAULT_SEQUENCES),help='Examples: MOT17-10-FRCNN MOT20-02')
    p.add_argument('--num-frames',type=int,default=3,help='Automatically select this many frames per sequence')
    p.add_argument('--frames',nargs='+',type=int,help='Explicit same frame numbers to render for ALL specified sequences')
    p.add_argument('--panel-width',type=int,default=1050,help='Width of each displayed frame in pixels; set 1920 for full HD')
    p.add_argument('--out',type=Path,help='Output directory')
    args=p.parse_args()
    if args.num_frames<1 or args.panel_width<320:
        p.error('--num-frames must be >=1, --panel-width >=320')
    render(args)


if __name__ == '__main__':
    try:
        main()
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        raise SystemExit('[ERROR] ' + str(exc))
