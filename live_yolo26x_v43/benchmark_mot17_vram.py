#!/usr/bin/env python3
"""Run *existing V4.3 modules* on one MOT17 image sequence and sample process VRAM.

No edits to the existing live.py, run.sh, tracking modules or benchmark output.
Requirements: Python standard library for this launcher; the pre-existing project
.venv is used for inference. nvidia-smi is optional for process VRAM sampling.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime

HERE = Path(__file__).resolve().parent
ROOT = Path.home() / 'Downloads' / 'BARAM_SR_StrongSORT'
IGNORED = {'.venv', 'venv', 'outputs', '.git', '__pycache__', 'node_modules',
           'third_party', '.cache', 'site-packages', 'wandb', 'runs'}


def find_sequences():
    """Search common local MOT17 locations; avoid unrelated output video dirs."""
    base = Path.home() / 'Downloads'
    seqs = {}
    if not base.is_dir():
        return []
    for folder, subdirs, files in os.walk(base):
        subdirs[:] = [d for d in subdirs if d not in IGNORED]
        p = Path(folder)
        if 'seqinfo.ini' in files and p.name.startswith('MOT17-') and (p / 'img1').is_dir():
            seqs[str(p.resolve())] = p.resolve()
            subdirs[:] = []
    # MOT17-09 typically has many pedestrians and is a useful quick VRAM stress check.
    preferred = ('MOT17-09-FRCNN', 'MOT17-11-FRCNN', 'MOT17-04-FRCNN')
    return sorted(seqs.values(), key=lambda p: (preferred.index(p.name) if p.name in preferred else 100, str(p)))


def sample_process_memory(pid: int):
    """GPU memory accounting reported by NVIDIA for exactly this python process."""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-compute-apps=pid,used_gpu_memory',
             '--format=csv,noheader,nounits'],
            check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, timeout=3)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    for line in result.stdout.splitlines():
        fields = [field.strip() for field in line.split(',')]
        if len(fields) < 2 or not fields[0].isdigit() or int(fields[0]) != pid:
            continue
        try:
            return float(re.search(r'\d+(?:\.\d+)?', fields[1]).group(0))
        except (ValueError, AttributeError):
            return None
    # Child process exists but GPU context not yet initialized.
    return None


def gpu_descriptor():
    try:
        c = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total',
                            '--format=csv,noheader,nounits'],
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                           text=True, timeout=3)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return c.stdout.strip().splitlines() if c.returncode == 0 else None


def main():
    p = argparse.ArgumentParser(description='BARAM V4.3 MOT17 sequence + actual process VRAM peak')
    p.add_argument('--seq', type=Path, help='MOT17-XX-FRCNN sequence directory containing img1 and seqinfo.ini')
    p.add_argument('--mode', choices=['sr_osnet_full', 'sr_osnet_selective'], default='sr_osnet_full')
    p.add_argument('--root', type=Path, default=ROOT)
    p.add_argument('--max-frames', type=int, default=0, help='0 = all sequence frames; use 300 for a quick check')
    p.add_argument('--sample-sec', type=float, default=.25)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--imgsz', type=int, default=640)
    p.add_argument('--mot-output', action='store_true', help='also save track coordinates in MOT format')
    p.add_argument('--no-video', action='store_true',
                   help='disable annotated.mp4 output (use for encoder-free FPS comparison)')
    p.add_argument('--no-internal-id', action='store_true',
                   help='hide internal StrongSORT track ID from annotated video')
    a = p.parse_args()
    if a.max_frames < 0 or a.sample_sec < .1:
        p.error('--max-frames must be >=0 and --sample-sec must be >=0.1')
    root = a.root.expanduser().resolve()
    py = root / '.venv/bin/python'
    for required in (py, HERE / 'live_mot17.py', HERE / 'identity_memory.py', HERE / 'occlusion_bridge.py'):
        if not required.exists():
            p.error(f'Missing required file: {required}')
    if not a.seq:
        found = find_sequences()
        if not found:
            p.error('MOT17 folder not found under ~/Downloads. Specify --seq /path/to/MOT17-09-FRCNN')
        a.seq = found[0]
        print(f'[AUTO] Selected existing dataset: {a.seq}', flush=True)
        if len(found) > 1:
            print('[INFO] Other sequences found:', ', '.join(x.name for x in found[:10]), flush=True)
    seq = a.seq.expanduser().resolve()
    if not (seq / 'img1').is_dir() or not (seq / 'seqinfo.ini').is_file():
        p.error(f'Not a MOT17 sequence directory: {seq} (need img1/ and seqinfo.ini)')
    smi = shutil.which('nvidia-smi')
    if not smi:
        print('[WARN] nvidia-smi is absent. PyTorch allocator peak will still be saved, but full-process GPU VRAM is unavailable.', flush=True)
    else:
        print('[GPU]', gpu_descriptor(), flush=True)
    outroot = root / 'outputs' / 'MOT17_V43_VRAM'
    outroot.mkdir(parents=True, exist_ok=True)
    cmd = [str(py), str(HERE / 'live_mot17.py'), '--root', str(root),
           '--mot17-seq', str(seq), '--mode', a.mode, '--no-display',
           '--max-frames', str(a.max_frames), '--device', a.device,
           '--imgsz', str(a.imgsz), '--out', str(outroot)]
    if a.mot_output:
        cmd.append('--mot-output')
    if not a.no_video:
        cmd.append('--record-annotated')
    if not a.no_internal_id:
        cmd.append('--show-internal-id')
    env = os.environ.copy()
    env['PYTHONPATH'] = os.pathsep.join([str(root / 'third_party/fast-reid'), str(root), env.get('PYTHONPATH', '')])
    env['OMP_NUM_THREADS'] = env.get('OMP_NUM_THREADS', '4')
    env['MKL_NUM_THREADS'] = env.get('MKL_NUM_THREADS', '4')
    env['PYTHONUNBUFFERED'] = '1'
    report_dir = outroot / ('vram_logs_' + datetime.now().strftime('%Y%m%d_%H%M%S'))
    report_dir.mkdir(parents=True, exist_ok=False)
    samples = []
    stop = threading.Event()
    proc = None
    output_dir = None
    return_code = 1
    started = time.monotonic()
    print('[RUN] sequence:', seq, flush=True)
    print('[RUN] mode:', a.mode, '| image size:', a.imgsz,
          '| max frames:', a.max_frames or 'all', flush=True)
    print('[INFO] Sampling process VRAM every', a.sample_sec, 'seconds', flush=True)
    print('[VIDEO] annotated.mp4:', 'ENABLED' if not a.no_video else 'DISABLED', flush=True)

    def poll_gpu(target_pid):
        while not stop.is_set():
            mib = sample_process_memory(target_pid) if smi else None
            samples.append((datetime.now().isoformat(timespec='milliseconds'), mib))
            stop.wait(a.sample_sec)

    try:
        proc = subprocess.Popen(cmd, cwd=str(HERE), env=env,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, bufsize=1)
        watcher = threading.Thread(target=poll_gpu, args=(proc.pid,), daemon=True)
        watcher.start()
        with (report_dir / 'terminal.log').open('w', encoding='utf-8') as log:
            for line in proc.stdout:
                print(line, end='', flush=True)
                log.write(line)
                if line.startswith('[OUTPUT] '):
                    output_dir = Path(line[len('[OUTPUT] '):].strip())
        return_code = proc.wait()
    except KeyboardInterrupt:
        print('\n[STOP] Cancelled by user. Terminating benchmark.', flush=True)
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        return_code = 130
    finally:
        stop.set()
        if 'watcher' in locals():
            watcher.join(timeout=4)
    elapsed = time.monotonic() - started
    with (report_dir / 'vram_samples.csv').open('w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['timestamp', 'process_vram_mib'])
        writer.writerows((ts, '' if mib is None else mib) for ts, mib in samples)
    values = [v for _, v in samples if v is not None]
    tracker_summary = None
    if output_dir and (output_dir / 'summary.json').is_file():
        tracker_summary = json.loads((output_dir / 'summary.json').read_text(encoding='utf-8'))
    video_path = output_dir / 'annotated.mp4' if output_dir and not a.no_video else None
    video_exists = bool(video_path and video_path.is_file() and video_path.stat().st_size > 1024)
    stats = {
        'sequence': str(seq), 'mode': a.mode, 'command': cmd,
        'return_code': return_code, 'sample_interval_seconds': a.sample_sec,
        'nvidia_smi_samples_valid': len(values),
        'nvidia_smi_process_vram_peak_mib': max(values) if values else None,
        'nvidia_smi_process_vram_mean_mib': round(sum(values)/len(values), 2) if values else None,
        'torch_cuda_peak_allocated_mib': (tracker_summary or {}).get('torch_cuda_peak_allocated_mib'),
        'torch_cuda_peak_reserved_mib': (tracker_summary or {}).get('torch_cuda_peak_reserved_mib'),
        'frames_processed': (tracker_summary or {}).get('frames_processed'),
        'fps': (tracker_summary or {}).get('mean_processing_fps_including_capture_display'),
        'elapsed_seconds': round(elapsed, 2),
        'tracker_output': str(output_dir) if output_dir else None,
        'annotated_video': str(video_path) if video_exists else None,
        'annotated_video_bytes': video_path.stat().st_size if video_exists else None,
        'video_requested': not a.no_video,
        'sampling_report_directory': str(report_dir),
        'gpu_description': gpu_descriptor() if smi else None,
        'warning': 'nvidia-smi per-process memory includes CUDA context and non-PyTorch allocations. '
                   'Sampler can miss transients shorter than sample interval; '
                   'PyTorch peak reserved is allocator-only, not entire process VRAM.'
    }
    (report_dir / 'vram_report.json').write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding='utf-8')
    print('\n========== MOT17 V4.3 VRAM RESULT ==========', flush=True)
    print('[DATASET]', seq.name, flush=True)
    print('[MODE]', a.mode, flush=True)
    print('[EXIT CODE]', return_code, flush=True)
    print('[FRAMES]', stats['frames_processed'], '| [FPS]', stats['fps'], flush=True)
    if values:
        print(f'[GPU PROCESS VRAM PEAK] {max(values):.0f} MiB ({max(values)/1024:.2f} GiB)', flush=True)
        print(f'[GPU PROCESS VRAM MEAN] {sum(values)/len(values):.0f} MiB', flush=True)
    else:
        print('[GPU PROCESS VRAM PEAK] nvidia-smi process data unavailable (NOT measured)', flush=True)
    print('[TORCH PEAK ALLOCATED]', stats['torch_cuda_peak_allocated_mib'], 'MiB', flush=True)
    print('[TORCH PEAK RESERVED]', stats['torch_cuda_peak_reserved_mib'], 'MiB', flush=True)
    print('[RESULT]', report_dir / 'vram_report.json', flush=True)
    print('[CSV]', report_dir / 'vram_samples.csv', flush=True)
    if output_dir:
        print('[TRACKING]', output_dir / 'summary.json', flush=True)
    if video_exists:
        print('[VIDEO SAVED]', video_path, flush=True)
        print(f'[VIDEO SIZE] {video_path.stat().st_size / (1024**2):.1f} MiB', flush=True)
    elif not a.no_video:
        print('[VIDEO NOT FOUND] Check terminal.log and tracking output folder.', flush=True)
    if return_code != 0:
        print('[ERROR] Benchmark did not finish successfully. Check terminal.log.', flush=True)
    return return_code


if __name__ == '__main__':
    sys.exit(main())
