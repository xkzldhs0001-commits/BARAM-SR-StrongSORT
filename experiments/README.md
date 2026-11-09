# Experiment files

- `poster_comparison.csv`: 2026-11-07 poster's four rows, including the poster's stated FPS. HOTA/IDF1/MOTA/IDSW were reported in the poster but their full TrackEval output is not committed.
- `run_20261106_comparison.csv`: 2026-11-06 run's FPS, Re-ID, and peak VRAM. **No valid TrackEval metrics** because the console log reports a SEQMAP_FILE type error.
- `v45_diagnostic_excerpt.csv`: selected results previously shared for a V4.5 ablation; the full original output and experiment package have not been recovered, so these figures must not be combined with the poster comparison.

All benchmark summaries are from MOT17/MOT20 training sequences, **not the held-out official leaderboard**.

Calculate changes for the poster table:

```bash
python scripts/analyze_experiment_results.py --dataset MOT17
python scripts/analyze_experiment_results.py --dataset MOT20
```

Calculate changes for the raw November 6 performance counters:

```bash
python scripts/analyze_experiment_results.py --dataset MOT17 \
  --csv experiments/run_20261106_comparison.csv \
  --baseline baseline --method v43_selective
python scripts/analyze_experiment_results.py --dataset MOT20 \
  --csv experiments/run_20261106_comparison.csv \
  --baseline baseline --method v43_selective
```

To verify the analysis script without datasets or GPU:

```bash
python -m unittest discover -s tests -p 'test_analyze_experiment_results.py' -v
```

See `../PROJECT_REPORT_KO.md` for methodology, results, limitations, and the evaluation failure.
