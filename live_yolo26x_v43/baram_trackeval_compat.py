#!/usr/bin/env python3
"""Compatibility entrypoint: official TrackEval, with legacy NumPy aliases on new NumPy.
Does not change TrackEval source or scoring formulas.
"""
from pathlib import Path
import runpy
import sys
import numpy as np

if len(sys.argv)<2:
    raise SystemExit('Usage: python baram_trackeval_compat.py /path/to/run_mot_challenge.py [TrackEval options]')
script=Path(sys.argv[1]).expanduser().resolve()
if not script.is_file():raise SystemExit('TrackEval script not found: '+str(script))
for name,val in (('float',float),('int',int),('bool',bool)):
    if name not in np.__dict__:setattr(np,name,val)
sys.argv=[str(script),*sys.argv[2:]]
runpy.run_path(str(script),run_name='__main__')
