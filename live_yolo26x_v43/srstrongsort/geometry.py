"""Geometry and association formulas from the BARAM proposal."""
from __future__ import annotations
import numpy as np

CHI2_95_DOF4 = 9.4877

def xyxy_to_xywh_center(box):
    x1, y1, x2, y2 = map(float, box)
    w, h = max(x2-x1, 1.0), max(y2-y1, 1.0)
    return np.array([(x1+x2)/2, (y1+y2)/2, w, h], dtype=float)

def xywh_center_to_xyxy(v):
    x, y, w, h = np.asarray(v, dtype=float)[:4]
    w, h = max(w, 1.0), max(h, 1.0)
    return np.array([x-w/2, y-h/2, x+w/2, y+h/2], dtype=float)

def iou(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    ix1, iy1 = np.maximum(a[:2], b[:2]); ix2, iy2 = np.minimum(a[2:], b[2:])
    inter = max(0., ix2-ix1)*max(0., iy2-iy1)
    area_a = max(0., a[2]-a[0])*max(0., a[3]-a[1])
    area_b = max(0., b[2]-b[0])*max(0., b[3]-b[1])
    return float(inter/(area_a+area_b-inter+1e-12))

def ars(a, b):
    """ARS = 1 - 4/pi^2 * (atan(w1/h1)-atan(w2/h2))^2."""
    aa = max(0., float(a[2]-a[0]))/max(1., float(a[3]-a[1]))
    ab = max(0., float(b[2]-b[0]))/max(1., float(b[3]-b[1]))
    return float(1. - 4./np.pi**2 * (np.arctan(aa)-np.arctan(ab))**2)

def normalize_feature(v):
    v = np.asarray(v, dtype=np.float32).reshape(-1)
    return v / max(float(np.linalg.norm(v)), 1e-12)

def appearance_cost(memory, detected):
    return float(1.0 - np.clip(np.dot(normalize_feature(memory),normalize_feature(detected)), -1., 1.))
