"""BARAM proposal-inspired tracker with robust 2-stage association.

Selective mode: OSNet is the ONLY feature extractor. The core ReID trigger is
>=2 IoU+ARS-valid tracks for one detection. Initial appearance bootstrap and
lost-track recovery are separately counted *extensions*, never hidden full-ReID.

bot_full mode uses an externally supplied trained BoT/ResNeSt50 extractor,
with the SAME tracking logic for an extractor/gating-controlled comparison.
It is not falsely labelled the official StrongSORT (ECC etc. are not included).
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.optimize import linear_sum_assignment
from .kalman import NSAKalman
from .geometry import (CHI2_95_DOF4, iou, ars, appearance_cost,
                       normalize_feature, xyxy_to_xywh_center)


@dataclass
class Detection:
    box: np.ndarray
    confidence: float


class Track:
    def __init__(self, track_id, det, feature=None):
        self.id = track_id
        self.kf = NSAKalman(det.box)
        self.confidence = float(det.confidence)
        self.memory = normalize_feature(feature) if feature is not None else None
        self.hits = 1
        self.age_since_update = 0
        self.age = 1
        self.ever_confirmed = False

    @property
    def box(self):
        return self.kf.box

    def predict(self):
        self.kf.predict()
        self.age_since_update += 1
        self.age += 1

    def update(self, det, feature, overlap, quality_memory):
        self.kf.update(det.box, det.confidence)
        self.confidence = float(det.confidence)
        self.hits += 1
        self.age_since_update = 0
        if feature is not None:
            f = normalize_feature(feature)
            if self.memory is None:
                self.memory = f
            else:
                alpha = (np.clip(1.0 - .3 * det.confidence * (1.0 - overlap), .7, 1.)
                         if quality_memory else .90)
                self.memory = normalize_feature(alpha * self.memory + (1. - alpha) * f)


class LegacySRTracker:
    """Stable tracker: confirmed-active -> lost recovery -> new tentative tracks.

    - `selective`: OSNet extraction for ambiguity; optional one-time bootstrap
      and conservative lost-track extraction; no ResNeSt in this branch.
    - `full`: OSNet on every detection (ablation, *not* canonical StrongSORT).
    - `bot_full`: trained BoT/ResNeSt50 on every detection (research baseline).

    The baseline shares simplified tracker logic, not StrongSORT official ECC.
    """
    def __init__(self, extractor, mode='selective', iou_threshold=.05,
                 ars_threshold=.9, lambda_app=.65, max_age=40, min_hits=2,
                 quality_memory=True, chi2_threshold=CHI2_95_DOF4,
                 bootstrap_reid=True, recovery_reid=True,
                 recovery_similarity=.72, bootstrap_confidence=.45,
                 new_track_confidence=.5, match_threshold=.86,
                 max_tentative_age=1, debug=False):
        if mode not in ('selective', 'full', 'bot_full'):
            raise ValueError('mode must be selective, full, or bot_full')
        self.extractor = extractor
        self.mode = mode
        self.iou_threshold = float(iou_threshold)
        self.ars_threshold = float(ars_threshold)
        self.lambda_app = float(lambda_app)
        self.max_age = int(max_age)
        self.min_hits = int(min_hits)
        self.quality_memory = bool(quality_memory)
        self.chi2_threshold = float(chi2_threshold)
        self.bootstrap_reid = bool(bootstrap_reid)
        self.recovery_reid = bool(recovery_reid)
        self.recovery_similarity = float(recovery_similarity)
        self.bootstrap_confidence = float(bootstrap_confidence)
        self.new_track_confidence = float(new_track_confidence)
        self.match_threshold = float(match_threshold)
        self.max_tentative_age = int(max_tentative_age)
        self.debug = bool(debug)
        self.tracks = []
        self.next_id = 1
        self.frame_id = 0
        self.detection_count = 0
        self.reid_count = 0
        self.reid_calls = 0
        self.ambiguity_count = 0
        self.bootstrap_count = 0
        self.recovery_count = 0
        self.lost_matches = 0
        self.new_tracks_count = 0
        self.dropped_tentatives = 0
        self.last_events = []

    def _extract(self, frame, detections, indices):
        if not indices:
            return {}
        h, w = frame.shape[:2]
        images, valid = [], []
        for j in sorted(set(indices)):
            x1, y1, x2, y2 = [int(round(t)) for t in detections[j].box]
            x1, x2 = max(0, min(w, x1)), max(0, min(w, x2))
            y1, y2 = max(0, min(h, y1)), max(0, min(h, y2))
            if x2 <= x1 or y2 <= y1:
                continue
            images.append(frame[y1:y2, x1:x2].copy())
            valid.append(j)
        if not images:
            return {}
        features = self.extractor(images)
        if len(features) != len(valid):
            raise RuntimeError('Extractor returned wrong number of features')
        self.reid_count += len(valid)
        self.reid_calls += 1
        return {j: normalize_feature(v) for j, v in zip(valid, features)}

    @staticmethod
    def _overlap_other_dets(detections, j):
        return max((iou(detections[j].box, detections[k].box)
                    for k in range(len(detections)) if k != j), default=0.)

    @staticmethod
    def _center_separation(a, b):
        aa, bb = xyxy_to_xywh_center(a), xyxy_to_xywh_center(b)
        sx = max(10., (aa[2] + bb[2])*.5)
        sy = max(10., (aa[3] + bb[3])*.5)
        return float(np.hypot((aa[0]-bb[0])/sx, (aa[1]-bb[1])/sy))

    def _candidate(self, track, det):
        """Motion gate + safe spatial fallback for detector box jitter.

        The fallback is an experimental stabilization relative to strict proposal
        Chi-square gating. Never match obviously distant detections on a cheap gate.
        """
        md = track.kf.mahalanobis_squared(det.box, det.confidence)
        ov = iou(track.box, det.box)
        shape = ars(track.box, det.box)
        sep = self._center_separation(track.box, det.box)
        # Strict NSA gate, or very strong overlap if KF is overconfident.
        valid = ((md <= self.chi2_threshold and (ov >= .015 or sep <= .75))
                 or (ov >= .25 and sep <= .70 and shape >= .95))
        return md, ov, shape, sep, valid

    def _assign(self, track_indices, det_indices, matrix, limit):
        """Hungarian with explicit unmatched columns, not forced bad pairings."""
        n, m = len(track_indices), len(det_indices)
        if not n or not m:
            return []
        # Each track gets its own dummy column: bad matches are rejected inside
        # the optimization instead of being chosen then thrown away afterwards.
        augmented = np.full((n, m+n), 1e6, dtype=float)
        augmented[:, :m] = matrix
        for i in range(n):
            augmented[i, m+i] = limit
        rows, cols = linear_sum_assignment(augmented)
        return [(track_indices[i], det_indices[j]) for i, j in zip(rows, cols)
                if j < m and np.isfinite(matrix[i, j]) and matrix[i, j] < limit]

    def update(self, frame, detections):
        self.frame_id += 1
        self.detection_count += len(detections)
        self.last_events = []
        for track in self.tracks:
            track.predict()
        n, m = len(self.tracks), len(detections)
        motion = np.full((n, m), np.inf)
        overlap = np.zeros((n, m))
        shape = np.zeros((n, m))
        sep = np.full((n, m), np.inf)
        valid = np.zeros((n, m), dtype=bool)
        for i, tr in enumerate(self.tracks):
            for j, det in enumerate(detections):
                md, ov, a, d, good = self._candidate(tr, det)
                motion[i, j], overlap[i, j], shape[i, j], sep[i, j] = md, ov, a, d
                valid[i, j] = good

        # Pure proposal selective criterion only compares confirmed tracks
        # which pass Mahalanobis + IoU + ARS. (No appearance on easy frames.)
        if self.mode in ('full', 'bot_full'):
            requested = list(range(m))
        else:
            requested = []
            for j in range(m):
                candidates = sum(
                    self.tracks[i].hits >= self.min_hits and
                    motion[i, j] <= self.chi2_threshold and
                    overlap[i, j] >= self.iou_threshold and
                    shape[i, j] >= self.ars_threshold
                    for i in range(n))
                if candidates >= 2:
                    requested.append(j)
            self.ambiguity_count += len(requested)
        features = self._extract(frame, detections, requested)
        self.last_events.extend([('ambiguous', j) for j in requested])

        # 1) Current tracks first, 2) Lost confirmed tracks after. Ensures an
        # old ghost track cannot preempt the continuously observed person's ID.
        active = [i for i, tr in enumerate(self.tracks) if tr.age_since_update == 1]
        lost = [i for i, tr in enumerate(self.tracks)
                if tr.age_since_update > 1 and tr.ever_confirmed]
        matched_t, matched_d = set(), set()
        pairs = []
        if active and m:
            costs = np.full((len(active), m), np.inf)
            for ri, i in enumerate(active):
                track = self.tracks[i]
                for j in range(m):
                    if not valid[i, j]:
                        continue
                    # IoU always influences easy-frame assignments. It is less
                    # sensitive than a shrinking KF covariance to detector jitter.
                    md_norm = min(motion[i, j] / self.chi2_threshold, 1.)
                    motion_cost = .55 * (1. - overlap[i, j]) + .45 * md_norm
                    # Small penalty if tentative track has weaker history.
                    if not track.ever_confirmed and track.hits < self.min_hits:
                        motion_cost += .02
                    feat = features.get(j)
                    if feat is not None and track.memory is not None:
                        app = appearance_cost(track.memory, feat)
                        costs[ri, j] = ((1-self.lambda_app)*motion_cost +
                                        self.lambda_app*app)
                    else:
                        costs[ri, j] = motion_cost
            active_pairs = self._assign(active, list(range(m)), costs, self.match_threshold)
            pairs.extend(active_pairs)
            matched_t.update(i for i, _ in active_pairs)
            matched_d.update(j for _, j in active_pairs)

        # Recovery is deliberate exception to the main >=2 gate; only on
        # confirmed tracks actually lost for >=1 frame, with both nearby box
        # and sufficiently similar existing ReID memory.
        free_dets = [j for j in range(m) if j not in matched_d]
        if lost and free_dets and self.recovery_reid:
            proposals = []
            for i in lost:
                for j in free_dets:
                    if (shape[i,j] >= self.ars_threshold and sep[i,j] <= 1.6
                            and detections[j].confidence >= self.bootstrap_confidence):
                        proposals.append((i,j))
            if proposals:
                if self.recovery_reid and self.mode == 'selective':
                    to_extract = [j for _, j in proposals if j not in features]
                    features.update(self._extract(frame, detections, to_extract))
                    self.recovery_count += len(set(to_extract))
                recover_cost = np.full((len(lost),len(free_dets)), np.inf)
                for i,j in proposals:
                    track = self.tracks[i]
                    feat = features.get(j)
                    # Recovery without ReID is too risky; do NOT match.
                    if feat is None or track.memory is None:
                        continue
                    app = appearance_cost(track.memory, feat)
                    if app > 1.-self.recovery_similarity:
                        continue
                    penalty = .10 * sep[i,j] + .015 * (track.age_since_update - 1)
                    recover_cost[lost.index(i), free_dets.index(j)] = app + penalty
                recovery_pairs = self._assign(lost, free_dets, recover_cost, .42)
                pairs.extend(recovery_pairs)
                matched_t.update(i for i,_ in recovery_pairs)
                matched_d.update(j for _,j in recovery_pairs)
                self.lost_matches += len(recovery_pairs)
                self.last_events.extend([('recovered', self.tracks[i].id) for i,_ in recovery_pairs])

        for i, j in pairs:
            tr = self.tracks[i]
            tr.update(detections[j], features.get(j),
                      self._overlap_other_dets(detections, j), self.quality_memory)
            if tr.hits >= self.min_hits:
                tr.ever_confirmed = True

        # Avoid creating tentative ghosts for every low-confidence detector
        # flicker. ReID bootstrap is run only on newly instantiated tracks.
        new_dets = []
        for j, det in enumerate(detections):
            if j in matched_d or det.confidence < self.new_track_confidence:
                continue
            # If this unmatched detection is practically the same box as an
            # already matched one, do not spawn a duplicate track.
            if any(iou(det.box, detections[k].box) >= .85 for k in matched_d):
                continue
            new_dets.append(j)

        if self.bootstrap_reid and self.mode == 'selective':
            fresh = [j for j in new_dets if j not in features]
            boot = self._extract(frame, detections, fresh)
            features.update(boot)
            self.bootstrap_count += len(boot)
            self.last_events.extend([('bootstrap', j) for j in fresh])

        for j in new_dets:
            new_track = Track(self.next_id, detections[j], features.get(j))
            new_track.ever_confirmed = self.min_hits <= 1
            self.tracks.append(new_track)
            self.last_events.append(('new_id', self.next_id))
            self.new_tracks_count += 1
            self.next_id += 1

        survivors = []
        for tr in self.tracks:
            if tr.ever_confirmed:
                keep = tr.age_since_update <= self.max_age
            else:
                keep = tr.age_since_update <= self.max_tentative_age
            if keep:
                survivors.append(tr)
            elif not tr.ever_confirmed:
                self.dropped_tentatives += 1
        self.tracks = survivors

        return [(tr.id, tr.box.copy(), tr.confidence) for tr in self.tracks
                if tr.age_since_update == 0 and tr.ever_confirmed]

    def stats(self):
        return dict(mode=self.mode, frames=self.frame_id,
                    detections=self.detection_count,
                    reid_features=self.reid_count, reid_batches=self.reid_calls,
                    extraction_ratio_percent=100*self.reid_count/max(self.detection_count,1),
                    ambiguous_detections=self.ambiguity_count,
                    bootstrapped_features=self.bootstrap_count,
                    recovery_features=self.recovery_count,
                    successfully_recovered_tracks=self.lost_matches,
                    new_track_ids=self.new_tracks_count,
                    dropped_tentatives=self.dropped_tentatives,
                    bootstrap_reid=self.bootstrap_reid,
                    recovery_reid=self.recovery_reid)
