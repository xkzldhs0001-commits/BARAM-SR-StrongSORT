"""V4.3 live-only ID bank: conservative overlap handling, novel-ID quarantine.

No new neural network. Takes already extracted OSNet embeddings. A previously
known ID is restored only by clean, repeated appearance evidence; an unknown
person gets a new ID only after clean evidence excludes *all* stored identities.
Unknown/occluded crops are deliberately not learned. This is not biometric
certainty and cannot guarantee zero switches under full occlusion.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from scipy.optimize import linear_sum_assignment


def unit(x):
    if x is None:
        return None
    a = np.asarray(x, dtype=np.float32).reshape(-1)
    if not a.size or not np.isfinite(a).all():
        return None
    n = float(np.linalg.norm(a))
    return a / n if n > 1e-10 else None


@dataclass
class Record:
    display_id: int
    seconds: float
    frame: int
    bbox: tuple
    templates: list = field(default_factory=list)
    observations: int = 0


@dataclass
class Pending:
    seen: int = 0
    clean_seen: int = 0
    last_frame: int = -1
    last_clean_frame: int = -1
    candidate: int | None = None
    streak: int = 0
    samples: list = field(default_factory=list)


class StableIdentityBank:
    def __init__(self, ttl_seconds=180., min_similarity=.72, multi_similarity=.80,
                 margin=.09, max_shape_ratio=4., max_gap_seconds=1.5,
                 max_center_distance=1.4, max_templates=8,
                 confirm_frames=3, swap_confirm_frames=3,
                 max_verify_frames=9, binding_hold_seconds=5.0, max_records=96):
        self.ttl = float(ttl_seconds)
        self.threshold = float(min_similarity)
        self.multi_threshold = float(multi_similarity)
        self.margin = float(margin)
        self.max_shape_ratio = float(max_shape_ratio)
        self.max_gap = float(max_gap_seconds)  # CLI compatibility; not used for ID authorization
        self.max_dist = float(max_center_distance)
        self.max_templates = max(1, int(max_templates))
        self.confirm_frames = max(2, int(confirm_frames))
        self.swap_confirm_frames = max(2, int(swap_confirm_frames))
        self.max_verify_frames = max(self.confirm_frames, int(max_verify_frames))
        self.binding_hold_seconds = float(binding_hold_seconds)
        self.max_records = max(2, int(max_records))
        self.records: dict[int, Record] = {}
        self.bindings: dict[int, int] = {}
        self.binding_last_seen: dict[int, float] = {}
        self.pending: dict[int, Pending] = {}
        self.suspect: dict[int, int] = {}
        self._suspect_target: dict[int, int | None] = {}
        self._swap_key = None
        self._swap_votes = 0
        self.occluded_internals = set()
        self.provisional_internals = set()
        self.next_id = 1
        self.total_restores = 0
        self.total_new = 0
        self.total_spatial = 0
        self.total_conflicts = 0
        self.total_corrected_swaps = 0
        self.total_pending_frames = 0
        self.total_occluded_frames = 0
        self.total_new_suppressed = 0

    def _appearance(self, f, rec):
        if f is None or not rec.templates:
            return None
        scores = [float(np.dot(f, t)) for t in rec.templates if t.shape == f.shape]
        if not scores:
            return None
        scores.sort(reverse=True)
        return .75*scores[0] + .25*float(np.mean(scores[:min(3, len(scores))]))

    def _threshold(self):
        # Count enrolled people, not currently visible people.
        return self.threshold if len(self.records) <= 1 else self.multi_threshold

    def _shape_ok(self, box, old):
        ra = max(1., float(box[2])) / max(1., float(box[3]))
        rb = max(1., float(old[2])) / max(1., float(old[3]))
        ratio = ra / rb
        return 1./self.max_shape_ratio <= ratio <= self.max_shape_ratio

    def _learn(self, rec, f, reliable):
        if not reliable or f is None:
            return
        score = self._appearance(f, rec)
        if score is not None and score < max(.65, self._threshold()-.10):
            return
        if score is None or score < .98:
            rec.templates.append(f.copy())
            if len(rec.templates) > self.max_templates:
                # Preserve the original clean enrollment and newest poses.
                rec.templates = (([rec.templates[0]] + rec.templates[-(self.max_templates-1):])
                                 if self.max_templates > 1 else rec.templates[:1])

    @staticmethod
    def _probe(p):
        return unit(np.mean(p.samples, axis=0)) if p.samples else None

    def _bind(self, internal, did, secs):
        for old, old_did in list(self.bindings.items()):
            if old != internal and old_did == did:
                self.bindings.pop(old, None)
                self.binding_last_seen.pop(old, None)
                self.suspect.pop(old, None)
                self._suspect_target.pop(old, None)
        self.bindings[internal] = did
        self.binding_last_seen[internal] = secs
        self.pending.pop(internal, None)
        self.suspect.pop(internal, None)
        self._suspect_target.pop(internal, None)

    def _new_record(self, iid, f, box, secs, frame, mapping, events, method):
        did = self.next_id
        self.next_id += 1
        rec = Record(did, secs, frame, box)
        if f is not None:
            rec.templates.append(f.copy())
        self.records[did] = rec
        self._bind(iid, did, secs)
        mapping[iid] = did
        self.total_new += 1
        events.append(dict(event='new', frame=frame, internal_id=iid,
                           display_id=did, similarity=None,
                           absent_seconds=None, method=method))
        return did

    def _restored(self, iid, did, score, old, box, secs, frame, mapping, events,
                  method='appearance_confirmed'):
        rec = self.records[did]
        gap = max(0., secs - rec.seconds)
        self._bind(iid, did, secs)
        mapping[iid] = did
        self.total_restores += 1
        if old is not None and old != did:
            self.total_corrected_swaps += 1
        events.append(dict(event='restored', frame=frame, internal_id=iid,
                           display_id=did, similarity=round(float(score), 4),
                           absent_seconds=round(gap, 2), method=method))

    def _coherent(self, p):
        """Reject inconsistent (mixed-body) probe galleries before creating IDs."""
        if len(p.samples) < 3:
            return False
        f = self._probe(p)
        return f is not None and all(float(np.dot(f, s)) >= .80 for s in p.samples)

    def update(self, tracks, seconds, frame):
        """Return ({track_id: display_id | 'VERIFY'}, events).

        Input: internal_id, bbox TLWH, feature (current-frame only), learnable,
        and optional occluded=True. Occulted/weak crops can keep existing ID
        labels provisional, but can never train or independently confirm IDs.
        """
        secs, frame = float(seconds), int(frame)
        obs = []
        seen = set()
        for t in tracks:
            iid = int(t['internal_id'])
            if iid in seen:
                raise ValueError('duplicate internal track ID')
            seen.add(iid)
            f = unit(t.get('feature'))
            box = tuple(map(float, t['bbox']))
            if len(box) != 4:
                raise ValueError('bbox must be TLWH')
            occ = bool(t.get('occluded', False))
            clean = bool(t.get('learnable', True)) and not occ and f is not None
            obs.append((iid, f, box, clean, occ))

        self.occluded_internals = {iid for iid, f, box, clean, occ in obs if occ}
        self.provisional_internals = set(self.occluded_internals)
        self.total_occluded_frames += len(self.occluded_internals)

        # Keep internal-track continuity through several seconds of occlusion;
        # the original tracker normally uses monotonic track IDs.
        for iid, last in list(self.binding_last_seen.items()):
            if secs - last > self.binding_hold_seconds:
                self.bindings.pop(iid, None)
                self.binding_last_seen.pop(iid, None)
                self.suspect.pop(iid, None)
                self._suspect_target.pop(iid, None)
        for iid, p in list(self.pending.items()):
            if iid not in seen and frame - p.last_frame > 30:
                self.pending.pop(iid, None)

        active = {self.bindings[iid] for iid in seen if iid in self.bindings}
        for did, rec in list(self.records.items()):
            if did not in active and secs-rec.seconds > self.ttl:
                self.records.pop(did, None)
        if len(self.records) > self.max_records:
            inactive = sorted((r for r in self.records.values() if r.display_id not in active),
                              key=lambda r:r.seconds)
            for rec in inactive[:max(0, len(self.records)-self.max_records)]:
                self.records.pop(rec.display_id, None)
        for iid, did in list(self.bindings.items()):
            if did not in self.records:
                self.bindings.pop(iid, None)
                self.binding_last_seen.pop(iid, None)

        mapping, events = {}, []
        bound = [(iid, f, box, clean, occ, self.bindings[iid])
                 for iid, f, box, clean, occ in obs if iid in self.bindings]

        # Atomic exchange: when two tracked people switch tracker internals,
        # correct both DISPLAY IDs simultaneously rather than detaching one and
        # accidentally allocating a new ID. Requires independent clean frames.
        clean_bound = [(iid, f, did) for iid, f, box, clean, occ, did in bound if clean]
        swap_done = set()
        if len(clean_bound) >= 2:
            refs = [did for iid, f, did in clean_bound]
            scores = np.full((len(clean_bound), len(refs)), -1., dtype=np.float32)
            for i, (iid, f, _) in enumerate(clean_bound):
                for j, did in enumerate(refs):
                    s = self._appearance(f, self.records[did])
                    if s is not None:
                        scores[i, j] = s
            rows, cols = linear_sum_assignment(-scores)
            proposal = {clean_bound[i][0]:refs[j] for i, j in zip(rows, cols)}
            changed = [iid for iid, f, old in clean_bound if proposal.get(iid) != old]
            stable = bool(changed)
            for i, (iid, f, old) in enumerate(clean_bound):
                if iid not in changed:
                    continue
                j = refs.index(proposal[iid])
                best = float(scores[i, j])
                own = float(scores[i, refs.index(old)])
                next_best = max((float(scores[i, k]) for k in range(len(refs)) if k != j),
                                default=-1.)
                if best < self._threshold() or best-own < max(.12, self.margin) or best-next_best < self.margin:
                    stable = False
                    break
            if stable:
                key = tuple(sorted(proposal.items()))
                self._swap_votes = self._swap_votes+1 if self._swap_key == key else 1
                self._swap_key = key
                self.provisional_internals.update(changed)
                if self._swap_votes >= self.swap_confirm_frames:
                    old_map = {iid:self.bindings[iid] for iid in changed}
                    # Reassign atomically: no temporary duplicate ownership.
                    for iid in changed:
                        self.bindings.pop(iid, None)
                        self.binding_last_seen.pop(iid, None)
                    for iid in changed:
                        target = proposal[iid]
                        f = next(ff for ti, ff, _ in clean_bound if ti == iid)
                        box = next(bb for ti, ff, bb, cc, oo, dd in bound if ti == iid)
                        self._restored(iid, target,
                                       self._appearance(f, self.records[target]),
                                       old_map[iid], box, secs, frame, mapping,
                                       events, method='atomic_swap')
                        swap_done.add(iid)
                    self._swap_votes, self._swap_key = 0, None
            else:
                self._swap_votes, self._swap_key = 0, None
        else:
            self._swap_votes, self._swap_key = 0, None

        # Bound person: don't learn ambiguous appearance or switch on a single
        # crop. If clean evidence contradicts the old binding over several
        # frames, relabel directly to a FREE known ID (without issuing a new ID).
        for iid, f, box, clean, occ, _ in bound:
            if iid in swap_done or iid not in self.bindings:
                continue
            did = self.bindings[iid]
            own = self._appearance(f, self.records[did]) if clean else None
            alternatives = sorted([(s, other) for other, rec in self.records.items() if other != did
                                   for s in [self._appearance(f, rec)] if s is not None], reverse=True) if clean else []
            alt_score, alt_did = alternatives[0] if alternatives else (-1., None)
            good_rival = bool(own is not None and alt_score >= self._threshold() and
                              alt_score-own >= max(.12, self.margin))
            unknown = bool(own is not None and own < .43 and not good_rival)
            suspect = good_rival or unknown
            if clean:
                target = alt_did if good_rival else -1 if unknown else None
                if suspect:
                    self.suspect[iid] = self.suspect.get(iid, 0)+1 if self._suspect_target.get(iid) == target else 1
                    self._suspect_target[iid] = target
                    self.total_conflicts += 1
                else:
                    self.suspect.pop(iid, None)
                    self._suspect_target.pop(iid, None)
            else:
                # Missing clean evidence does NOT confirm a switch.
                target = self._suspect_target.get(iid)
            if self.suspect.get(iid, 0) >= 2:
                self.provisional_internals.add(iid)
            if clean and self.suspect.get(iid, 0) >= self.swap_confirm_frames:
                target = self._suspect_target.get(iid)
                currently_occupied = {self.bindings[x] for x in seen if x != iid and x in self.bindings}
                if target is not None and target >= 0 and target not in currently_occupied:
                    old = did
                    self._restored(iid, target, alt_score, old, box, secs,
                                   frame, mapping, events, method='direct_clean_switch')
                    continue
                if target == -1:
                    self.bindings.pop(iid, None)
                    self.binding_last_seen.pop(iid, None)
                    self.suspect.pop(iid, None)
                    self._suspect_target.pop(iid, None)
                    self.pending[iid] = Pending()
                    mapping[iid] = 'VERIFY'
                    continue
            mapping[iid] = did
            self.binding_last_seen[iid] = secs

        # Collect clean probes for new/re-entered trackers only. A partially
        # occluded or weak box NEVER causes a new display ID on a timeout.
        unbound = [(iid, f, box, clean, occ) for iid, f, box, clean, occ in obs if iid not in mapping]
        for iid, f, box, clean, occ in unbound:
            p = self.pending.setdefault(iid, Pending())
            if p.last_frame != frame:
                p.seen += 1
                p.last_frame = frame
            if clean:
                if p.last_clean_frame >= 0 and frame - p.last_clean_frame > 10:
                    p.samples.clear()
                    p.streak = 0
                p.samples.append(f.copy())
                if len(p.samples) > 5:
                    p.samples.pop(0)
                p.clean_seen += 1
                p.last_clean_frame = frame

        occupied = {did for iid, did in mapping.items() if isinstance(did, int)}
        available = sorted((rec for did, rec in self.records.items() if did not in occupied),
                           key=lambda r:r.display_id)
        candidates = {}
        if unbound and available:
            n, m = len(unbound), len(available)
            scores = np.full((n, m), -np.inf, dtype=np.float32)
            threshold = self._threshold()
            for i, (iid, f, box, clean, occ) in enumerate(unbound):
                p = self.pending[iid]
                if not clean or not self._coherent(p):
                    continue
                probe = self._probe(p)
                for j, rec in enumerate(available):
                    if not self._shape_ok(box, rec.bbox):
                        continue
                    sim = self._appearance(probe, rec)
                    if sim is not None and sim >= threshold:
                        scores[i, j] = sim
            cost = np.full((n, m+n), 1e4, dtype=np.float32)
            cost[:, :m] = np.where(np.isfinite(scores), -scores, 1e4)
            for i in range(n):
                cost[i, m+i] = -threshold + 1e-6
            rows, cols = linear_sum_assignment(cost)
            for i, j in zip(rows, cols):
                if j >= m or not np.isfinite(scores[i,j]):
                    continue
                iid, f, box, clean, occ = unbound[i]
                p = self.pending[iid]
                probe = self._probe(p)
                score = float(scores[i,j])
                rivals = ([float(scores[i,k]) for k in range(m) if k!=j and np.isfinite(scores[i,k])]
                          + [float(scores[k,j]) for k in range(n) if k!=i and np.isfinite(scores[k,j])])
                # Compare against *all* records, including IDs already visible.
                # This avoids taking the last dormant ID just because the best
                # matching appearance is temporarily occupied by another track.
                for other, rec in self.records.items():
                    if other != available[j].display_id:
                        s = self._appearance(probe, rec)
                        if s is not None:
                            rivals.append(s)
                if rivals and score-max(rivals) < self.margin:
                    continue
                candidates[iid] = (available[j].display_id, score)

        for iid, f, box, clean, occ in unbound:
            p = self.pending[iid]
            choice = candidates.get(iid)
            if choice is not None and clean:
                target, score = choice
                if p.candidate != target or (p.last_clean_frame >= 0 and
                                              len(p.samples) == 1 and p.clean_seen > 1):
                    p.streak = 0
                p.streak += 1
                p.candidate = target
            elif clean:
                p.candidate = None
                p.streak = 0
            # If crop missing/overlapping, neither advance nor finalize streak.
            if choice is not None and clean and p.streak >= self.confirm_frames:
                target, score = choice
                self._restored(iid, target, score, None, box, secs, frame,
                               mapping, events)
                occupied.add(target)
                continue

            # CRITICAL V4.2 fix: never use 'all IDs occupied', three elapsed
            # frames or max VERIFY timeout alone as a reason to create a new ID.
            # Demand repeated clean samples coherent with each other AND
            # distinguishable from every registered identity.
            novel = False
            if clean and p.clean_seen >= max(5, self.confirm_frames+2) and \
                    p.seen >= self.max_verify_frames and self._coherent(p) and \
                    choice is None:
                probe = self._probe(p)
                similarities = [s for r in self.records.values()
                                for s in [self._appearance(probe, r)] if s is not None]
                # More conservative than restore: borderline / same-looking
                # probes remain VERIFY instead of multiplying display IDs.
                known_limit = max(.55, self._threshold() - .12)
                novel = not similarities or max(similarities) < known_limit
                if not novel:
                    self.total_new_suppressed += 1
            if novel or (not self.records and clean):
                self._new_record(iid, self._probe(p), box, secs, frame,
                                 mapping, events, 'new_clean_verified' if novel else 'new')
                occupied.add(mapping[iid])
            else:
                mapping[iid] = 'VERIFY'
                self.total_pending_frames += 1

        # Update geometry and appearance only after a clean, unambiguous match.
        # This keeps mixed appearance crops from poisoning the long-term bank.
        for iid, f, box, clean, occ in obs:
            did = mapping.get(iid)
            if not isinstance(did, int) or did not in self.records:
                continue
            rec = self.records[did]
            rec.seconds, rec.frame = secs, frame
            rec.observations += 1
            if clean and iid not in self.provisional_internals:
                rec.bbox = box
                own = self._appearance(f, rec)
                other = max((s for rid, r in self.records.items() if rid != did
                             for s in [self._appearance(f, r)] if s is not None), default=-1.)
                if own is None or own >= other + .05:
                    self._learn(rec, f, True)

        return mapping, events


AppearanceIdentityBank = StableIdentityBank
