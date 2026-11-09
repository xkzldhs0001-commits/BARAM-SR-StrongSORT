"""SR tracker: original StrongSORT Track/Kalman; selective OSNet and custom association.

No upstream source edits. When no Re-ID is necessary, only motion/shape is used.
Model-specific behavior is isolated here; baselines remain native upstream code.
"""
from __future__ import annotations
from dataclasses import dataclass
from collections import defaultdict
import time
import numpy as np
from .geometry import (iou_tlwh,ars_tlwh,center_distance_normalized,
                       motion_distance,normalize,CHI2_95_4D,match_with_unmatched)

@dataclass
class SRConfig:
    iou_gate:float=.05
    ars_gate:float=.8
    proximity:float=1.4
    pre_crossing:float=1.65
    maximum_cost:float=.86
    motion_weight:float=.50
    appearance_weight:float=.55
    max_age:int=30
    min_hits:int=3
    new_track_conf:float=.35
    recovery_similarity:float=.78
    recovery_window:int=100
    recovery_max_distance:float=1.5
    appearance_refresh:int=0
    alpha_floor:float=.7
    quality_iou_ceiling:float=.45
    bootstrap_new:bool=True


class SelectiveTracker:
    def __init__(self,upstream,extractor,config=None):
        self.src=upstream
        self.extractor=extractor
        self.cfg=config or SRConfig()
        self.tracks=[]
        self.gallery={}
        self.features={}
        self.next_id=1
        self.frame=0
        self.stats=defaultdict(int)
        self.stats['reid_seconds']=0.
        self.stats['association_seconds']=0.
        self.ids_history={}

    def _fresh_features(self,image,detections,indices):
        indices=sorted(set(indices))
        if not indices:return {}
        boxes=[detections[j].tlwh for j in indices]
        t=time.perf_counter()
        features=self.extractor.extract(image,boxes)
        self.stats['reid_seconds']+=time.perf_counter()-t
        if len(features)!=len(indices):raise RuntimeError('Re-ID extractor produced incorrect batch length')
        self.stats['reid_batches']+=1
        self.stats['reid_features']+=len(indices)
        return {j:f for j,feature in zip(indices,features) if (f:=normalize(feature)) is not None}

    def _movement(self,detections):
        T=len(self.tracks);D=len(detections)
        m=np.full((T,D),float('inf'))
        if not T or not D:return m
        for i,tr in enumerate(self.tracks):
            for j,dt in enumerate(detections):m[i,j]=motion_distance(tr,dt)
        return m

    def _retired_recovery(self,dets,indices,feature_dict):
        """Strict appearance+spatial+age gallery recovery. Never blindly reuse an ID."""
        recovered={}
        for j in indices:
            feat=feature_dict.get(j)
            if feat is None:continue
            candidates=[]
            for tid,(vector,box,fr) in self.gallery.items():
                gap=self.frame-fr
                if not (0<gap<=self.cfg.recovery_window):continue
                sim=float(vector@feat)
                if sim<self.cfg.recovery_similarity:continue
                distance=center_distance_normalized(np.asarray(box).reshape(1,4),[dets[j].tlwh])[0,0]
                if distance>self.cfg.recovery_max_distance:continue
                candidates.append((sim-.05*distance,tid))
            if candidates:
                candidates.sort(reverse=True)
                if len(candidates)>1 and candidates[0][0]-candidates[1][0]<.06:continue
                tid=candidates[0][1]
                if tid not in recovered.values():recovered[j]=tid
        return recovered

    def process(self,image,detections,sequence=None,ecc=False):
        self.frame+=1
        self.stats['frames']+=1;self.stats['detections']+=len(detections)
        # Keep original strongsort motion/NSA update and camera compensation.
        for tr in self.tracks:
            if ecc:tr.camera_update(sequence,self.frame)
            tr.predict()
        start=time.perf_counter()
        n,m=len(self.tracks),len(detections)
        tbox=np.array([x.to_tlwh() for x in self.tracks],dtype=float).reshape(-1,4)
        dbox=np.array([x.tlwh for x in detections],dtype=float).reshape(-1,4)
        iou=iou_tlwh(tbox,dbox);ars=ars_tlwh(tbox,dbox)
        motion=self._movement(detections)
        proximity=center_distance_normalized(tbox,dbox)
        # Valid 4D chi-square motion and spatial/shape candidate gate.
        candidate=(motion<=CHI2_95_4D)&(ars>=self.cfg.ars_gate)&(
            (iou>=self.cfg.iou_gate)|(proximity<=self.cfg.proximity))
        ambiguous=np.zeros(m,dtype=bool)
        if n and m:
            # Many-to-one or one-to-many association ambiguity.
            ambiguous=(candidate.sum(axis=0)>=2) | np.any(candidate &
                              (candidate.sum(axis=1)[:,None]>=2),axis=0)
            # Early pre-crossing trigger while two current tracks are near a detection.
            near=(motion<=CHI2_95_4D)&(proximity<=self.cfg.pre_crossing)
            ambiguous|=near.sum(axis=0)>=2
        selection={j for j in range(m) if ambiguous[j]}
        self.stats['ambiguous_detections']+=len(selection)
        # Bootstrap appearance only for NEW, previously unseen, high-confidence tracks.
        if n and m:
            unmatched_pre=[j for j in range(m) if not candidate[:,j].any()]
        else:unmatched_pre=list(range(m))
        if self.cfg.bootstrap_new:
            selection.update(j for j in unmatched_pre if detections[j].confidence>=self.cfg.new_track_conf)
        # Reappearance/occlusion and optional periodic confidence-protective refresh.
        for j in range(m):
            nearest=np.where(candidate[:,j])[0] if n else []
            if len(nearest)==1:
                tr=self.tracks[int(nearest[0])]
                if tr.time_since_update>1 or (tr.track_id not in self.features):selection.add(j)
                if self.cfg.appearance_refresh and tr.age%self.cfg.appearance_refresh==0:selection.add(j)
            if j not in selection and self.gallery and j in unmatched_pre and detections[j].confidence>=self.cfg.new_track_conf:
                selection.add(j)
        self.stats['selected_detections']+=len(selection)
        self.stats['association_seconds']+=time.perf_counter()-start
        feats=self._fresh_features(image,detections,selection)
        start=time.perf_counter()
        # Normalize squared motion distance to [0,1] and IoU into same range;
        # NEVER add raw chi2 distance directly to cosine distance.
        cost=np.full((n,m),np.inf,dtype=float)
        for i,tr in enumerate(self.tracks):
            for j in range(m):
                if not candidate[i,j]:continue
                cm=self.cfg.motion_weight*min(1.,motion[i,j]/CHI2_95_4D)+(1.-self.cfg.motion_weight)*(1.-iou[i,j])
                cf=feats.get(j);mem=self.features.get(tr.track_id)
                if cf is not None and mem is not None:
                    app=np.clip((1.-float(mem@cf))/2,0,1)
                    if app>.40:continue  # reject dissimilar identity after crossing
                    cost[i,j]=self.cfg.appearance_weight*app+(1-self.cfg.appearance_weight)*cm
                else:
                    # Additional conservative penalty for motion-only ambiguous matches.
                    cost[i,j]=cm + (.08 if ambiguous[j] else 0.)
        # Use DeepSORT's matching-cascade ordering for SR: tracks observed most
        # recently get first choice, then older tracks claim what remains. The
        # official comparison uses the authors' --woC setting, which disables
        # this cascade in that baseline; keep this SR behavior configurable via
        # its own implementation rather than patching the upstream code.
        matches=[]
        unmatched_d=set(range(m))
        assigned_t=set()
        confirmed_ages=sorted({tr.time_since_update for tr in self.tracks
                               if tr.is_confirmed()})
        for age in confirmed_ages:
            group=[i for i,tr in enumerate(self.tracks)
                   if tr.is_confirmed() and tr.time_since_update==age]
            available=sorted(unmatched_d)
            if not group or not available:
                continue
            local,_,_=match_with_unmatched(
                cost[np.ix_(group,available)],self.cfg.maximum_cost)
            for local_i,local_j in local:
                ti,dj=group[local_i],available[local_j]
                matches.append((ti,dj));assigned_t.add(ti);unmatched_d.discard(dj)

        # Tentative tracks have no stable appearance history. Associate them
        # after confirmed tracks using the same gated motion/IoU cost.
        tentative=[i for i,tr in enumerate(self.tracks) if tr.is_tentative()]
        available=sorted(unmatched_d)
        if tentative and available:
            local,_,_=match_with_unmatched(
                cost[np.ix_(tentative,available)],self.cfg.maximum_cost)
            for local_i,local_j in local:
                ti,dj=tentative[local_i],available[local_j]
                matches.append((ti,dj));assigned_t.add(ti);unmatched_d.discard(dj)
        unmatched_t=[i for i in range(n) if i not in assigned_t]
        unmatched_d=sorted(unmatched_d)
        for i,j in matches:
            tr=self.tracks[i];det=detections[j]
            tr.mean,tr.covariance=tr.kf.update(tr.mean,tr.covariance,det.to_xyah(),det.confidence)
            tr.hits+=1;tr.time_since_update=0
            tr.scores.append(det.confidence)
            if tr.is_tentative() and tr.hits>=tr._n_init:
                tr.state=self.src['TrackState'].Confirmed
            vector=feats.get(j)
            if vector is not None:
                # The quality gate only affects appearance memory, never Kalman.
                maxoverlap=max((iou_tlwh([det.tlwh],[detections[k].tlwh])[0,0]
                                for k in range(m) if k!=j), default=0.)
                if maxoverlap<=self.cfg.quality_iou_ceiling or tr.track_id not in self.features:
                    alpha=np.clip(1.-.3*float(det.confidence)*(1.-maxoverlap),self.cfg.alpha_floor,1.)
                    prior=self.features.get(tr.track_id)
                    merged=vector if prior is None else normalize(alpha*prior+(1-alpha)*vector)
                    if merged is not None:
                        self.features[tr.track_id]=merged
                        tr.features=[merged.copy()]
        for i in unmatched_t:self.tracks[i].mark_missed()
        # Expire track embeddings into gallery for cautious re-ID when reappearing.
        for t in self.tracks:
            if t.is_deleted() and t.track_id in self.features:
                self.gallery[t.track_id]=(self.features[t.track_id].copy(),t.to_tlwh(),self.frame)
        # Expire stale gallery entries; no unlimited memory growth.
        self.gallery={k:v for k,v in self.gallery.items() if self.frame-v[2]<=self.cfg.recovery_window}
        self.tracks=[t for t in self.tracks if not t.is_deleted()]
        self.features={t.track_id:self.features[t.track_id] for t in self.tracks if t.track_id in self.features}
        recovery=self._retired_recovery(detections,unmatched_d,feats)
        for j in unmatched_d:
            if detections[j].confidence<self.cfg.new_track_conf:continue
            reused=recovery.get(j)
            tid=reused if reused is not None else self.next_id
            if reused is None:self.next_id+=1;self.stats['new_track_ids']+=1
            else:self.gallery.pop(reused,None);self.stats['recovered_tracks']+=1
            f=feats.get(j)
            tr=self.src['Track'](detections[j].to_xyah(),tid,self.cfg.min_hits,self.cfg.max_age,
                                 None,float(detections[j].confidence))
            if f is not None:
                tr.features=[f.copy()];self.features[tid]=f.copy()
            self.tracks.append(tr)
        self.stats['association_seconds']+=time.perf_counter()-start
        return [(t.track_id,t.to_tlwh().copy()) for t in self.tracks if t.is_confirmed() and t.time_since_update<=1]
