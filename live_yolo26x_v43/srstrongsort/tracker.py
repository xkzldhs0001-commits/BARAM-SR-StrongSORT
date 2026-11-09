"""BARAM v5 selective Re-ID tracker.

Proposal core: YOLO26n -> NSA Kalman -> chi2/IoU/ARS -> selective OSNet ->
normalized motion/appearance Hungarian -> quality-weighted EMA.

Explicit *experimental* extensions for ID continuity:
- pre-crossing selective ReID (nearby predicted confirmed tracks)
- retain confirmed lost identities for a bounded gallery TTL
- spatially relaxed, appearance-verified lost-ID recovery (never blind reuse)
- avoid corrupting appearance memory with occluded crops
- reject ambiguous recovery when multiple identities look equally plausible

Tracking while completely occluded cannot recover detector false negatives; tracks
are not output without detections. The longer gallery is not an AFLink or GSI
implementation. The optional BoT baseline shares this association logic and is
NOT the byte-identical, ECC-enabled official StrongSORT implementation.
"""
from __future__ import annotations
from dataclasses import dataclass
import time
import numpy as np
from scipy.optimize import linear_sum_assignment
from .kalman import NSAKalman
from .geometry import CHI2_95_DOF4, iou, ars, normalize_feature, appearance_cost, xyxy_to_xywh_center


@dataclass
class Detection:
    box: np.ndarray
    confidence: float


class Track:
    def __init__(self, track_id, det, feature=None, frame_id=0):
        self.id=int(track_id)
        self.kf=NSAKalman(det.box)
        self.memory=normalize_feature(feature) if feature is not None else None
        self.bank=[self.memory.copy()] if self.memory is not None else []
        self.last_feature_frame=frame_id if feature is not None else -1000000
        self.confidence=float(det.confidence)
        self.hits=1
        self.age_since_update=0
        self.age=1
        self.ever_confirmed=False

    @property
    def box(self):return self.kf.box

    def predict(self):
        # Reduce velocity drift for a track missing for many frames; preserve P.
        if self.age_since_update>8:
            self.kf.x[4:]*=.94
        self.kf.predict()
        self.age_since_update+=1
        self.age+=1

    def similarity(self, feature):
        if feature is None or self.memory is None:return -1.
        feat=normalize_feature(feature)
        candidates=[self.memory]+self.bank
        # Keep some robustness against pose/illumination differences.
        return float(max(np.dot(v,feat) for v in candidates))

    def update(self,det,feature,overlap,quality_memory,frame_id,reacquired=False):
        if reacquired and self.age_since_update>8:
            self.kf=NSAKalman(det.box)  # reset a stale motion estimate, retain ID and memory
        else:
            self.kf.update(det.box,det.confidence)
        self.confidence=float(det.confidence)
        self.hits+=1
        self.age_since_update=0
        if feature is not None and overlap<.40:
            f=normalize_feature(feature)
            if self.memory is None:self.memory=f
            else:
                alpha=(float(np.clip(1.-.3*det.confidence*(1.-overlap),.7,1.))
                       if quality_memory else .90)
                self.memory=normalize_feature(alpha*self.memory+(1.-alpha)*f)
            # Only store unoccluded, sufficiently confident views.
            if det.confidence>=.45 and overlap<.20:
                self.bank.append(f.copy());self.bank=self.bank[-8:]
            self.last_feature_frame=frame_id


class SRTracker:
    def __init__(self,extractor,mode='selective',iou_threshold=.05,
                 ars_threshold=.9,lambda_app=.65,max_age=40,min_hits=2,
                 quality_memory=True,chi2_threshold=CHI2_95_DOF4,
                 bootstrap_reid=True,recovery_reid=True,
                 recovery_similarity=.72,bootstrap_confidence=.45,
                 new_track_confidence=.5,match_threshold=.82,
                 max_tentative_age=1,debug=False,gating_policy='adaptive',
                 refresh_interval=0,crowd_overlap=.25,cost_iou_weight=.55,
                 recovery_max_separation=1.6,appearance_gate=.55,
                 suppress_weak_birth=True,cost_policy='normalized',
                 gallery_age=180,crossing_radius=1.45,
                 recovery_margin=.06,recovery_max_scale=2.5):
        if mode not in ('selective','full','bot_full'):raise ValueError('Invalid mode')
        if gating_policy not in ('adaptive','proposal'):raise ValueError('Invalid gating_policy')
        if cost_policy not in ('normalized','raw_proposal'):raise ValueError('Invalid cost_policy')
        self.extractor=extractor;self.mode=mode
        for key,value in locals().copy().items():
            if key not in ('self','extractor','mode','__class__'):
                setattr(self,key,value)
        self.tracks=[];self.next_id=1;self.frame_id=0
        self.detection_count=0;self.reid_count=0;self.reid_calls=0
        self.reid_seconds=0.;self.association_seconds=0.
        self.ambiguity_count=0;self.bootstrap_count=0;self.recovery_count=0
        self.refresh_count=0;self.lost_matches=0;self.new_tracks_count=0
        self.dropped_tentatives=0;self.long_recoveries=0;self.crossing_features=0
        self.last_events=[];self.last_associations=[]

    def _extract(self,frame,dets,indices):
        if not indices:return {}
        h,w=frame.shape[:2];crops=[];ids=[]
        for j in sorted(set(indices)):
            x1,y1,x2,y2=np.rint(dets[j].box).astype(int)
            x1=max(0,min(w,x1));x2=max(0,min(w,x2))
            y1=max(0,min(h,y1));y2=max(0,min(h,y2))
            if x2<=x1 or y2<=y1:continue
            crops.append(frame[y1:y2,x1:x2].copy());ids.append(j)
        if not crops:return {}
        start=time.perf_counter()
        feats=self.extractor(crops)
        try:
            import torch
            if torch.cuda.is_available():torch.cuda.synchronize()
        except ImportError:pass
        self.reid_seconds+=time.perf_counter()-start
        if len(feats)!=len(ids):raise RuntimeError('ReID extractor batch dimension mismatch')
        self.reid_count+=len(ids);self.reid_calls+=1
        return {j:normalize_feature(v) for j,v in zip(ids,feats)}

    @staticmethod
    def _overlap_other_dets(dets,j):
        return max((iou(dets[j].box,det.box) for k,det in enumerate(dets) if k!=j),default=0.)

    @staticmethod
    def _center_separation(a,b):
        aa,bb=xyxy_to_xywh_center(a),xyxy_to_xywh_center(b)
        sx=max(10.,(aa[2]+bb[2])*.5);sy=max(10.,(aa[3]+bb[3])*.5)
        return float(np.hypot((aa[0]-bb[0])/sx,(aa[1]-bb[1])/sy))

    def _candidate(self,track,det):
        md=track.kf.mahalanobis_squared(det.box,det.confidence)
        ov=iou(track.box,det.box);shape=ars(track.box,det.box)
        sep=self._center_separation(track.box,det.box)
        valid=(md<=self.chi2_threshold and (ov>=.015 or sep<=.75)) or (ov>=.25 and sep<=.70 and shape>=.95)
        return md,ov,shape,sep,valid

    @staticmethod
    def _assign(track_ids,det_ids,costs,limit):
        n,m=len(track_ids),len(det_ids)
        if not n or not m:return []
        # Only permit edges below the association threshold.  Do not rely
        # solely on post-Hungarian filtering: include unmatched dummy columns
        # so a rejected high-cost edge cannot consume a real detection.
        accepted=np.isfinite(costs)&(costs < limit)
        matrix=np.full((n,m+n),1e6,dtype=float)
        matrix[:,:m]=np.where(accepted,costs,1e6)
        for i in range(n):
            matrix[i,m+i]=limit
        rr,cc=linear_sum_assignment(matrix)
        return [(track_ids[i],det_ids[j]) for i,j in zip(rr,cc)
                if j<m and accepted[i,j]]

    def _motion_cost(self,md,ov):
        return self.cost_iou_weight*(1.-ov)+(1.-self.cost_iou_weight)*min(md/self.chi2_threshold,1.)

    def update(self,frame,detections):
        self.frame_id+=1;self.detection_count+=len(detections)
        self.last_events=[];self.last_associations=[]
        start=time.perf_counter()
        for tr in self.tracks:tr.predict()
        n,m=len(self.tracks),len(detections)
        md=np.full((n,m),np.inf);overlap=np.zeros((n,m));shape=np.zeros((n,m))
        sep=np.full((n,m),np.inf);valid=np.zeros((n,m),dtype=bool)
        for i,tr in enumerate(self.tracks):
            # No need to run chi-square for very old tracks: recovery via appearance.
            if tr.age_since_update>self.max_age:continue
            for j,det in enumerate(detections):
                a,b,c,d,e=self._candidate(tr,det)
                md[i,j]=a;overlap[i,j]=b;shape[i,j]=c;sep[i,j]=d;valid[i,j]=e
        confirmed=[i for i,t in enumerate(self.tracks) if t.ever_confirmed]
        active=[i for i,t in enumerate(self.tracks) if t.age_since_update==1]
        source={}
        if self.mode!='selective':
            source={j:'full' for j in range(m)}
        else:
            eligible=np.zeros((n,m),bool)
            for i in confirmed:
                if self.tracks[i].age_since_update>self.max_age:continue
                eligible[i]=(md[i]<=self.chi2_threshold)&(overlap[i]>=self.iou_threshold)&(shape[i]>=self.ars_threshold)
            for j in range(m):
                if eligible[:,j].sum()>=2:source[j]='proposal_ambiguity'
            if self.gating_policy=='adaptive':
                # Early crossing signal BEFORE the boxes overlap: two plausible
                # confirmed track predictions close to the same detection.
                for j,det in enumerate(detections):
                    if j in source:continue
                    close=[i for i in confirmed if self.tracks[i].age_since_update<=2 and
                           self._center_separation(self.tracks[i].box,det.box)<=self.crossing_radius]
                    if len(close)>=2:
                        source[j]='pre_crossing';continue
                    if any(eligible[i,j] and eligible[i].sum()>=2 for i in confirmed):
                        source[j]='reciprocal_ambiguity';continue
                    if eligible[:,j].any() and self._overlap_other_dets(detections,j)>=self.crowd_overlap:
                        source[j]='crowd_overlap';continue
                    if self.refresh_interval>0 and eligible[:,j].sum()==1:
                        i=int(np.flatnonzero(eligible[:,j])[0]);tr=self.tracks[i]
                        if self.frame_id-tr.last_feature_frame>=self.refresh_interval and det.confidence>=self.bootstrap_confidence and self._overlap_other_dets(detections,j)<.1:
                            source[j]='refresh'
        self.ambiguity_count+=sum(s not in ('refresh','full') for s in source.values())
        self.crossing_features+=sum(s=='pre_crossing' for s in source.values())
        features=self._extract(frame,detections,list(source))
        self.refresh_count+=sum(source.get(j)=='refresh' for j in features)
        self.last_events.extend((reason,j) for j,reason in source.items())
        # Track-Detection Hungarian for tracks updated in previous frame.
        costs=np.full((len(active),m),np.inf)
        for row,i in enumerate(active):
            tr=self.tracks[i]
            for j in range(m):
                if not valid[i,j]:continue
                motion=(float(md[i,j]) if self.cost_policy=='raw_proposal'
                        else self._motion_cost(md[i,j],overlap[i,j]))
                feat=features.get(j)
                if feat is not None and tr.memory is not None and source.get(j)!='refresh':
                    similarity=tr.similarity(feat)
                    if 1.-similarity>self.appearance_gate:continue
                    # Appearance dominant when two people can plausibly cross.
                    app_weight=max(self.lambda_app,.78) if source.get(j) in (
                        'pre_crossing','proposal_ambiguity','crowd_overlap','reciprocal_ambiguity') else self.lambda_app
                    motion=(1.-app_weight)*motion+app_weight*(1.-similarity)
                costs[row,j]=motion
        limit=self.chi2_threshold if self.cost_policy=='raw_proposal' else self.match_threshold
        pairs=self._assign(active,list(range(m)),costs,limit)
        matched_t={i for i,_ in pairs};matched_d={j for _,j in pairs}
        self.association_seconds+=time.perf_counter()-start

        # Re-acquire any confirmed lost identity, including from gallery beyond max_age.
        # Crucial: long-term matching is appearance-driven and has a uniqueness margin;
        # distance is NOT a hard gate for long absences.
        recovery_candidates=[i for i,tr in enumerate(self.tracks)
                             if i not in matched_t and tr.ever_confirmed and tr.memory is not None
                             and tr.age_since_update<=self.gallery_age]
        free_d=[j for j in range(m) if j not in matched_d]
        recover_pairs=set()
        if self.recovery_reid and recovery_candidates and free_d:
            # For long-lost identities use features even when the old predicted
            # location is far from the re-entering person.
            prospective=[]
            for j in free_d:
                for i in recovery_candidates:
                    tr=self.tracks[i]
                    if detections[j].confidence<self.bootstrap_confidence:continue
                    # Very broad size consistency check, independent of old center.
                    aa=xyxy_to_xywh_center(tr.box);bb=xyxy_to_xywh_center(detections[j].box)
                    ratio=max(aa[2]/max(bb[2],1),bb[2]/max(aa[2],1),aa[3]/max(bb[3],1),bb[3]/max(aa[3],1))
                    if ratio>self.recovery_max_scale:continue
                    if tr.age_since_update<=self.max_age:
                        if self._center_separation(tr.box,detections[j].box)>max(self.recovery_max_separation, .4+tr.age_since_update*.08):continue
                    prospective.append((i,j))
            if self.mode=='selective':
                extra=sorted({j for i,j in prospective if j not in features})
                features.update(self._extract(frame,detections,extra))
                self.recovery_count+=len(extra)
            sim=np.full((len(recovery_candidates),len(free_d)),-np.inf)
            ipos={i:k for k,i in enumerate(recovery_candidates)}
            jpos={j:k for k,j in enumerate(free_d)}
            for i,j in prospective:
                feat=features.get(j)
                if feat is not None:
                    sim[ipos[i],jpos[j]]=self.tracks[i].similarity(feat)
            rc=np.full_like(sim,np.inf)
            for ri,i in enumerate(recovery_candidates):
                for cj,j in enumerate(free_d):
                    score=sim[ri,cj]
                    if score<self.recovery_similarity:continue
                    # When two lost identities look very similar, refuse a risky
                    # ID claim instead of forcing a cross-person switch.
                    other=np.delete(sim[:,cj],ri)
                    if len(other) and np.max(other)>-np.inf and score-np.max(other)<self.recovery_margin:continue
                    # Additional row margin only if there is another plausible
                    # detection with a strong competing match.
                    row_other=np.delete(sim[ri],cj)
                    if len(row_other) and np.max(row_other)>=self.recovery_similarity and score-np.max(row_other)<self.recovery_margin:continue
                    tr=self.tracks[i]
                    penalty=.025*min(tr.age_since_update/float(self.gallery_age),1.)
                    rc[ri,cj]=1.-score+penalty
            recovered=self._assign(recovery_candidates,free_d,rc,1.-self.recovery_similarity+.03)
            for i,j in recovered:
                matched_t.add(i);matched_d.add(j);pairs.append((i,j));recover_pairs.add((i,j))
                self.lost_matches+=1
                if self.tracks[i].age_since_update>self.max_age:self.long_recoveries+=1
                self.last_events.append(('recovered',self.tracks[i].id))
        for i,j in pairs:
            tr=self.tracks[i]
            self.last_associations.append(dict(frame=self.frame_id,track_id=tr.id,detection_index=j,
                method='recovery' if (i,j) in recover_pairs else 'active',
                age_before_update=tr.age_since_update,
                iou=float(overlap[i,j]) if np.isfinite(md[i,j]) else None,
                mahalanobis=float(md[i,j]) if np.isfinite(md[i,j]) else None,
                appearance_similarity=tr.similarity(features.get(j)) if j in features and tr.memory is not None else None))
            tr.update(detections[j],features.get(j),self._overlap_other_dets(detections,j),
                      self.quality_memory,self.frame_id,reacquired=(i,j) in recover_pairs)
            if tr.hits>=self.min_hits:tr.ever_confirmed=True
        fresh=[]
        for j,det in enumerate(detections):
            if j in matched_d or det.confidence<self.new_track_confidence:continue
            if any(iou(det.box,detections[k].box)>=.85 for k in matched_d):continue
            if self.suppress_weak_birth and any(self.tracks[i].ever_confirmed and i in matched_t and
                    iou(self.tracks[i].box,det.box)>.8 for i in matched_t):continue
            fresh.append(j)
        if self.bootstrap_reid and self.mode=='selective':
            unseen=[j for j in fresh if j not in features]
            boot=self._extract(frame,detections,unseen)
            features.update(boot);self.bootstrap_count+=len(boot)
        for j in fresh:
            tr=Track(self.next_id,detections[j],features.get(j),self.frame_id)
            tr.ever_confirmed=self.min_hits<=1
            self.tracks.append(tr)
            self.last_events.append(('new_id',self.next_id))
            self.new_tracks_count+=1;self.next_id+=1
        survivors=[]
        for tr in self.tracks:
            keep=(tr.age_since_update<=self.gallery_age if tr.ever_confirmed and tr.memory is not None
                  else tr.age_since_update<=self.max_age if tr.ever_confirmed
                  else tr.age_since_update<=self.max_tentative_age)
            if keep:survivors.append(tr)
            elif not tr.ever_confirmed:self.dropped_tentatives+=1
        self.tracks=survivors
        return [(tr.id,tr.box.copy(),tr.confidence) for tr in self.tracks
                if tr.age_since_update==0 and tr.ever_confirmed]

    def stats(self):
        return dict(mode=self.mode,frames=self.frame_id,detections=self.detection_count,
                    reid_features=self.reid_count,reid_batches=self.reid_calls,
                    reid_seconds=self.reid_seconds,association_seconds=self.association_seconds,
                    extraction_ratio_percent=100*self.reid_count/max(self.detection_count,1),
                    ambiguous_detections=self.ambiguity_count,pre_crossing_features=self.crossing_features,
                    bootstrapped_features=self.bootstrap_count,recovery_features=self.recovery_count,
                    refreshed_features=self.refresh_count,recovered_tracks=self.lost_matches,
                    long_recoveries=self.long_recoveries,new_track_ids=self.new_tracks_count,
                    dropped_tentatives=self.dropped_tentatives,gallery_age=self.gallery_age,
                    recovery_similarity=self.recovery_similarity,gating_policy=self.gating_policy)
