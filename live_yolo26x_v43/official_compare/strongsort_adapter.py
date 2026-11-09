"""Official StrongSORT reference bridge + BARAM selective modifications.

Upstream code is cloned separately and loaded unmodified. A compatibility shim
replaces only a removed scikit-learn import. Shared opts are set explicitly.
"""
from __future__ import annotations
import copy
import sys
import types
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[1]
CHI2_95_4 = 9.4877


def prepare_official(repo: Path, ema_alpha=.9, mc_lambda=.98):
    repo = Path(repo).resolve()
    if not (repo / 'deep_sort' / 'tracker.py').exists():
        raise FileNotFoundError('Official StrongSORT checkout missing: '+str(repo))
    # The official 2022 source imports sklearn.utils.linear_assignment_
    # (removed by modern sklearn). Adapt that import without editing the
    # upstream matching algorithm or its thresholds.
    shim = types.ModuleType('sklearn.utils.linear_assignment_')
    def linear_assignment(cost):
        arr=np.asarray(cost)
        if not arr.size: return np.empty((0, 2), dtype=int)
        r,c=linear_sum_assignment(arr)
        return np.stack((r,c),axis=1)
    shim.linear_assignment=linear_assignment
    sys.modules['sklearn.utils.linear_assignment_']=shim
    opt_module=types.ModuleType('opts')
    opt_module.opt=types.SimpleNamespace(
        BoT=True, ECC=True, NSA=True, EMA=True, MC=True, woC=True,
        AFLink=False, GSI=False, EMA_alpha=ema_alpha, MC_lambda=mc_lambda,
        ecc={}, max_cosine_distance=.45, nn_budget=1)
    sys.modules['opts']=opt_module
    if str(repo) not in sys.path: sys.path.insert(0,str(repo))
    from deep_sort.tracker import Tracker
    from deep_sort.track import Track, TrackState
    from deep_sort.nn_matching import NearestNeighborDistanceMetric
    from deep_sort.detection import Detection
    return types.SimpleNamespace(Tracker=Tracker,Track=Track, TrackState=TrackState,
        Metric=NearestNeighborDistanceMetric,Detection=Detection,opt=opt_module.opt)


def iou_xywh(a,b):
    x1=max(float(a[0]),float(b[0])); y1=max(float(a[1]),float(b[1]))
    x2=min(float(a[0]+a[2]),float(b[0]+b[2])); y2=min(float(a[1]+a[3]),float(b[1]+b[3]))
    area=max(0.,x2-x1)*max(0.,y2-y1)
    return float(area/max(1e-6,float(a[2]*a[3]+b[2]*b[3]-area)))


def ars_xywh(a,b):
    aa=np.arctan(max(1e-5,a[2])/max(1e-5,a[3]))
    bb=np.arctan(max(1e-5,b[2])/max(1e-5,b[3]))
    return float(1.-4./(np.pi*np.pi)*(aa-bb)**2)


def unit(f):
    if f is None: return None
    a=np.asarray(f,dtype=np.float32).reshape(-1)
    norm=float(np.linalg.norm(a))
    return a/norm if norm>1e-10 else None


def make_sr_classes(api, lambda_app=.65, gate_iou=.05, gate_ars=.9,
                    match_threshold=.82, reid_similarity=.72,
                    gallery_age=180, pre_crossing_iou=.08):
    """SR adds selective features and quality memory update; baseline untouched.

    Motion Mahalanobis threshold is 9.4877 (4 dof).  Normalized weighted
    costs are an experimental implementation choice; tune on train split.
    """
    BaseTrack=api.Track
    class SRTrack(BaseTrack):
        def update(self, detection):
            self.mean,self.covariance=self.kf.update(
                self.mean,self.covariance,detection.to_xyah(),detection.confidence)
            feat=unit(detection.feature)
            if feat is not None:
                alpha=float(np.clip(1.-.3*detection.confidence*(1.-getattr(detection,'occlusion',0.)),.7,1.))
                if self.features:
                    merged=unit(alpha*unit(self.features[-1])+(1.-alpha)*feat)
                    self.features=[merged if merged is not None else feat]
                else: self.features=[feat]
            self.hits+=1; self.time_since_update=0
            if self.state==api.TrackState.Tentative and self.hits>=self._n_init:
                self.state=api.TrackState.Confirmed

    class SRTracker(api.Tracker):
        def __init__(self):
            super().__init__(api.Metric('cosine',.45,budget=1),max_age=30,n_init=3)
            self.frame_number=0
            self.gallery={}
            self.stats=dict(reid_features=0,reid_batches=0,bootstraps=0,
                            ambiguous=0, cross_preemptive=0, gallery_recoveries=0,
                            new_track_ids=0)

        def select_reid(self, tlwhs, confs):
            """BARAM_VECTOR_GATE_OPT_V1: identical thresholds and decisions; batch gates."""
            chosen=set(); active=[t for t in self.tracks if not t.is_deleted()]
            predicted=[t.to_tlwh() for t in active]
            crowded_tracks=set()
            for i,box in enumerate(predicted):
                for j in range(i+1,len(predicted)):
                    if iou_xywh(box,predicted[j])>=pre_crossing_iou:
                        crowded_tracks.update((i,j))
            # Before: one Kalman gating_distance() call PER (track, detection).
            # After:  one call PER track on the SAME detection measurement set.
            measurements=(np.stack([xyah(box) for box in tlwhs])
                          if len(tlwhs) else np.empty((0,4),dtype=float))
            gates=([t.kf.gating_distance(t.mean,t.covariance,measurements)
                    for t in active] if len(measurements) else [])
            for j,(bbox,conf) in enumerate(zip(tlwhs,confs)):
                candidates=[]
                for i,t in enumerate(active):
                    d2=float(gates[i][j])
                    if d2<=CHI2_95_4:
                        if iou_xywh(predicted[i],bbox)>=gate_iou and ars_xywh(predicted[i],bbox)>=gate_ars:
                            candidates.append(i)
                if len(candidates)!=1:
                    chosen.add(j)
                    if len(candidates)>1:self.stats['ambiguous']+=1
                    else:self.stats['bootstraps']+=1
                elif candidates[0] in crowded_tracks:
                    chosen.add(j);self.stats['cross_preemptive']+=1
                elif active[candidates[0]].time_since_update>1:
                    chosen.add(j)
                elif iou_xywh(predicted[candidates[0]],bbox)<.18:
                    chosen.add(j)
            return sorted(chosen)

        def update(self,detections):
            self.frame_number+=1
            # Expire long-term archived identity memory.
            self.gallery={tid:(t,stamp) for tid,(t,stamp) in self.gallery.items()
                          if self.frame_number-stamp<=gallery_age}
            matches,unmatched_tracks,unmatched_dets=self._match(detections)
            for ti,di in matches:self.tracks[ti].update(detections[di])
            for ti in unmatched_tracks:self.tracks[ti].mark_missed()
            # Match unclaimed detections to archived IDs only with a reliable
            # appearance embedding. Never blindly recycle an ID.
            still=[]
            for di in unmatched_dets:
                det=detections[di]; f=unit(det.feature)
                matches_gallery=[]
                if f is not None:
                    for tid,(lost,_) in self.gallery.items():
                        if not lost.features:continue
                        appearance=unit(lost.features[-1]);
                        if appearance is None or appearance.shape!=f.shape:continue
                        sim=float(np.dot(appearance,f))
                        if sim>=reid_similarity: matches_gallery.append((sim,tid))
                matches_gallery.sort(reverse=True)
                if matches_gallery and (len(matches_gallery)==1 or
                    matches_gallery[0][0]-matches_gallery[1][0]>=.06):
                    _,tid=matches_gallery[0]
                    track,_=self.gallery.pop(tid)
                    track.state=api.TrackState.Confirmed
                    track.mean,track.covariance=track.kf.initiate(det.to_xyah())
                    track.time_since_update=0
                    track.update(det)
                    self.tracks.append(track)
                    self.stats['gallery_recoveries']+=1
                else:still.append(di)
            for di in still:self._initiate_track(detections[di])
            for t in self.tracks:
                if t.is_deleted() and t.features and t.track_id not in self.gallery:
                    self.gallery[t.track_id]=(t,self.frame_number)
            self.tracks=[t for t in self.tracks if not t.is_deleted()]
            # Appearance metric state is retained across bypass frames.
            # Selective Re-ID can confirm a motion-only track with no embedding.
            # Never list such an ID as active in an appearance-sample dictionary.
            ff=[]; targets=[]
            for t in self.tracks:
                if t.is_confirmed() and t.features:
                    embedding=unit(t.features[-1])
                    if embedding is not None:
                        ff.append(embedding);targets.append(t.track_id)
            self.metric.partial_fit(np.asarray(ff),np.asarray(targets),targets)

        def _initiate_track(self,detection):
            f=unit(detection.feature)
            self.tracks.append(SRTrack(detection.to_xyah(),self._next_id,
                self.n_init,self.max_age, f,detection.confidence))
            self._next_id+=1;self.stats['new_track_ids']+=1

        def _match(self,detections):
            from scipy.optimize import linear_sum_assignment
            n=len(self.tracks);m=len(detections)
            if n==0 or m==0:return [],list(range(n)),list(range(m))
            cost=np.full((n,m),1e5,dtype=float)
            # Reuse the same detections across tracks; unchanged association logic.
            measurements=np.stack([d.to_xyah() for d in detections])
            for i,t in enumerate(self.tracks):
                if t.is_deleted():continue
                d2=t.kf.gating_distance(t.mean,t.covariance,measurements)
                for j,d in enumerate(detections):
                    if not np.isfinite(d2[j]) or d2[j]>CHI2_95_4: continue
                    mi=float(d2[j]/CHI2_95_4)
                    geo=(1.-.25)*mi+.25*(1.-iou_xywh(t.to_tlwh(),d.tlwh))
                    f=unit(d.feature)
                    if f is not None and t.features:
                        old=unit(t.features[-1]);
                        if old is not None and old.shape==f.shape:
                            app=1.-float(np.dot(old,f))
                            cost[i,j]=(1.-lambda_app)*geo+lambda_app*app
                        else:cost[i,j]=geo
                    else:cost[i,j]=geo
            # Include explicit unmatched choices; avoid forcing false matches.
            dummy=np.full((n,n),match_threshold+1e-5)
            r,c=linear_sum_assignment(np.hstack([cost,dummy]))
            matches=[(int(i),int(j)) for i,j in zip(r,c)
                     if j<m and cost[i,j]<=match_threshold]
            used_t={i for i,_ in matches};used_d={j for _,j in matches}
            return matches,[i for i in range(n) if i not in used_t],\
                [j for j in range(m) if j not in used_d]

    return SRTracker


def xyah(tlwh):
    x,y,w,h=map(float,tlwh)
    return np.array([x+w/2.,y+h/2.,w/max(h,1e-6),h],dtype=float)


class SRDetection:
    def __init__(self,tlwh,confidence,feature=None,occlusion=0.):
        self.tlwh=np.asarray(tlwh,dtype=float)
        self.confidence=float(confidence)
        self.feature=unit(feature)
        self.occlusion=float(occlusion)
    def to_xyah(self):return xyah(self.tlwh)
    def to_tlbr(self):
        arr=self.tlwh.copy();arr[2:]+=arr[:2];return arr


def make_ecc_matrix(previous_bgr,current_bgr,scale=.3,iterations=35):
    """ECC motion warp from previous frame coordinates to current frame.

    Both tracker modes consume this identical 3x3 transform. On ECC failure,
    preserve tracks with identity transform rather than using fabricated motion.
    """
    import cv2
    if previous_bgr is None: return np.eye(3,dtype=float)
    def gray(im):
        im=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
        return cv2.resize(im,None,fx=scale,fy=scale,interpolation=cv2.INTER_AREA)
    prev=gray(previous_bgr);curr=gray(current_bgr)
    warp=np.eye(2,3,dtype=np.float32)
    try:
        cv2.findTransformECC(prev,curr,warp,cv2.MOTION_EUCLIDEAN,
            (cv2.TERM_CRITERIA_COUNT|cv2.TERM_CRITERIA_EPS,iterations,1e-5),None,1)
        result=np.eye(3,dtype=float)
        result[:2,:3]=warp
        result[:2,2]/=scale
        if not np.all(np.isfinite(result)):return np.eye(3)
        return result
    except (cv2.error,ValueError):return np.eye(3,dtype=float)
