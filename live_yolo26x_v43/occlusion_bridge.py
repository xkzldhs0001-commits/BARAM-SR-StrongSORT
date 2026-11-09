"""Conservative, tracking-only low-confidence rescue for brief occlusions.

The detector still uses a high threshold to initiate new tracks. A lower-score
box is admitted only if it is a spatially unique match to an existing, recently
observed CONFIRMED SR track. Unmatched low boxes NEVER start new tracks.

This changes live-demo behavior only, not the benchmark SR-StrongSORT code.
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import linear_sum_assignment


def iou(a,b):
    ax1,ay1,aw,ah=map(float,a);bx1,by1,bw,bh=map(float,b)
    x1=max(ax1,bx1);y1=max(ay1,by1)
    x2=min(ax1+aw,bx1+bw);y2=min(ay1+ah,by1+bh)
    inter=max(0.,x2-x1)*max(0.,y2-y1)
    return inter/max(1e-9,aw*ah+bw*bh-inter)


def center_distance(a,b):
    ax,ay,aw,ah=map(float,a);bx,by,bw,bh=map(float,b)
    scale=max(24.,(ah+bh)*.5)
    return float(np.hypot(ax+aw*.5-bx-bw*.5,ay+ah*.5-by-bh*.5)/scale)


def choose_low_conf_rescue(tracker, high_boxes, low_boxes, *, max_missed=8,
                           min_iou=.12, max_distance=.60, margin=.09,
                           high_overlap=.18, duplicate_iou=.55,
                           max_low=40, return_pairs=False):
    """Return low box indices eligible for matching existing tracks only.

    High detections take priority. All low boxes are scored one-to-one against
    predicted, recently observed confirmed tracks. Ambiguous geometry rejected.
    The candidate boxes have no appearance embeddings: they refresh geometry
    without polluting the long-term OSNet appearance template.
    """
    candidates=[]; candidate_ids=[]
    for t in tracker.tracks:
        if t.is_deleted() or not t.is_confirmed():continue
        if int(t.time_since_update)>max_missed:continue
        b=np.asarray(t.to_tlwh(),dtype=float)
        if not np.all(np.isfinite(b)) or b[2]<=1 or b[3]<=1:continue
        # Never rescue this track if a high-confidence person is already nearby.
        if any(iou(b,h)>=high_overlap for h in high_boxes):continue
        candidates.append(b)
        candidate_ids.append(int(t.track_id))
    if not candidates or not low_boxes:return []
    # The detector is already NMS-filtered, but don't add duplicate low boxes.
    lows=[]
    for j,b in enumerate(low_boxes):
        b=np.asarray(b,dtype=float)
        if not np.all(np.isfinite(b)) or b[2]<8 or b[3]<8:continue
        if any(iou(b,h)>=duplicate_iou for h in high_boxes):continue
        lows.append(j)
        if len(lows)>=max_low:break
    if not lows:return []
    scores=np.full((len(candidates),len(lows)),-np.inf,dtype=float)
    for i,tbox in enumerate(candidates):
        for k,j in enumerate(lows):
            box=low_boxes[j]
            overlap=iou(tbox,box)
            dist=center_distance(tbox,box)
            if overlap<min_iou and dist>max_distance:continue
            # Do not connect a body fragment that is extremely smaller than
            # the previous whole-person detection, or a very different shape.
            old_area=max(1.,tbox[2]*tbox[3]);new_area=max(1.,box[2]*box[3])
            area_ratio=new_area/old_area
            if area_ratio<.15 or area_ratio>4.0:continue
            scores[i,k]=.75*overlap+.25*max(0.,1.-dist/max_distance)
    if not np.isfinite(scores).any():return []
    n,m=scores.shape
    cost=np.full((n,m+n),1e5,dtype=float)
    cost[:,:m]=np.where(np.isfinite(scores),-scores,1e5)
    for i in range(n):cost[i,m+i]=-1e-6
    rows,cols=linear_sum_assignment(cost)
    accepted=[]
    for i,j in zip(rows,cols):
        if j>=m or not np.isfinite(scores[i,j]):continue
        best=scores[i,j]
        rivals=([scores[i,k] for k in range(m) if k!=j and np.isfinite(scores[i,k])]
               +[scores[k,j] for k in range(n) if k!=i and np.isfinite(scores[k,j])])
        if rivals and best-max(rivals)<margin:continue
        accepted.append((lows[j],candidate_ids[i]))
    accepted.sort()
    return accepted if return_pairs else [j for j,_ in accepted]


def patched_live_sr_tracker(SR, *args, **kwargs):
    """Wrap SR with V4 appearance relink and forbid low-only new identities."""
    from live_relink import make_live_sr_tracker
    Base=make_live_sr_tracker(SR,*args,**kwargs)
    class LiveSRWithOcclusion(Base):
        def __init__(self):
            super().__init__()
            self.low_conf_rescues=0
            self.last_low_update_ids=set()
        def _initiate_track(self,detection):
            if getattr(detection,'rescue_low',False):
                return  # Never create a track from weak detections.
            return super()._initiate_track(detection)
        def _match(self,detections):
            # Stage 1: normal high-confidence detections always match FIRST.
            # Stage 2: only the remaining tracks may use low-score boxes.
            high=[j for j,d in enumerate(detections) if not getattr(d,'rescue_low',False)]
            low=[j for j,d in enumerate(detections) if getattr(d,'rescue_low',False)]
            if not low:
                matches,ut,ud=super()._match(detections)
                self.last_low_update_ids=set()
                return matches,ut,ud
            high_det=[detections[j] for j in high]
            matches_h, remaining_t, remaining_h=super()._match(high_det)
            matches=[(ti,high[di]) for ti,di in matches_h]
            # A low-confidence detection may update ONLY its geometrically
            # nominated, recently observed confirmed track. Otherwise a weak
            # body/hand false-positive could attach to a different track.
            remaining_t_set=set(remaining_t)
            matched_low=set()
            for di in low:
                det=detections[di]
                desired_id=getattr(det,'rescue_track_id',None)
                eligible=[i for i in remaining_t_set
                          if int(self.tracks[i].track_id)==desired_id]
                if len(eligible)!=1:continue
                ti=eligible[0]
                previous=self.tracks
                try:
                    self.tracks=[previous[ti]]
                    matched,_,_=super()._match([det])
                finally:
                    self.tracks=previous
                if matched==[(0,0)]:
                    matches.append((ti,di))
                    remaining_t_set.remove(ti)
                    matched_low.add(di)
            remaining_t=sorted(remaining_t_set)
            self.last_low_update_ids={int(self.tracks[i].track_id) for i,j in matches
                                      if getattr(detections[j],'rescue_low',False)}
            self.low_conf_rescues+=len(self.last_low_update_ids)
            unmatched_high=[high[j] for j in remaining_h]
            # Keep unmatched low detections in update() so they are logged,
            # but _initiate_track() refuses to create a new low-only identity.
            return matches,remaining_t,unmatched_high+[j for j in low if j not in matched_low]
    return LiveSRWithOcclusion
