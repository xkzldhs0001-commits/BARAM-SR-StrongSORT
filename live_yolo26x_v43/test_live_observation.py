"""CPU-only clean-crop/occlusion tests with adapter IOU shim."""
import unittest
import sys
import types
from unittest.mock import patch
import numpy as np
from live import observed_tracks_with_fresh_features


def iou(a,b):
    ax,ay,aw,ah=a;bx,by,bw,bh=b
    x1=max(ax,bx);y1=max(ay,by)
    x2=min(ax+aw,bx+bw);y2=min(ay+ah,by+bh)
    inter=max(0,x2-x1)*max(0,y2-y1)
    return inter/max(1e-9,aw*ah+bw*bh-inter)

class Track:
    def __init__(self,i,b):
        self.track_id=i
        self.b=b
        self.time_since_update=0
    def is_confirmed(self): return True
    def is_deleted(self): return False
    def to_tlwh(self): return self.b

class Tracker:
    def __init__(self,tracks):
        self.tracks=tracks
        self.last_low_update_ids=set()

class ObservationTests(unittest.TestCase):
    def obs(self,tracks,boxes,feat_map):
        fake_pkg=types.ModuleType('official_compare')
        fake_bridge=types.ModuleType('official_compare.strongsort_adapter')
        fake_bridge.iou_xywh=iou
        with patch.dict(sys.modules, {'official_compare':fake_pkg,
                                      'official_compare.strongsort_adapter':fake_bridge}):
            return observed_tracks_with_fresh_features(Tracker(tracks),boxes,feat_map,
                                                       frame_size=(1280,720))

    def test_separated_feature_clean(self):
        t=[Track(1,[100,50,48,130])]
        obs=self.obs(t,[[100,50,48,130]],{0:np.array([1.,0.])})
        self.assertTrue(obs[0]['learnable'])
        self.assertFalse(obs[0]['occluded'])

    def test_overlap_never_learn(self):
        t=[Track(1,[100,50,70,130]),Track(2,[132,50,70,130])]
        obs=self.obs(t,[[100,50,70,130],[132,50,70,130]],
                     {0:np.array([1.,0.]),1:np.array([0.,1.])})
        self.assertTrue(all(x['occluded'] for x in obs))
        self.assertTrue(all(not x['learnable'] for x in obs))

    def test_weak_rescue_provisional(self):
        t=[Track(1,[100,50,48,130])]
        tracker=Tracker(t)
        tracker.last_low_update_ids={1}
        fake_pkg=types.ModuleType('official_compare')
        fake_bridge=types.ModuleType('official_compare.strongsort_adapter')
        fake_bridge.iou_xywh=iou
        with patch.dict(sys.modules, {'official_compare':fake_pkg,
                                      'official_compare.strongsort_adapter':fake_bridge}):
            obs=observed_tracks_with_fresh_features(tracker,[[100,50,48,130]],
                         {0:np.array([1.,0.])},frame_size=(1280,720))
        self.assertTrue(obs[0]['occluded'])
        self.assertFalse(obs[0]['learnable'])

if __name__=='__main__':unittest.main()
