import unittest
import numpy as np
from baram.geometry import iou_tlwh,ars_tlwh,center_distance_normalized,match_with_unmatched,normalize
from baram.selective import SRConfig,SelectiveTracker

class GeometryTests(unittest.TestCase):
 def test_iou(self):
  x=iou_tlwh([[0,0,10,10]],[[0,0,10,10],[20,20,10,10]])
  self.assertEqual(x.shape,(1,2));self.assertAlmostEqual(x[0,0],1);self.assertEqual(x[0,1],0)
 def test_ars(self):
  x=ars_tlwh([[0,0,10,20]],[[99,99,10,20]])[0,0]
  self.assertAlmostEqual(x,1.)
 def test_empty(self):
  self.assertEqual(iou_tlwh([],[]).shape,(0,0))
  self.assertEqual(ars_tlwh([],[[0,0,1,1]]).shape,(0,1))
 def test_cost_no_forced_high_matching(self):
  matched,tracks,dets=match_with_unmatched([[1.0]],unmatched_cost=.8)
  self.assertEqual(matched,[]);self.assertEqual(tracks,[0]);self.assertEqual(dets,[0])
 def test_two_unequal_matching(self):
  matched,unmatched,unused=match_with_unmatched([[0.1,.9],[.7,.2]],.6)
  self.assertEqual(sorted(matched),[(0,0),(1,1)])
 def test_normalize(self):
  self.assertIsNone(normalize(np.zeros(5)))
  self.assertAlmostEqual(np.linalg.norm(normalize([3,4])),1.)

class MiniTrack:
 def __init__(self,measurement,track_id,n_init,max_age,feature,score):
  self.track_id=track_id;self._n_init=n_init;self._max_age=max_age;self.hits=1
  self.age=1;self.time_since_update=0;self.state=1
  self.scores=[score];self.features=[]
  self.mean=np.zeros(8);self.covariance=np.eye(8)
  self._xyah=np.array(measurement)
  self.kf=MiniKF(self)
 def to_tlwh(self):
  x,y,a,h=self._xyah
  return np.array([x-a*h/2,y-h/2,a*h,h])
 def predict(self):self.age+=1;self.time_since_update+=1
 def is_confirmed(self):return self.state==2
 def is_tentative(self):return self.state==1
 def is_deleted(self):return self.state==3
 def mark_missed(self):
  if self.state==1 or self.time_since_update>self._max_age:self.state=3

class MiniKF:
 def __init__(self,track):self.t=track
 def project(self,mean,cov,confidence=0.):return self.t._xyah, np.eye(4)*25
 def update(self,mean,cov,z,confidence):self.t._xyah=np.asarray(z);return mean,cov

class Det:
 def __init__(self,tlwh,conf=.8):self.tlwh=np.asarray(tlwh,dtype=float);self.confidence=conf
 def to_xyah(self):
  x,y,w,h=self.tlwh
  return np.array([x+w/2,y+h/2,w/h,h])

class Extractor:
 def __init__(self):self.called=0
 def extract(self,image,boxes):
  self.called+=len(boxes)
  return [np.array([1.,0.,0.]) for _ in boxes]

class SelectiveTests(unittest.TestCase):
 def setUp(self):
  self.extractor=Extractor();self.sr=SelectiveTracker({'Track':MiniTrack,'TrackState':type('S',(),{'Confirmed':2})},self.extractor,SRConfig())
 def test_single_stationary_person_one_bootstrap(self):
  for i in range(10):self.sr.process(np.zeros((20,20,3)),[Det([10+i*.1,10,20,30])])
  self.assertEqual(self.sr.next_id,2)
  self.assertLessEqual(self.extractor.called,2)
  self.assertEqual(len(self.sr.tracks),1)
 def test_no_track_initialization_below_confidence(self):
  self.sr.process(np.zeros((20,20,3)),[Det([0,0,10,20],.1)])
  self.assertEqual(self.sr.next_id,1)
 def test_dropout_keeps_id(self):
  for i in range(3):self.sr.process(np.zeros((20,20,3)),[Det([0,0,10,20])])
  self.sr.process(np.zeros((20,20,3)),[])
  result=self.sr.process(np.zeros((20,20,3)),[Det([1,0,10,20])])
  self.assertEqual(result[0][0],1)

if __name__=='__main__':unittest.main()
