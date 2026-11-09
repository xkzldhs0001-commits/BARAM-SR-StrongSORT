"""CPU-only tests for position-gated, one-to-one relink."""
import unittest
import numpy as np
from live_relink import reconnect_unmatched

class Track:
    def __init__(self, feature, box, missed=3):
        self.features=[feature]
        self.box=box
        self.time_since_update=missed
    def to_tlwh(self):return self.box
    def is_deleted(self):return False

class Detection:
    def __init__(self, feature, box):
        self.feature=feature
        self.tlwh=box

class RelinkRegression(unittest.TestCase):
    def test_close_match(self):
        a=np.array([1.,0.,0.],dtype=np.float32)
        t=[Track(a,[10,10,50,140])]
        d=[Detection(a,[18,12,50,140])]
        self.assertEqual(reconnect_unmatched(t,d,[0],[0])[0],[(0,0)])
    def test_far_does_not_hijack(self):
        a=np.array([1.,0.,0.],dtype=np.float32)
        t=[Track(a,[10,10,50,140])]
        d=[Detection(a,[600,12,50,140])]
        self.assertEqual(reconnect_unmatched(t,d,[0],[0])[0],[])
    def test_discriminates(self):
        a=np.array([1.,0.,0.],dtype=np.float32)
        b=np.array([0.,1.,0.],dtype=np.float32)
        t=[Track(a,[10,10,50,140]),Track(b,[150,10,50,140])]
        d=[Detection(b,[145,12,50,140]),Detection(a,[15,12,50,140])]
        self.assertEqual(sorted(reconnect_unmatched(t,d,[0,1],[0,1])[0]),[(0,1),(1,0)])
    def test_crossing_does_not_force_second_pass_match(self):
        # Two nearly overlapping tracker predictions are ambiguous. The
        # fallback must not glue a detection to either track by guesswork.
        a=np.array([1.,0.,0.],dtype=np.float32)
        b=np.array([0.,1.,0.],dtype=np.float32)
        tracks=[Track(a,[10,10,50,140]),Track(b,[25,10,50,140])]
        det=[Detection(a,[10,10,50,140])]
        recovered,ut,ud=reconnect_unmatched(tracks,det,[0,1],[0])
        self.assertEqual(recovered,[])
        self.assertEqual(ut,[0,1])
        self.assertEqual(ud,[0])

    def test_too_old(self):
        a=np.array([1.,0.,0.],dtype=np.float32)
        t=[Track(a,[10,10,50,140],missed=22)]
        d=[Detection(a,[10,10,50,140])]
        self.assertEqual(reconnect_unmatched(t,d,[0],[0])[0],[])

if __name__=='__main__':unittest.main()
