"""8D NSA Kalman: [cx,cy,w,h,vx,vy,vw,vh], confidence-dependent R.

Noise is scaled to object size, unlike fixed-pixel noise in the v2 prototype.
These magnitudes are engineering hyperparameters, not proposal-specified values.
"""
from __future__ import annotations
import numpy as np
from .geometry import xyxy_to_xywh_center, xywh_center_to_xyxy

class NSAKalman:
    def __init__(self, box):
        self.F = np.eye(8, dtype=float)
        self.F[:4,4:] = np.eye(4)
        self.H = np.zeros((4,8),dtype=float)
        self.H[:4,:4] = np.eye(4)
        self.x = np.r_[xyxy_to_xywh_center(box),np.zeros(4)].astype(float)
        self._set_noise()
        std = self._std()
        self.P = np.diag(np.r_[(2*std)**2, (4*std)**2])

    def _std(self):
        w,h=np.maximum(self.x[2:4],1.)
        return np.array([max(2.,w*.10), max(2.,h*.10),
                         max(2.,w*.10),max(2.,h*.10)],dtype=float)

    def _set_noise(self):
        sd=self._std()
        self.R=np.diag(sd**2)
        self.Q=np.diag(np.r_[(sd*.25)**2,(sd*.55)**2])

    @property
    def box(self):
        return xywh_center_to_xyxy(self.x)

    def predict(self):
        self._set_noise()
        self.x=self.F@self.x
        self.x[2:4]=np.maximum(self.x[2:4],1.)
        self.P=self.F@self.P@self.F.T+self.Q
        return self.box

    def innovation(self, box, confidence):
        z=xyxy_to_xywh_center(box)
        v=z-self.H@self.x
        self._set_noise()
        R=max(1.-float(np.clip(confidence,0.,1.)),1e-4)*self.R
        S=self.H@self.P@self.H.T+R
        return v,S,R

    def mahalanobis_squared(self,box,confidence):
        v,S,_=self.innovation(box,confidence)
        return max(0.,float(v@np.linalg.solve(S,v)))

    def update(self,box,confidence):
        v,S,R=self.innovation(box,confidence)
        K=np.linalg.solve(S,self.H@self.P).T
        self.x+=K@v
        self.x[2:4]=np.maximum(self.x[2:4],1.)
        IKH=np.eye(8)-K@self.H
        self.P=IKH@self.P@IKH.T+K@R@K.T
        self.P=(self.P+self.P.T)*.5
