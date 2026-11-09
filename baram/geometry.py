"""Geometry & association gates; pure numpy/scipy, deliberately independently tested."""
import numpy as np
from scipy.optimize import linear_sum_assignment

CHI2_95_4D=9.4877


def iou_tlwh(a,b):
    a=np.asarray(a,dtype=float).reshape(-1,4); b=np.asarray(b,dtype=float).reshape(-1,4)
    if not len(a) or not len(b):return np.zeros((len(a),len(b)),float)
    ax,ay,aw,ah=[a[:,k,None] for k in range(4)]
    bx,by,bw,bh=[b[None,:,k] for k in range(4)]
    inter=np.maximum(0,np.minimum(ax+aw,bx+bw)-np.maximum(ax,bx))*np.maximum(0,np.minimum(ay+ah,by+bh)-np.maximum(ay,by))
    return inter/np.maximum(aw*ah+bw*bh-inter,1e-12)


def ars_tlwh(a,b):
    a=np.asarray(a,dtype=float).reshape(-1,4);b=np.asarray(b,dtype=float).reshape(-1,4)
    if not len(a) or not len(b):return np.zeros((len(a),len(b)))
    theta_a=np.arctan(a[:,2]/np.maximum(a[:,3],1e-8))
    theta_b=np.arctan(b[:,2]/np.maximum(b[:,3],1e-8))
    return np.clip(1-4/np.pi**2*(theta_a[:,None]-theta_b[None,:])**2,0,1)


def center_distance_normalized(a,b):
    a=np.asarray(a,float).reshape(-1,4);b=np.asarray(b,float).reshape(-1,4)
    if not len(a) or not len(b):return np.zeros((len(a),len(b)))
    ac=a[:,:2]+a[:,2:]/2; bc=b[:,:2]+b[:,2:]/2
    scale=np.maximum(np.linalg.norm(a[:,None,2:]+b[None,:,2:],axis=2)/2,1.)
    return np.linalg.norm(ac[:,None,:]-bc[None,:,:],axis=2)/scale


def motion_distance(track, det):
    """Official NSA Kalman innovation covariance, evaluated at detection confidence.

    The upstream KalmanFilter.project() implementation multiplies each measurement
    std by (1-confidence), not the variance by (1-confidence). We retain upstream
    behavior exactly and regularize the covariance for extreme confidences.
    """
    z=det.to_xyah()
    pred,cov=track.kf.project(track.mean,track.covariance,float(det.confidence))
    v=z-pred
    try:
        return float(v@np.linalg.solve(cov+np.eye(4)*1e-9,v))
    except np.linalg.LinAlgError:
        return float('inf')


def normalize(x):
    x=np.asarray(x,dtype=np.float32).reshape(-1)
    n=np.linalg.norm(x)
    if not np.isfinite(n) or n<1e-8:return None
    return x/n


def match_with_unmatched(cost, unmatched_cost=.88):
    """Hungarian with explicit dummy columns: impossible/high-cost matches stay unmatched."""
    c=np.asarray(cost,dtype=float)
    if c.ndim!=2:raise ValueError('Expected 2D cost matrix')
    m,n=c.shape
    if not m or not n:return [],list(range(m)),list(range(n))
    valid=np.isfinite(c)&(c<unmatched_cost)
    padded=np.full((m,n+m),unmatched_cost,dtype=float)
    padded[:,:n]=np.where(valid,c,1e6)
    ri,ci=linear_sum_assignment(padded)
    matched=[(int(i),int(j)) for i,j in zip(ri,ci) if j<n and valid[i,j]]
    used_i={i for i,j in matched};used_j={j for i,j in matched}
    return matched,[i for i in range(m) if i not in used_i],[j for j in range(n) if j not in used_j]
