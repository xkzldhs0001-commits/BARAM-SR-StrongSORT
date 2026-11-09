"""Appearance features: trained official BoT/FastReID and trained OSNet (no random fallbacks)."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import cv2


def crop_batch(image,boxes,height=256,width=128):
    h,w=image.shape[:2];result=[]
    for x,y,bw,bh in boxes:
        l=max(0,min(w,int(np.floor(x))));t=max(0,min(h,int(np.floor(y))))
        r=max(0,min(w,int(np.ceil(x+bw))));b=max(0,min(h,int(np.ceil(y+bh))))
        crop=image[t:b,l:r]
        if crop.size==0:crop=np.zeros((height,width,3),dtype=np.uint8)
        else:crop=cv2.resize(crop,(width,height),interpolation=cv2.INTER_LINEAR)
        result.append(crop)
    return result


class BoTFastReID:
    def __init__(self,root:Path,device='cuda:0',batch_size=24):
        import torch
        from fastreid.config import get_cfg
        from fastreid.engine import DefaultPredictor
        config=root/'third_party'/'fast-reid'/'configs'/'DukeMTMC'/'bagtricks_S50.yml'
        weights=root/'weights'/'duke_bot_S50.pth'
        if not config.is_file() or not weights.is_file():
            raise FileNotFoundError(f'Official BoT files required: {config} and {weights}; no random fallback.')
        cfg=get_cfg();cfg.merge_from_file(str(config));cfg.defrost()
        cfg.MODEL.WEIGHTS=str(weights);cfg.MODEL.DEVICE=device
        cfg.freeze()
        self.predictor=DefaultPredictor(cfg)
        self.device=device
        self.height,self.width=map(int,cfg.INPUT.SIZE_TEST)
        self.torch=torch
        self.batch_size=batch_size
    def extract(self,image,boxes):
        if not boxes:return np.empty((0,0),dtype=np.float32)
        crops=crop_batch(image,boxes,self.height,self.width)
        output=[]
        for start in range(0,len(crops),self.batch_size):
            batch=crops[start:start+self.batch_size]
            # FastReID DefaultPredictor expects a B,C,H,W tensor, not raw BGR arrays.
            rgb=np.stack([cv2.cvtColor(im,cv2.COLOR_BGR2RGB) for im in batch])
            tensor=self.torch.from_numpy(rgb.transpose(0,3,1,2).copy()).float()
            with self.torch.inference_mode():
                features=self.predictor(tensor)
            arr=features.detach().cpu().numpy() if hasattr(features,'detach') else np.asarray(features)
            if arr.ndim!=2 or len(arr)!=len(batch):raise RuntimeError('Invalid BoT embedding output shape')
            output.extend(arr)
        x=np.asarray(output,dtype=np.float32)
        return x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-9)


class OSNet:
    def __init__(self,root:Path,device='cuda:0',batch_size=24):
        import torch
        import torchreid
        self.torch=torch
        self.device=device
        self.batch_size=batch_size
        weights=root/'weights'/'osnet_x1_0.pth.tar'
        if not weights.is_file():raise FileNotFoundError(f'Pretrained OSNet weights required: {weights}')
        model=torchreid.models.build_model('osnet_x1_0',num_classes=1000,loss='softmax',pretrained=False)
        torchreid.utils.load_pretrained_weights(model,str(weights))
        self.model=model.to(device).eval()
    def extract(self,image,boxes):
        if not boxes:return np.empty((0,512),dtype=np.float32)
        crops=crop_batch(image,boxes)
        output=[]
        for start in range(0,len(crops),self.batch_size):
            rgb=np.stack([cv2.cvtColor(im,cv2.COLOR_BGR2RGB) for im in crops[start:start+self.batch_size]])
            inputs=self.torch.from_numpy(rgb.transpose(0,3,1,2).copy()).to(self.device).float()/255.
            mean=self.torch.tensor([.485,.456,.406],device=self.device)[None,:,None,None]
            std=self.torch.tensor([.229,.224,.225],device=self.device)[None,:,None,None]
            with self.torch.inference_mode(): feat=self.model((inputs-mean)/std)
            output.extend(feat.detach().float().cpu().numpy())
        x=np.asarray(output,dtype=np.float32)
        return x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-9)
