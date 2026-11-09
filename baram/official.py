"""Original StrongSORT baseline wrapper; the upstream Tracker/Track source is not edited."""
import numpy as np

class OfficialStrongSORT:
    def __init__(self,source):
        self.source=source
        metric=source['Metric']('cosine',source['opt'].max_cosine_distance,1)
        self.tracker=source['Tracker'](metric)
    def process(self,features,detections,sequence,frame,ecc=False):
        original=self.source['Detection']
        boxes=[]
        for d,f in zip(detections,features):
            boxes.append(original(d.tlwh.copy(),d.confidence,np.asarray(f,dtype=np.float64).copy()))
        if ecc:self.tracker.camera_update(sequence,frame)
        self.tracker.predict()
        self.tracker.update(boxes)
        return [(t.track_id,t.to_tlwh().copy()) for t in self.tracker.tracks
                if t.is_confirmed() and t.time_since_update<=1]
