import csv
import tempfile
import unittest
from pathlib import Path
from baram_profile_analysis import analyze,load_frames,locate_archive,crowd_bin,percentile

COLS=['frame','detections','high_conf_detections','image_read_ms','reid_features',
      'detector_ms','ecc_ms','predict_select_ms','reid_ms','association_ms',
      'id_memory_ms','wall_ms']


def fake_csv(p,base=False, high=True):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('w',newline='') as f:
        columns=COLS if high else [c for c in COLS if c not in ('high_conf_detections','image_read_ms')]
        w=csv.DictWriter(f,columns);w.writeheader()
        for i in range(1,8):
            row={'frame':i,'detections':i+1,'high_conf_detections':i,
            'image_read_ms':1,'reid_features':i if base else (i if i%2 else 0),
            'detector_ms':5,'ecc_ms':0,'predict_select_ms':2,
            'reid_ms':4 if base else 2,'association_ms':1,'id_memory_ms':0 if base else 1,
            'wall_ms':15 if base else 14}
            w.writerow({k:row[k] for k in columns})


class ProfileTests(unittest.TestCase):
    def test_density(self):
        self.assertEqual([crowd_bin(n) for n in (0,1,5,6,15,16,100)],
                         ['0','1–5','1–5','6–15','6–15','16+','16+'])

    def test_percentile(self):
        self.assertEqual(percentile([1,2,3,4,5],95),4.8)

    def test_analyze_new(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);files={}
            for m in ('baseline','v43_selective'):
                path=p/'runs/MOT17/MOT17-10-FRCNN'/m/'MOT17-10-FRCNN_mode_time'/'frames.csv'
                fake_csv(path,base=(m=='baseline'))
                files[('MOT17','MOT17-10-FRCNN',m)]=path
            summary,stage,crowd=analyze(files,p/'result',warmup=1,source='unit_test')
            self.assertEqual(len(summary),2)
            self.assertTrue((p/'result/MOT17_stage_latency.png').is_file())
            self.assertTrue((p/'result/MOT17_crowd_latency.png').is_file())
            self.assertEqual(next(x for x in summary if x['method']=='baseline')['frames'],6)
            self.assertAlmostEqual(next(x for x in stage if x['method']=='baseline' and x['stage']=='I/O & Other')['mean_ms_per_frame'],3.0)
            self.assertEqual(len(locate_archive(p)),2)

    def test_old_archive_without_high_conf(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'frames.csv';fake_csv(path,base=True,high=False)
            frames,original=load_frames(path,2)
            self.assertEqual(len(frames),5)
            self.assertFalse(frames[0]['high_conf_available'])
            self.assertEqual(frames[0]['detections'],4)

    def test_reject_mismatch(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);files={}
            for m in ('baseline','v43_selective'):
                path=p/m/'frames.csv'
                fake_csv(path,base=(m=='baseline'))
                files[('MOT20','MOT20-02',m)]=path
            with (p/'baseline/frames.csv').open('a') as out:
                out.write('8,4,4,1,4,5,0,2,4,1,0,15\n')
            with self.assertRaises(RuntimeError):
                analyze(files,p/'result',warmup=0)

if __name__=='__main__':unittest.main()
