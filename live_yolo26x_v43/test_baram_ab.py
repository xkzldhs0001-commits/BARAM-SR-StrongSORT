from pathlib import Path
import tempfile
import unittest
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from run_baram_ab import collect_sequences, read_metric_summary, read_metric_details

class TestABUtilities(unittest.TestCase):
    def test_mot17_one_variant_per_scene_and_mot20(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t)
            for name in ('MOT17-09-DPM','MOT17-09-FRCNN','MOT17-02-SDP','MOT20-01'):
                p=root/'data'/'train'/name
                (p/'img1').mkdir(parents=True)
                (p/'gt').mkdir()
                (p/'seqinfo.ini').write_text('[Sequence]\nseqLength=1\n')
                (p/'gt'/'gt.txt').write_text('1,1,1,1,1,1,1,1,1\n')
            a=collect_sequences(root)
            self.assertEqual([p.name for p in a['MOT17']],['MOT17-02-SDP','MOT17-09-FRCNN'])
            self.assertEqual([p.name for p in a['MOT20']],['MOT20-01'])

    def test_trackeval_summary_and_detailed(self):
        with tempfile.TemporaryDirectory() as t:
            d=Path(t)/'v43_full';d.mkdir()
            (d/'pedestrian_summary.txt').write_text('HOTA DetA MOTA IDSW IDF1\n60.52 61.11 70.33 12 72.78\n')
            (d/'pedestrian_detailed.csv').write_text('seq,HOTA,IDF1,MOTA,IDSW\nMOT17-09-FRCNN,60.52,72.78,70.33,12\n')
            self.assertEqual(read_metric_summary(Path(t)),{'HOTA':60.52,'IDF1':72.78,'MOTA':70.33,'IDSW':12.})
            self.assertEqual(read_metric_details(Path(t))['MOT17-09-FRCNN']['IDSW'],12.)

if __name__=='__main__':unittest.main()
