import unittest
import tempfile
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from collect import parse

class SummaryParseTests(unittest.TestCase):
 def test_valid(self):
  with tempfile.TemporaryDirectory() as td:
   f=Path(td)/'pedestrian_summary.txt'
   f.write_text('HOTA DetA AssA MOTA IDF1 IDSW CLR_FN CLR_FP\n44.000 50.000 40.000 20.000 30.000 21 180 200\n')
   r=parse(f)
   self.assertEqual(r['HOTA'],'44.000')
   self.assertEqual(r['IDSW'],'21')
 def test_missing(self):
  with self.assertRaises(FileNotFoundError):parse(Path('/madeup/notexistent/pedestrian_summary.txt'))
