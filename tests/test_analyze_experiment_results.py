import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from analyze_experiment_results import compare, read_rows


class ExperimentAnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.poster = read_rows(ROOT / "experiments" / "poster_comparison.csv")
        cls.run = read_rows(ROOT / "experiments" / "run_20261106_comparison.csv")

    def test_mot17_reid_reduction(self):
        stats = compare(self.poster, "MOT17", "StrongSORT", "SR-StrongSORT")
        self.assertAlmostEqual(stats["reid_extractions"]["relative_percent"], -41.8, delta=0.1)

    def test_mot20_hota_improvement(self):
        stats = compare(self.poster, "MOT20", "StrongSORT", "SR-StrongSORT")
        self.assertAlmostEqual(stats["hota"]["delta"], 2.57, places=2)

    def test_run_data_does_not_contain_valid_trackeval_metrics(self):
        stats = compare(self.run, "MOT20", "baseline", "v43_selective")
        self.assertNotIn("hota", stats)
        self.assertLess(stats["fps"]["delta"], 0)

    def test_missing_method_is_an_error(self):
        with self.assertRaises(ValueError):
            compare(self.poster, "MOT17", "StrongSORT", "unknown")


if __name__ == "__main__":
    unittest.main()
