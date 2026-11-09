"""CPU-only V4.3 ID bank tests: no GPU, Ultralytics, or model weights required."""
import unittest
import numpy as np
from identity_memory import StableIdentityBank

A = np.array([1., 0., 0., 0.], dtype=np.float32)
B = np.array([0., 1., 0., 0.], dtype=np.float32)
C = np.array([0., 0., 1., 0.], dtype=np.float32)


def person(tid, feat, x=0, *, clean=True, occ=False):
    return dict(internal_id=tid, bbox=(x, 0, 48, 135), feature=feat,
                learnable=clean, occluded=occ)


class IDBankTests(unittest.TestCase):
    def setUp(self):
        self.b = StableIdentityBank(ttl_seconds=180., confirm_frames=3,
                                    swap_confirm_frames=3, max_verify_frames=9)
        self.frame = 0
        self.sec = 0.

    def step(self, *tracks):
        self.frame += 1
        self.sec += .12
        return self.b.update(tracks, self.sec, self.frame)

    def enroll(self):
        # First known entrant has ID 1; a distinct second entrant must be
        # observed cleanly long enough not to be a duplicate detection.
        out, _ = self.step(person(1, A, 0), person(2, B, 300))
        self.assertEqual(out, {1:1, 2:'VERIFY'})
        for _ in range(9):
            out, _ = self.step(person(1, A, 0), person(2, B, 300))
        self.assertEqual(out, {1:1, 2:2})
        self.assertEqual(self.b.total_new, 2)

    def test_reentrance_reversed_order(self):
        self.enroll()
        for _ in range(46):
            self.step()
        out = None
        for _ in range(5):
            out, _ = self.step(person(11, B, 30))
        self.assertEqual(out[11], 2)
        for _ in range(5):
            out, _ = self.step(person(11, B, 30), person(12, A, 200))
        self.assertEqual(out, {11:2, 12:1})
        self.assertEqual(self.b.total_new, 2)

    def test_genuine_new_person_with_all_old_ids_occupied(self):
        self.enroll()
        for _ in range(4):
            out, _ = self.step(person(1, A, 0), person(2, B, 300), person(3, C, 600))
            self.assertEqual(out[3], 'VERIFY')
        for _ in range(7):
            out, _ = self.step(person(1, A, 0), person(2, B, 300), person(3, C, 600))
        self.assertEqual(out[3], 3)
        self.assertEqual(self.b.total_new, 3)

    def test_ambiguous_crops_do_not_create_new_ids(self):
        self.enroll()
        before = len(self.b.records[1].templates)
        for _ in range(40):
            out, _ = self.step(person(1, B, 100, clean=False, occ=True),
                               person(2, A, 120, clean=False, occ=True),
                               person(9, None, 110, clean=False, occ=True))
        self.assertEqual(out, {1:1, 2:2, 9:'VERIFY'})
        self.assertEqual(self.b.total_new, 2)
        self.assertEqual(len(self.b.records[1].templates), before)
        self.assertIn(1, self.b.provisional_internals)

    def test_atomic_swap_after_clean_separation(self):
        self.enroll()
        for _ in range(4):
            self.step(person(1, B, 150, clean=False, occ=True),
                      person(2, A, 180, clean=False, occ=True))
        for i in range(3):
            out, ev = self.step(person(1, B, 300), person(2, A, 0))
        self.assertEqual(out, {1:2, 2:1})
        self.assertEqual(self.b.total_corrected_swaps, 2)
        self.assertEqual(self.b.total_new, 2)
        self.assertTrue(all(e['method'] == 'atomic_swap' for e in ev))

    def test_one_bad_crop_does_not_flip(self):
        self.enroll()
        self.step(person(1, B, 0), person(2, A, 300))
        out, _ = self.step(person(1, A, 0), person(2, B, 300))
        self.assertEqual(out, {1:1, 2:2})

    def test_unreliable_new_track_cannot_multiply_ids(self):
        self.enroll()
        for _ in range(24):
            out, _ = self.step(person(4, None, 0, clean=False))
        self.assertEqual(out[4], 'VERIFY')
        self.assertEqual(self.b.total_new, 2)

    def test_same_appearance_as_occupied_id_not_new(self):
        self.enroll()
        for _ in range(20):
            out, _ = self.step(person(1, A, 0), person(2, B, 300),
                               person(8, A, 30))
        self.assertEqual(out[8], 'VERIFY')
        self.assertEqual(self.b.total_new, 2)

    def test_borderline_dormant_ID_not_a_new_person(self):
        self.enroll()
        for _ in range(50):
            self.step()
        shifted = np.array([.73, .0, .683, .0], dtype=np.float32)
        for _ in range(20):
            out, _ = self.step(person(20, shifted, 100))
        self.assertEqual(out[20], 'VERIFY')
        self.assertEqual(self.b.total_new, 2)

    def test_ttl(self):
        self.enroll()
        self.sec += 181.
        self.step()
        self.assertEqual(self.b.records, {})
        self.assertEqual(self.b.bindings, {})

    def test_unique_bound_display_ids(self):
        self.enroll()
        self.step(person(3, A, 10), person(4, A, 20))
        for _ in range(6):
            self.step(person(3, A, 10), person(4, A, 20))
        self.assertEqual(len(self.b.bindings.values()), len(set(self.b.bindings.values())))

    def test_clean_samples_required_to_restore(self):
        self.enroll()
        for _ in range(50):
            self.step()
        for _ in range(2):
            out, _ = self.step(person(12, A, 20))
        for _ in range(12):
            out, _ = self.step(person(12, None, 20, clean=False))
        self.assertEqual(out[12], 'VERIFY')
        for _ in range(5):
            out, _ = self.step(person(12, A, 20))
        self.assertEqual(out[12], 1)

if __name__ == '__main__':
    unittest.main()
