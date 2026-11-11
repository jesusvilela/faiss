"""Deterministic Stage-6 test controls (see REVIVAL_STREAM_PREREG.md)."""
import unittest
import numpy as np
from revival_stream_witness import FixedFrameWitness, make_sample, run_metrics


class Stage6Tests(unittest.TestCase):
    def setUp(self):
        self.centers, self.rows, self.labels, self.ids = make_sample()

    def accumulate(self, chunk):
        result = FixedFrameWitness(self.centers, generation=5)
        for begin in range(0, len(self.rows), chunk):
            end = begin + chunk
            result.observe(self.rows[begin:end], self.labels[begin:end], self.ids[begin:end])
        return result

    def assert_same(self, x, y):
        np.testing.assert_array_equal(x.best_ids, y.best_ids)
        np.testing.assert_array_equal(x.best_points, y.best_points)
        np.testing.assert_array_equal(x.best_sq, y.best_sq)
        np.testing.assert_array_equal(x.second_sq, y.second_sq)
        np.testing.assert_array_equal(x.counts, y.counts)

    def test_chunking_invariance(self):
        one = self.accumulate(1)
        for chunk in (3, 17, len(self.rows)):
            with self.subTest(chunk=chunk):
                self.assert_same(one, self.accumulate(chunk))

    def test_merging_disjoint_shards(self):
        baseline = self.accumulate(len(self.rows))
        shards = []
        for shard in range(3):
            i = np.arange(shard, len(self.rows), 3)
            summary = FixedFrameWitness(self.centers, 5)
            summary.observe(self.rows[i], self.labels[i], self.ids[i])
            shards.append(summary)
        for order in ((0, 1, 2), (2, 1, 0), (1, 2, 0)):
            result = shards[order[0]].clone()
            for j in order[1:]:
                result.merge(shards[j])
            self.assert_same(baseline, result)

    def test_similarity_frames(self):
        result = run_metrics(64)
        self.assertEqual(result["winner_id_changes"], 0)
        self.assertLessEqual(result["max_absolute_split_frame_residual"], 1e-5)

    def test_ties_abstain(self):
        c = np.zeros((1, 3), dtype=np.float32)
        s = FixedFrameWitness(c, 1)
        s.observe(np.array([[1., 0., 0.], [-1., 0., 0.]], dtype=np.float32),
                  np.zeros(2, dtype=np.int64), np.array([1, 2], dtype=np.int64))
        self.assertEqual(s.split(0, 1, c)["reason"], "ambiguous_witness")
        s2 = FixedFrameWitness(c, 1)
        s2.observe(np.array([[1., 0., 0.], [1.0000001, 0., 0.]], dtype=np.float32),
                   np.zeros(2, dtype=np.int64), np.array([1, 2], dtype=np.int64))
        self.assertEqual(s2.split(0, 1, c)["reason"], "ambiguous_witness")

    def test_no_rows_or_zero_residual_abstains(self):
        s = FixedFrameWitness(self.centers, 5)
        self.assertEqual(s.split(1, 5, self.centers)["reason"], "no_assigned_rows")
        s.observe(self.centers[0:1], np.array([0]), np.array([1]))
        self.assertEqual(s.split(0, 5, self.centers)["reason"], "zero_residual")

    def test_generation_and_drift(self):
        summary = self.accumulate(17)
        self.assertEqual(summary.split(0, 6, self.centers)["reason"], "stale_generation")
        shifted = self.centers.copy()
        shifted[0, 0] += 1.0
        self.assertEqual(summary.split(0, 5, shifted)["reason"], "centroid_drift")
        moved = self.centers.copy()
        moved[0, 0] += 0.1
        self.assertEqual(summary.split(0, 5, moved)["status"], "ok")

    def test_fp32_collapse(self):
        center = np.array([[1e6, 0., 0.]], dtype=np.float32)
        s = FixedFrameWitness(center, 5)
        s.observe(np.array([[1e6 + .125, 0., 0.]], dtype=np.float32),
                  np.array([0]), np.array([8]))
        self.assertEqual(s.split(0, 5, center)["reason"], "fp32_collapse")

    def test_remote_target_irrelevant(self):
        s = self.accumulate(17)
        out = s.split(0, 5, self.centers)
        self.assertEqual(out["status"], "ok")
        # The old target position is deliberately not an input to the split.
        target = np.array([1e8, 1e8, 1e8], dtype=np.float32)
        self.assertGreater(float(np.linalg.norm(target)), 1e8)
        repeat = s.split(0, 5, self.centers)
        np.testing.assert_array_equal(out["child"], repeat["child"])

    def test_fixed_capacity(self):
        s1 = FixedFrameWitness(self.centers, 5)
        before = s1.allocated_bytes
        s1.observe(self.rows[:3], self.labels[:3], self.ids[:3])
        self.assertEqual(s1.allocated_bytes, before)
        s1.observe(self.rows, self.labels, self.ids + len(self.ids))
        self.assertEqual(s1.allocated_bytes, before)
        self.assertEqual(before, 2*3*3*4 + 3*(8+8+8+8))

    def test_merge_rejects_incompatible_frame(self):
        a = self.accumulate(17)
        with self.assertRaises(ValueError):
            a.clone().merge(FixedFrameWitness(self.centers, 6))
        shifted = self.centers.copy()
        shifted[0, 0] += 1.
        with self.assertRaises(ValueError):
            a.clone().merge(FixedFrameWitness(shifted, 5))

    def test_reject_invalid_data(self):
        s = FixedFrameWitness(self.centers, 5)
        with self.assertRaises(ValueError):
            s.observe(self.rows[:2], self.labels[:2], [1, 1])
        with self.assertRaises(ValueError):
            s.observe(np.array([[np.nan,0.,0.]], dtype=np.float32), [0], [10])

    def test_posthoc_aggregate_obstruction(self):
        # Exploratory finite counterexample, NOT a preregistered success test.
        a = np.array([[2., 0.], [-1., 0.], [-1., 0.]], dtype=np.float32)
        b = np.array([[0., 2.], [0., -1.], [0., -1.]], dtype=np.float32)
        c = np.zeros((1, 2), dtype=np.float32)
        self.assertEqual(len(a), len(b))
        np.testing.assert_array_equal(a.sum(axis=0), b.sum(axis=0))
        self.assertEqual(float(np.sum(a * a)), float(np.sum(b * b)))
        sa, sb = FixedFrameWitness(c, 5), FixedFrameWitness(c, 5)
        ids = np.arange(3, dtype=np.int64)
        labels = np.zeros(3, dtype=np.int64)
        sa.observe(a, labels, ids)
        sb.observe(b, labels, ids)
        wa = sa.witness(0)[0]
        wb = sb.witness(0)[0]
        np.testing.assert_array_equal(wa, [2., 0.])
        np.testing.assert_array_equal(wb, [0., 2.])
        self.assertFalse(np.array_equal(wa, wb))

    def test_missing_ids_rejected(self):
        s = FixedFrameWitness(self.centers, 5)
        with self.assertRaises(ValueError):
            s.observe(self.rows[:1], self.labels[:1], None)

    def test_optional_faiss_assignment_parity(self):
        try:
            import faiss
        except ImportError as exc:
            self.skipTest(str(exc))
        _, nearest = faiss.knn(self.rows, self.centers, 1)
        np.testing.assert_array_equal(nearest.ravel(), self.labels)


if __name__ == "__main__":
    unittest.main(verbosity=2)
