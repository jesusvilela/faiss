"""Stage-7 tests for live NumPy e-means observer; production unchanged.

Reproduce with the source-copied faiss.contrib.e_means package, as in CI.
"""

import unittest
import numpy as np

from faiss.contrib.e_means import NumpyEmeans
from revival_lifecycle_observer import (
    ObservedNumpyEmeans, sample, run_pair,
)


def fresh_observed(*, unused_threshold=0.1):
    _, centroids = sample()
    model = ObservedNumpyEmeans(
        d=3, k=3, nepochs=2, batch_size=20, alpha0=.2, seed=1234,
        unused_threshold=unused_threshold, capture_assignments=False,
        assignment_chunk_size=5,
    )
    model.begin(100, centroids)
    return model


class Stage7LifecycleTests(unittest.TestCase):

    def test_disabled_revival_is_identical_to_uninstrumented(self):
        result = run_pair(0.0)
        self.assertEqual(result["default_revived_clusters"], 0)
        self.assertTrue(result["final_centroids_exactly_equal"])
        self.assertEqual(result["logical_batches"], 10)
        self.assertEqual(result["witness_assign_callbacks"], 50)

    def test_enabled_revival_is_identical_to_uninstrumented(self):
        result = run_pair(0.1)
        self.assertGreater(result["default_revived_clusters"], 0)
        self.assertEqual(result["post_update_parity"],
                         "exact_all_checked_states")
        self.assertTrue(result["final_centroids_exactly_equal"])

    def test_captures_exact_source_assignment_callback_count(self):
        rows, _ = sample()
        m = fresh_observed()
        w, stats = m.capture_logical_batch(
            [rows[:3], rows[3:10], rows[10:20]], first_id=300)
        self.assertEqual(w.callback_calls, 5)
        self.assertEqual(m.observed_assign_callbacks, 5)
        self.assertEqual(w.seen, 20)
        self.assertEqual(sum(w.summary.counts), 20)
        self.assertEqual(stats["n"], 20)

    def test_rejects_short_logical_batch(self):
        rows, _ = sample()
        m = fresh_observed()
        with self.assertRaisesRegex(RuntimeError, "incomplete observed"):
            m.capture_logical_batch([rows[:10]], first_id=0)

    def test_rejects_overfull_and_empty_logical_chunks(self):
        rows, _ = sample()
        with self.assertRaisesRegex(ValueError, "empty/overfull"):
            fresh_observed().capture_logical_batch([rows[:21]], first_id=0)
        with self.assertRaisesRegex(ValueError, "empty/overfull"):
            fresh_observed().capture_logical_batch([rows[:0]], first_id=0)

    def test_rejects_missing_begin(self):
        m = ObservedNumpyEmeans(
            d=3, k=3, nepochs=2, batch_size=20, alpha0=.2,
            unused_threshold=.1)
        rows, _ = sample()
        with self.assertRaisesRegex(RuntimeError, "call begin"):
            m.capture_logical_batch([rows[:20]], first_id=0)

    def test_rejects_parallel_pending_batch(self):
        rows, _ = sample()
        m = fresh_observed()
        m.accumulate(rows[:10])
        with self.assertRaisesRegex(RuntimeError, "must update"):
            m.capture_logical_batch([rows[10:20]], first_id=10)

    def test_rejects_nested_capture(self):
        m = fresh_observed()
        m._observed_session = object()
        rows, _ = sample()
        with self.assertRaisesRegex(RuntimeError, "cannot nest"):
            m.capture_logical_batch([rows[:20]], first_id=0)

    def test_rejects_negative_ids_and_int64_overflow(self):
        rows, _ = sample()
        with self.assertRaisesRegex(ValueError, "first_id"):
            fresh_observed().capture_logical_batch([rows[:20]], first_id=-1)
        with self.assertRaisesRegex(ValueError, "overflows"):
            fresh_observed().capture_logical_batch(
                [rows[:20]], first_id=np.iinfo(np.int64).max - 5)

    def test_rejects_wrong_chunk_dimension(self):
        rows, _ = sample()
        with self.assertRaisesRegex(ValueError, "invalid assignment chunk"):
            fresh_observed().capture_logical_batch([rows[:20, :2]], first_id=0)

    def test_source_method_not_monkeypatched(self):
        self.assertIsNot(NumpyEmeans._assign_rows,
                         ObservedNumpyEmeans._assign_rows)
        _, centroids = sample()
        ordinary = NumpyEmeans(
            d=3, k=3, nepochs=2, batch_size=20, alpha0=.2)
        self.assertIsInstance(ordinary, NumpyEmeans)
        self.assertNotIsInstance(ordinary, ObservedNumpyEmeans)
        ordinary.begin(100, centroids)
        self.assertFalse(hasattr(ordinary, "_observed_session"))

    def test_unsupported_spherical_abstains_at_construction(self):
        with self.assertRaisesRegex(ValueError, "Euclidean"):
            ObservedNumpyEmeans(d=3, k=3, spherical=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
