"""Exploratory matched-control metamorphic audit for upstream FAISS PR #5913.

Frozen matrix (before implementation): seed=0..5; d in {2,5};
k in {2,4}; spherical in {False,True}; capture in {False,True};
partition in {all, 1+11, 3+4+5}; 2 epochs * 24 rows each, batch=12,
same generated array and same chunk partition for paired ordinary and
caller-owned label-output models. This file is fork-audit-only, not part
of the upstream contribution.

Kill on any nearest-neighbor label disagreement, any difference in
complete/partial batch stats, post-update centroids/usage/objective/split
counts, epoch stats or final centroids. No claims about cross-partition
numerical invariance, GPU behavior, speedups or representative workloads.
"""
import unittest

import faiss
import numpy as np

from faiss.contrib.e_means import NumpyEmeans


def _problem(seed, k, d, spherical):
    rng = np.random.default_rng(51311 + seed * 17 + k * 31 + d)
    anchors = rng.normal(0, 2, size=(k, d))
    x = np.vstack([
        a + rng.normal(0, .15, size=(24 // k, d))
        for a in anchors
    ]).astype(np.float32)
    initial = anchors.astype(np.float32)
    if spherical:
        # This test provides normalized inputs; it does not test normalization.
        x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-6)
        initial = initial / np.maximum(
            np.linalg.norm(initial, axis=1, keepdims=True), 1e-6)
        x = x.astype(np.float32)
        initial = initial.astype(np.float32)
    permutation = rng.permutation(24)
    return x[permutation], initial


def _run_one(seed, d, k, spherical, capture, partition):
    rows, centroids = _problem(seed, k, d, spherical)
    opts = dict(
        nepochs=2, batch_size=12, alpha0=.3, unused_threshold=0,
        capture_assignments=capture, spherical=spherical,
        assignment_chunk_size=3, seed=1234)
    original = NumpyEmeans(d, k, **opts)
    instrumented = NumpyEmeans(d, k, **opts)
    for m in (original, instrumented):
        m.begin(len(rows), centroids)
    complete_labels_seen = 0
    for epoch in range(2):
        for begin in (0, 12):
            batch = rows[begin:begin + 12]
            frame = instrumented.current_centroids().copy()
            offsets = {
                "all": (12,),
                "one_then_rest": (1, 11),
                "three_pieces": (3, 4, 5),
            }[partition]
            pos = 0
            for n in offsets:
                chunk = batch[pos:pos + n]
                a = original.accumulate(chunk)
                out = np.full(n, -37, dtype=np.int64)
                b = instrumented.accumulate(chunk, labels_out=out)
                distance, expected = faiss.knn(
                    chunk, frame, 1,
                    **({"metric": faiss.METRIC_INNER_PRODUCT}
                       if spherical else {}))
                np.testing.assert_array_equal(out, expected.ravel())
                if not np.shares_memory(out, b["labels"]):
                    raise AssertionError("caller-owned label buffer was copied")
                if a["n"] != b["n"] or a["objective"] != b["objective"]:
                    raise AssertionError("statistics diverged")
                for key in ("sums", "counts"):
                    if a[key] is None:
                        if b[key] is not None:
                            raise AssertionError(key + " prematurely complete")
                    else:
                        np.testing.assert_array_equal(a[key], b[key])
                pos += n
                complete_labels_seen += n
            if pos != 12:
                raise AssertionError("partition did not cover batch")
            original.update(a)
            instrumented.update(b)
            np.testing.assert_array_equal(
                original.current_centroids(), instrumented.current_centroids())
            np.testing.assert_array_equal(original._usage, instrumented._usage)
            if (original._objective != instrumented._objective
                    or original.n_splits != instrumented.n_splits):
                raise AssertionError("post-update state diverged")
        original.end_epoch()
        instrumented.end_epoch()
        # Epoch wall-clock 'time' is not deterministic; compare only
        # the algorithmic objective, imbalance and revival counts.
        for key in ("obj", "imbalance_factor", "nsplit"):
            if (original.iteration_stats[-1][key] !=
                    instrumented.iteration_stats[-1][key]):
                raise AssertionError("epoch statistic diverged: " + key)
    np.testing.assert_array_equal(original.finalize(), instrumented.finalize())
    np.testing.assert_array_equal(original.obj, instrumented.obj)
    if complete_labels_seen != 48:
        raise AssertionError("unexpected assignment coverage")


class MetamorphicMatchedControls(unittest.TestCase):
    def test_euclidean_matched_controls(self):
        self._check(False)

    def test_spherical_matched_controls(self):
        self._check(True)

    def _check(self, spherical):
        for seed in range(6):
            for d in (2, 5):
                for k in (2, 4):
                    for capture in (False, True):
                        for partition in ("all", "one_then_rest", "three_pieces"):
                            with self.subTest(
                                    seed=seed, d=d, k=k,
                                    capture=capture, partition=partition,
                                    spherical=spherical):
                                _run_one(
                                    seed, d, k, spherical,
                                    capture, partition)


if __name__ == "__main__":
    unittest.main(verbosity=2)
