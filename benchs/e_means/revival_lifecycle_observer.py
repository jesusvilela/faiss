"""Stage-7 diagnostic-only NumPy e-means lifecycle observer.

This opt-in subclass intercepts labels at the existing _assign_rows callback.
It never replaces FAISS's k-means algorithm, updates, donor selection, or
revival. Private-method compatibility is scoped to the source in the
preregistration. See REVIVAL_LIFECYCLE_PREREG.md.

To run from a fork checkout, install faiss-cpu and copy contrib/e_means into
the installed faiss.contrib package as in the CI workflow.
"""

from __future__ import annotations

import numpy as np
from faiss.contrib.e_means import NumpyEmeans

from revival_stream_witness import FixedFrameWitness


class WitnessBatchCapture:
    """Observe one logical statistical batch against one immutable frame.

    The adapter retains the frame and O(k*d) summary. It receives stable row
    IDs as a consecutive range within each logical batch, not a stored array.
    """

    def __init__(self, centroids, generation, expected_n, first_id):
        if not isinstance(first_id, (int, np.integer)) or first_id < 0:
            raise ValueError("nonnegative integer first_id required")
        if expected_n < 1:
            raise ValueError("nonempty statistical batch required")
        if first_id + expected_n > np.iinfo(np.int64).max:
            raise ValueError("row ID range overflows int64")
        self.summary = FixedFrameWitness(centroids, int(generation))
        self.generation = int(generation)
        self.expected_n = int(expected_n)
        self.first_id = int(first_id)
        self.seen = 0
        self.callback_calls = 0

    def observe_callback(self, rows, assignment_frame, labels):
        if not np.array_equal(np.asarray(assignment_frame, dtype=np.float32),
                              self.summary.centers):
            raise RuntimeError("assignment frame changed inside logical batch")
        count = len(rows)
        if self.seen + count > self.expected_n:
            raise RuntimeError("assignment callbacks exceeded logical batch")
        ids = np.arange(self.first_id + self.seen,
                        self.first_id + self.seen + count,
                        dtype=np.int64)
        self.summary.observe(rows, labels, ids)
        self.seen += count
        self.callback_calls += 1

    def require_complete(self, stats):
        if self.seen != self.expected_n:
            raise RuntimeError("incomplete observed assignment batch")
        if (stats is None or stats["n"] != self.expected_n or
                stats["sums"] is None or stats["counts"] is None):
            raise RuntimeError("incomplete FAISS logical statistics batch")
        if int(np.sum(self.summary.counts)) != self.expected_n:
            raise RuntimeError("witness count mismatch")


class ObservedNumpyEmeans(NumpyEmeans):
    """Subclass for experiments; ordinary NumpyEmeans remains unchanged."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.spherical:
            raise ValueError("Stage-7 observer only supports Euclidean geometry")
        self._observed_session = None
        self.observed_assign_callbacks = 0

    def _assign_rows(self, x, centroids):
        # The only FAISS assignment call: pass back the original result.
        distance, labels = super()._assign_rows(x, centroids)
        self.observed_assign_callbacks += 1
        if self._observed_session is not None:
            self._observed_session.observe_callback(x, centroids, labels)
        return distance, labels

    def capture_logical_batch(self, chunks, *, first_id):
        """Collect actual assigned rows without a second neighbor search.

        Calls public accumulate() for each chunk; update() is deliberately
        performed by the caller *after* checking the captured statistics.
        The assigned-centroid snapshot is copied before accumulation starts.
        A failed capture can leave a partial FAISS batch; discard the model
        after failure, as required by the source's failure semantics.
        """
        if self.recipe is None or self._failed:
            raise RuntimeError("call begin() on a fresh healthy model first")
        if self._observed_session is not None:
            raise RuntimeError("cannot nest witness capture")
        if self._pending_stats is not None:
            raise RuntimeError("must update or abandon previous logical batch")
        expected = int(self._local_batch_size())
        if self._batch_centroids is not None:
            raise RuntimeError("stale batch frame")
        witness = WitnessBatchCapture(
            self.current_centroids().copy(), self._step, expected, first_id)
        self._observed_session = witness
        stats = None
        try:
            for chunk in chunks:
                arr = np.ascontiguousarray(chunk, dtype=np.float32)
                if arr.ndim != 2 or arr.shape[1] != self.d:
                    raise ValueError("invalid assignment chunk shape")
                if len(arr) < 1 or witness.seen + len(arr) > expected:
                    raise ValueError("empty/overfull assignment chunk")
                stats = self.accumulate(arr)
            witness.require_complete(stats)
        finally:
            self._observed_session = None
        return witness, stats


def sample(seed=20261111):
    rng = np.random.default_rng(seed)
    left = rng.normal(loc=[0., 0., 0.], scale=.12, size=(50, 3))
    right = rng.normal(loc=[10., 0., 0.], scale=.12, size=(50, 3))
    rows = np.vstack([left, right]).astype(np.float32)
    rows = rows[rng.permutation(len(rows))]
    initial = np.array([[0., 0., 0.], [10., 0., 0.],
                        [100., 100., 100.]], dtype=np.float32)
    return rows, initial


def _equal_model_state(left, right):
    np.testing.assert_array_equal(left.current_centroids(),
                                  right.current_centroids())
    np.testing.assert_array_equal(left._usage, right._usage)
    np.testing.assert_array_equal(left._pass_counts, right._pass_counts)
    if left._pass_sums is not None:
        np.testing.assert_array_equal(left._pass_sums, right._pass_sums)
    if left.n_splits != right.n_splits:
        raise AssertionError("revival count changed")
    if left._objective != right._objective:
        raise AssertionError("moving objective changed")


def run_pair(unused_threshold=0.1):
    """Return a compact proof-of-execution record, not a performance result."""
    import faiss

    rows, initial = sample()
    opts = dict(d=3, k=3, nepochs=2, batch_size=20, alpha0=.2,
                unused_threshold=unused_threshold, seed=1234,
                capture_assignments=False, assignment_chunk_size=5)
    baseline = NumpyEmeans(**opts)
    observed = ObservedNumpyEmeans(**opts)
    for model in (baseline, observed):
        model.begin(len(rows), initial)
    results = []
    total_callbacks = 0
    for epoch in range(2):
        for start in range(0, len(rows), 20):
            batch = rows[start:start + 20]
            chunks = [batch[:3], batch[3:10], batch[10:]]
            frame = observed.current_centroids().copy()
            old_step = observed._step

            old_stats = None
            for chunk in chunks:
                old_stats = baseline.accumulate(chunk)
            witness, stats = observed.capture_logical_batch(
                chunks, first_id=epoch * len(rows) + start)
            total_callbacks += witness.callback_calls
            if witness.callback_calls != 5:
                raise AssertionError("unexpected internal assignment-call count")
            if stats["n"] != 20 or old_stats["n"] != 20:
                raise AssertionError("source returned wrong logical batch size")

            # This is a test oracle only, never called by the adapter.
            oracle_dist, oracle_labels = faiss.knn(batch, frame, 1)
            np.testing.assert_array_equal(
                np.bincount(oracle_labels.ravel(), minlength=3),
                witness.summary.counts,
            )
            np.testing.assert_array_equal(
                np.bincount(oracle_labels.ravel(), minlength=3).astype(np.float32),
                stats["counts"],
            )
            if not np.isclose(float(stats["objective"]),
                              float(oracle_dist.sum(dtype=np.float64)),
                              rtol=0, atol=5e-5):
                raise AssertionError("source assignment objective mismatch")
            for j in range(3):
                assigned = np.flatnonzero(oracle_labels.ravel() == j)
                point, radius, rid, reason = witness.summary.witness(j)
                if len(assigned) == 0:
                    if reason != "no_assigned_rows":
                        raise AssertionError("incorrect empty witness state")
                    continue
                residual = batch[assigned].astype(np.float64) - frame[j]
                scores = (residual * residual).sum(axis=1)
                max_index = int(np.argmax(scores))
                expected_id = epoch * len(rows) + start + int(assigned[max_index])
                if reason == "ok" and rid != expected_id:
                    raise AssertionError("farthest witness ID disagrees with oracle")

            baseline.update(old_stats)
            observed.update(stats)
            _equal_model_state(baseline, observed)
            if observed._step != old_step + 1:
                raise AssertionError("logical step was not committed")
            diagnostics = {
                "epoch": epoch, "start": start,
                "step": old_step,
                "source_revivals": observed.n_splits,
                "counts": witness.summary.counts.tolist(),
                "witness_reasons": [
                    witness.summary.witness(j)[3] for j in range(3)],
                "candidate_post_update": [
                    witness.summary.split(
                        j, old_step, observed.current_centroids())["reason"]
                    for j in range(3)],
            }
            # Exercise rejection explicitly; recording a previous step
            # never authorizes use as evidence for a new step.
            if witness.summary.split(
                    0, old_step + 1, observed.current_centroids()
                )["reason"] != "stale_generation":
                raise AssertionError("stale witness was not rejected")
            results.append(diagnostics)
        baseline.end_epoch()
        observed.end_epoch()
        if baseline.iteration_stats[-1]["obj"] != observed.iteration_stats[-1]["obj"]:
            raise AssertionError("epoch objective diverged")
        if baseline.iteration_stats[-1]["nsplit"] != observed.iteration_stats[-1]["nsplit"]:
            raise AssertionError("epoch split count diverged")

    final_baseline = baseline.finalize()
    final_observed = observed.finalize()
    np.testing.assert_array_equal(final_baseline, final_observed)
    np.testing.assert_array_equal(baseline.obj, observed.obj)
    if unused_threshold > 0 and observed.n_splits < 1:
        raise AssertionError("revival-enabled fixture failed to revive a centroid")
    if unused_threshold == 0 and observed.n_splits != 0:
        raise AssertionError("revival-disabled fixture unexpectedly revived")
    if total_callbacks != 50:
        raise AssertionError("unexpected total internal assignment calls")
    return {
        "source": "source-copied FAISS NumPy e-means + compiled faiss-cpu",
        "revival_threshold": unused_threshold,
        "epochs": 2, "logical_batches": len(results),
        "witness_assign_callbacks": total_callbacks,
        "default_revived_clusters": observed.n_splits,
        "post_update_parity": "exact_all_checked_states",
        "final_centroids_exactly_equal": True,
        "step_diagnostics": results,
    }


if __name__ == "__main__":
    import json
    for threshold in (0.0, 0.1):
        result = run_pair(threshold)
        print(json.dumps(result, sort_keys=True))
