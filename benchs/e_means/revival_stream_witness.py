"""Stage-6 bounded streaming witness experiment, independent of FAISS internals.

Preregistration: REVIVAL_STREAM_PREREG.md. This is not a production change.
One fixed centroid snapshot per logical assignment step. Global IDs must be
unique across the entire step. Farthest and runner-up distances are computed
in float64 from the supplied float32 coordinates.
"""

from __future__ import annotations

import numpy as np


class FixedFrameWitness:
    """O(k*d)-memory summary for one frozen centroid frame.

    The accumulator sees externally provided assignments and global row IDs.
    Assignments must correspond to reference_centroids and generation.
    One accumulator cannot validate global ID uniqueness with O(k*d) memory;
    that uniqueness is a caller obligation, including across shards.
    """

    def __init__(self, reference_centroids, generation, ambiguity_rtol=1e-6):
        centers = np.asarray(reference_centroids, dtype=np.float32)
        if centers.ndim != 2 or not centers.size or not np.isfinite(centers).all():
            raise ValueError("finite nonempty (k,d) centroids required")
        if not isinstance(generation, (int, np.integer)) or generation < 0:
            raise ValueError("nonnegative integer generation required")
        if not (0 <= ambiguity_rtol < 1):
            raise ValueError("invalid ambiguity_rtol")
        self.centers = centers.copy()
        self.generation = int(generation)
        self.ambiguity_rtol = float(ambiguity_rtol)
        k, d = centers.shape
        self.best_ids = np.full(k, -1, dtype=np.int64)
        self.best_points = np.zeros((k, d), dtype=np.float32)
        self.best_sq = np.full(k, -np.inf, dtype=np.float64)
        self.second_sq = np.full(k, -np.inf, dtype=np.float64)
        self.counts = np.zeros(k, dtype=np.int64)

    @property
    def allocated_bytes(self):
        """Bytes in retained NumPy arrays, excluding Python object overhead."""
        return sum(a.nbytes for a in (
            self.centers, self.best_ids, self.best_points,
            self.best_sq, self.second_sq, self.counts,
        ))

    def observe(self, rows, labels, ids):
        x = np.asarray(rows, dtype=np.float32)
        lab = np.asarray(labels)
        ind = np.asarray(ids)
        k, d = self.centers.shape
        if x.ndim != 2 or x.shape[1] != d:
            raise ValueError("row dimension mismatch")
        if lab.shape != (len(x),) or ind.shape != (len(x),):
            raise ValueError("labels and row IDs must match rows")
        if not (np.issubdtype(lab.dtype, np.integer) and
                np.issubdtype(ind.dtype, np.integer)):
            raise ValueError("integer labels and IDs required")
        if (np.any(lab < 0) or np.any(lab >= k) or
                np.any(ind < 0) or not np.isfinite(x).all()):
            raise ValueError("nonfinite vector, invalid label or row ID")
        if len(np.unique(ind)) != len(ind):
            raise ValueError("duplicate row IDs inside chunk")
        for row, cluster, row_id in zip(x, lab, ind):
            j, rid = int(cluster), int(row_id)
            residual = row.astype(np.float64) - self.centers[j].astype(np.float64)
            score = float(np.dot(residual, residual))
            if not np.isfinite(score):
                raise ValueError("nonfinite squared distance")
            self.counts[j] += 1
            better = (score > self.best_sq[j] or
                      (score == self.best_sq[j] and rid < self.best_ids[j]))
            if better:
                self.second_sq[j] = max(self.second_sq[j], self.best_sq[j])
                self.best_sq[j] = score
                self.best_ids[j] = rid
                self.best_points[j] = row
            else:
                self.second_sq[j] = max(self.second_sq[j], score)

    def merge(self, other):
        """Merge independent disjoint row-ID shards against identical anchors."""
        if not isinstance(other, FixedFrameWitness):
            raise TypeError("merge needs FixedFrameWitness")
        if (self.generation != other.generation or
                self.ambiguity_rtol != other.ambiguity_rtol or
                not np.array_equal(self.centers, other.centers)):
            raise ValueError("incompatible centroid reference frame")
        for j in range(len(self.counts)):
            if not other.counts[j]:
                continue
            if self.counts[j] == 0:
                self.best_ids[j] = other.best_ids[j]
                self.best_points[j] = other.best_points[j]
                self.best_sq[j] = other.best_sq[j]
                self.second_sq[j] = other.second_sq[j]
            else:
                if self.best_ids[j] == other.best_ids[j]:
                    raise ValueError("best witness row ID duplicated across shards")
                score0, score1 = self.best_sq[j], other.best_sq[j]
                other_wins = (score1 > score0 or
                              (score1 == score0 and other.best_ids[j] < self.best_ids[j]))
                self.second_sq[j] = max(score0 if other_wins else score1,
                                        self.second_sq[j], other.second_sq[j])
                if other_wins:
                    self.best_sq[j] = score1
                    self.best_ids[j] = other.best_ids[j]
                    self.best_points[j] = other.best_points[j]
            self.counts[j] += other.counts[j]
        return self

    def clone(self):
        copy = FixedFrameWitness(self.centers, self.generation, self.ambiguity_rtol)
        for name in ("best_ids", "best_points", "best_sq", "second_sq", "counts"):
            getattr(copy, name)[:] = getattr(self, name)
        return copy

    def witness(self, cluster):
        """Return (point,radius,row_id,reason) without a frame transition."""
        j = int(cluster)
        if j < 0 or j >= len(self.counts):
            raise IndexError("invalid cluster index")
        if self.counts[j] == 0:
            return None, 0.0, None, "no_assigned_rows"
        top, second = self.best_sq[j], self.second_sq[j]
        if top <= 0:
            return None, 0.0, None, "zero_residual"
        if second != -np.inf and top - second <= self.ambiguity_rtol * top:
            return None, 0.0, None, "ambiguous_witness"
        return self.best_points[j].copy(), float(np.sqrt(top)), int(self.best_ids[j]), "ok"

    def split(self, cluster, generation, current_centroids, *,
              epsilon=1e-4, max_drift_fraction=0.25):
        """Attempt bounded FP32 split; abstain with an explicit reason."""
        if generation != self.generation:
            return {"status": "abstain", "reason": "stale_generation"}
        current = np.asarray(current_centroids, dtype=np.float32)
        if current.shape != self.centers.shape or not np.isfinite(current).all():
            raise ValueError("current centroid frame mismatch")
        if epsilon <= 0 or not np.isfinite(epsilon) or max_drift_fraction < 0:
            raise ValueError("invalid split parameters")
        w, radius, rid, reason = self.witness(cluster)
        if reason != "ok":
            return {"status": "abstain", "reason": reason}
        j = int(cluster)
        c = current[j].astype(np.float64)
        reference = self.centers[j].astype(np.float64)
        drift = float(np.linalg.norm(c - reference))
        if drift > max_drift_fraction * radius:
            return {"status": "abstain", "reason": "centroid_drift"}
        v = w.astype(np.float64) - c
        length = float(np.linalg.norm(v))
        if length <= 0 or not np.isfinite(length):
            return {"status": "abstain", "reason": "no_current_direction"}
        delta = epsilon * min(1.0, radius / length) * v
        low = (c - delta).astype(np.float32)
        high = (c + delta).astype(np.float32)
        if not (np.isfinite(low).all() and np.isfinite(high).all()):
            return {"status": "abstain", "reason": "fp32_overflow"}
        if np.array_equal(low, high):
            return {"status": "abstain", "reason": "fp32_collapse"}
        return {"status": "ok", "reason": "ok", "donor": low,
                "child": high, "radius": radius, "witness_id": rid,
                "observed_step_bound": epsilon * radius,
                "centroid_drift": drift}


def make_sample(seed=20261111):
    rng = np.random.default_rng(seed)
    centers = np.array([[0., 0., 0.], [10., 0., 0.], [0., 12., 0.]], dtype=np.float32)
    x = []
    for j, c in enumerate(centers):
        block = (c + rng.normal(0., .08, size=(64, 3))).astype(np.float32)
        block = np.vstack((block, c + np.array([2., .04 * j, .07], dtype=np.float32)))
        x.append(block)
    rows = np.vstack(x)
    labels = np.repeat(np.arange(3, dtype=np.int64), 65)
    ids = np.arange(len(rows), dtype=np.int64)
    return centers, rows, labels, ids


def run_metrics(frames=64, seed=20261111):
    centers, x, lab, ids = make_sample(seed)
    base = FixedFrameWitness(centers, 5)
    for begin in range(0, len(x), 17):
        end = begin + 17
        base.observe(x[begin:end], lab[begin:end], ids[begin:end])
    outputs = [base.split(j, 5, centers) for j in range(3)]
    if any(o["status"] != "ok" for o in outputs):
        raise AssertionError("unexpected base witness abstention")
    rng = np.random.default_rng(seed + 1)
    largest = 0.
    wrong = 0
    for _ in range(frames):
        q, r = np.linalg.qr(rng.normal(size=(3, 3)))
        q = (q @ np.diag(np.sign(np.diag(r)))).astype(np.float32)
        s = np.float32(rng.uniform(.5, 2.))
        b = rng.uniform(-10., 10., size=3).astype(np.float32)
        ct = (centers @ q.T * s + b).astype(np.float32)
        xt = (x @ q.T * s + b).astype(np.float32)
        transformed = FixedFrameWitness(ct, 5)
        for begin in range(0, len(x), 17):
            end = begin + 17
            transformed.observe(xt[begin:end], lab[begin:end], ids[begin:end])
        for j in range(3):
            out = transformed.split(j, 5, ct)
            if out["status"] != "ok":
                raise AssertionError("transform unexpectedly caused abstention")
            if out["witness_id"] != outputs[j]["witness_id"]:
                wrong += 1
            for key in ("donor", "child"):
                expected = outputs[j][key] @ q.T * s + b
                largest = max(largest, float(np.max(np.abs(expected-out[key]))))
    return {"frames": frames, "winner_id_changes": wrong,
            "max_absolute_split_frame_residual": largest,
            "array_bytes": base.allocated_bytes, "k": 3, "d": 3,
            "rows": len(x), "all_initial_splits_noncollapsed": True}


if __name__ == "__main__":
    import argparse
    import json
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--frames", type=int, default=64)
    p.add_argument("--faiss", action="store_true")
    a = p.parse_args()
    result = run_metrics(a.frames)
    if a.faiss:
        import faiss
        centers, rows, labels, _ = make_sample()
        _, assigned = faiss.knn(rows, centers, 1)
        result["faiss_assignment_mismatches"] = int(np.count_nonzero(assigned.ravel() != labels))
        if result["faiss_assignment_mismatches"]:
            raise AssertionError("FAISS nearest-centroid assignment differs")
    print(json.dumps(result, indent=2, sort_keys=True))
