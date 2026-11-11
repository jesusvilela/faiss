# Copyright (c) contributors to the FAISS e-means covariance study.
# Licensed under the MIT license in the repository root.
"""Opt-in NumPy diagnostic for e-means revival; no production changes.

Run:
    python benchs/e_means/revival_frame_audit.py
    python benchs/e_means/revival_frame_audit.py --tests
    python benchs/e_means/revival_frame_audit.py --faiss
"""
import argparse
import json
import unittest
import numpy as np

EPS = np.float32(1e-4)
SOURCE_REF = "ddc768060c42e9627b6bf22123fe7e207c9a6efc"


def reference_split(centroids):
    """Copy of the FP32 NumPy source split for one donor and one target."""
    out = np.asarray(centroids, dtype=np.float32).copy()
    if out.ndim != 2 or out.shape[0] != 2:
        raise ValueError("expected exactly two centroid rows")
    donor = out[[0]]
    sign = np.ones(out.shape[1], dtype=np.float32)
    sign[1::2] = -1.0
    offset = EPS * sign
    low = donor * (1.0 - offset)
    high = donor * (1.0 + offset)
    collapsed = (low == high).all(axis=1)
    value = high[:, 0]
    with np.errstate(over="ignore"):
        neighbor = np.nextafter(value, np.float32(np.inf))
    neighbor = np.where(np.isfinite(neighbor), neighbor,
                        np.nextafter(value, np.float32(-np.inf)))
    high[:, 0] = np.where(collapsed, neighbor, value)
    out[0], out[1] = low[0], high[0]
    return out


def farthest_witness(points, labels, centroid, donor_id=0):
    """Return farthest assigned sample and observed radius, or (None, 0).

    Exact similarity covariance needs a unique farthest point. Floating
    roundoff/ties can change which sample wins; this is not a tie solution.
    """
    points = np.asarray(points, dtype=np.float32)
    labels = np.asarray(labels)
    c = np.asarray(centroid, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != c.size or labels.shape != (len(points),):
        raise ValueError("incompatible points, labels or centroid")
    if not np.isfinite(points).all() or not np.isfinite(c).all():
        raise ValueError("non-finite points or centroid")
    local = points[labels == donor_id]
    if len(local) == 0:
        return None, 0.0
    residual = local.astype(np.float64) - c.astype(np.float64)
    norm2 = np.einsum("ij,ij->i", residual, residual)
    which = int(np.argmax(norm2))
    if norm2[which] <= 0.0:
        return None, 0.0
    # A tie is not a unique geometric witness.  The relative tolerance is
    # intentional: near-ties may change winner after FP32 frame transforms.
    if np.count_nonzero(np.isclose(norm2, norm2[which], rtol=1e-6, atol=0.0)) != 1:
        return None, 0.0
    return local[which].copy(), float(np.sqrt(norm2[which]))


def witness_split(centroids, witness, radius, epsilon=float(EPS)):
    """Proposed bounded split, separate from FAISS implementation.

    Split distance <= epsilon * OBSERVED radius in exact arithmetic. A stale
    witness/radius is not guaranteed to bound the current local dispersion.
    """
    out = np.asarray(centroids, dtype=np.float32).copy()
    if out.ndim != 2 or out.shape[0] != 2:
        raise ValueError("expected exactly two centroid rows")
    c = out[0].astype(np.float64)
    w = np.asarray(witness, dtype=np.float64)
    if not np.isfinite(c).all() or w.shape != c.shape or not np.isfinite(w).all():
        raise ValueError("invalid witness")
    if not np.isfinite(radius) or radius <= 0 or not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("invalid radius or epsilon")
    v = w - c
    norm = float(np.linalg.norm(v))
    if not np.isfinite(norm) or norm == 0:
        raise ValueError("missing geometric direction")
    delta = epsilon * min(1.0, radius / norm) * v
    out[0] = (c - delta).astype(np.float32)
    out[1] = (c + delta).astype(np.float32)
    if not np.isfinite(out).all():
        raise ValueError("split is not representable in FP32")
    if np.array_equal(out[0], out[1]):
        raise ValueError("revival abstained: FP32 operands collapsed")
    return out


def actual_faiss_split(centroids):
    """Optional integration check when matching FAISS checkout is installed."""
    from faiss.contrib.e_means import NumpyEmeans
    x = np.asarray(centroids, dtype=np.float32).copy()
    model = NumpyEmeans(x.shape[1], 2, unused_threshold=0.1)
    model._centroids = x
    model._usage = np.array([8.0, 0.0], dtype=np.float32)
    model._apply_splits(np.array([0]), np.array([1]))
    return model._centroids.copy()


def error(a, b):
    return float(np.max(np.abs(np.asarray(a, dtype=np.float64) -
                               np.asarray(b, dtype=np.float64))))


def experiment(frames=64, seed=20261111):
    c = np.array([[2., 3., 5.], [11., -4., 6.]], dtype=np.float32)
    b = np.array([4., -7., 2.], dtype=np.float32)
    q90 = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]], dtype=np.float32)
    points = c[0] + np.array([[.05, .01, -.02], [-.02, .08, .01],
                              [.01, -.01, .2], [-.08, .02, .01]], dtype=np.float32)
    ids = np.zeros(len(points), dtype=np.int64)
    witness, radius = farthest_witness(points, ids, c[0])
    assert witness is not None
    baseline = witness_split(c, witness, radius)
    rng = np.random.default_rng(seed)
    worst = 0.
    for _ in range(frames):
        raw = rng.normal(size=(3, 3))
        q, r = np.linalg.qr(raw)
        q = (q @ np.diag(np.sign(np.diag(r)))).astype(np.float32)
        scale = np.float32(rng.uniform(.5, 2.))
        shift = rng.uniform(-10., 10., size=3).astype(np.float32)
        ct = (c @ q.T) * scale + shift
        pt = (points @ q.T) * scale + shift
        wt, rt = farthest_witness(pt, ids, ct[0])
        if wt is None:
            raise AssertionError("nondegenerate witness disappeared")
        worst = max(worst, error(witness_split(ct, wt, rt),
                                 (baseline @ q.T) * scale + shift))
    far = np.array([[.05, 0., 0.], [1e6, 0., 0.]], dtype=np.float32)
    proposed = witness_split(far, [.15, 0., 0.], .1)
    return {
        "source_ref": SOURCE_REF, "seed": seed, "frames": frames,
        "original_translation_gap": error(reference_split(c + b), reference_split(c) + b),
        "original_rotation_gap": error(reference_split(c @ q90.T),
                                       reference_split(c) @ q90.T),
        "witness_frame_max_gap": worst,
        "far_target_naive_step": float(np.linalg.norm(far[1].astype(float) -
                                                       far[0].astype(float)) * EPS),
        "far_target_witness_step": float(np.linalg.norm(proposed[0].astype(float) -
                                                         far[0].astype(float))),
    }


class FrameTests(unittest.TestCase):
    def test_original_not_translation_covariant(self):
        self.assertGreater(experiment(0)["original_translation_gap"], 5e-4)

    def test_original_not_rotation_covariant(self):
        self.assertGreater(experiment(0)["original_rotation_gap"], 1e-4)

    def test_witness_similarity_covariance(self):
        self.assertLessEqual(experiment(64)["witness_frame_max_gap"], 1e-5)

    def test_remote_target_bounded(self):
        out = experiment(0)
        self.assertGreater(out["far_target_naive_step"], 50.)
        self.assertLessEqual(out["far_target_witness_step"], 1.1e-5)

    def test_abstain_if_no_local_information(self):
        c = np.array([1., 2., 3.], dtype=np.float32)
        self.assertEqual(farthest_witness(c[None, :], [1], c), (None, 0.))
        self.assertEqual(farthest_witness(c[None, :], [0], c), (None, 0.))
        with self.assertRaises(ValueError):
            witness_split(np.stack((c, c)), c, 0.)

    def test_stale_witness_step_still_bounded(self):
        c = np.array([[10., 0., 0.], [1e6, 0., 0.]], dtype=np.float32)
        out = witness_split(c, np.array([0., 0., 0.], dtype=np.float32), .2)
        self.assertLessEqual(float(np.linalg.norm(out[0].astype(float) -
                                                   c[0].astype(float))), 2.2e-5)

    def test_tied_farthest_witness_abstains(self):
        c = np.zeros(3, dtype=np.float32)
        x = np.array([[1., 0., 0.], [-1., 0., 0.]], dtype=np.float32)
        self.assertEqual(farthest_witness(x, [0, 0], c), (None, 0.0))

    def test_near_tied_witness_abstains(self):
        c = np.zeros(3, dtype=np.float32)
        x = np.array([[1., 0., 0.], [0., 1.0000001, 0.]], dtype=np.float32)
        self.assertEqual(farthest_witness(x, [0, 0], c), (None, 0.0))

    def test_non_finite_input_rejected(self):
        c = np.zeros(3, dtype=np.float32)
        with self.assertRaises(ValueError):
            farthest_witness(np.array([[np.nan, 0., 0.]], dtype=np.float32), [0], c)
        with self.assertRaises(ValueError):
            witness_split(np.vstack([c, c]), np.array([np.nan, 0., 0.]), 1.0)

    def test_fp32_split_collapse_abstains(self):
        # At 1e6, FP32 ULP=0.0625: epsilon * radius is far smaller.
        c = np.array([[1e6, 0., 0.], [0., 0., 0.]], dtype=np.float32)
        w = np.array([1e6 + .125, 0., 0.], dtype=np.float32)
        self.assertGreater(float(np.spacing(c[0, 0])), 1e-2)
        with self.assertRaisesRegex(ValueError, "FP32 operands collapsed"):
            witness_split(c, w, .125)

    def test_actual_faiss_parity_if_installed(self):
        c = np.array([[2., 3., 5.], [11., -4., 6.]], dtype=np.float32)
        try:
            actual = actual_faiss_split(c)
        except (ImportError, ModuleNotFoundError) as exc:
            self.skipTest(str(exc))
        np.testing.assert_array_equal(actual, reference_split(c))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tests", action="store_true")
    parser.add_argument("--faiss", action="store_true")
    parser.add_argument("--frames", type=int, default=64)
    args = parser.parse_args()
    if args.tests:
        unittest.main(argv=[__file__], verbosity=2)
    else:
        result = experiment(args.frames)
        if args.faiss:
            c = np.array([[2., 3., 5.], [11., -4., 6.]], dtype=np.float32)
            result["actual_faiss_parity_gap"] = error(actual_faiss_split(c),
                                                     reference_split(c))
        print(json.dumps(result, indent=2, sort_keys=True))
