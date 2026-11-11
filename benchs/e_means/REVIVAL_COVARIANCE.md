# E-means revival: rigid-frame covariance diagnostic (experimental)

**Status: research diagnostic, not a production replacement.** The default
FAISS behavior is deliberately unchanged. The source arithmetic inspected is
`facebookresearch/faiss@ddc768060c42e9627b6bf22123fe7e207c9a6efc` in
`contrib/e_means/e_means.py::_apply_splits` and the analogous Torch/Triton paths.
The working fork may have later commits; recheck the formula before applying
these findings elsewhere.

## Observable claim

In the nondegenerate FP32 case, source revival splits a donor `c` into
`(I +- eps D)c`, with diagonal alternating signs `D` and `eps=1e-4`.
Translation by `b` changes this rule by `+-eps D b`. This fails translation
covariance for nonzero `b`; arbitrary rotations also generally fail. This is a
property of the split formula, not evidence of worse FAISS clustering quality.
In 2D the default *disables* revival; choose `d>=3` or enable it explicitly.

## Experimental candidate and limitations

The standalone candidate stores one donor-local witness point `w_j` and an
observed local radius `r_j>0`. It proposes the displacement

    delta = eps * min(1, r_j / ||w_j-c_j||) * (w_j-c_j)

and splits at `c_j +- delta`. A witness is the farthest assigned point in a
specified mini-batch (unique maximum required for an unambiguous geometric
selection). Under a common translation, rotation and positive uniform scale,
this construction is equivariant **in exact arithmetic** when the point,
centroid and radius all transform together. The displacement has length at
most `eps*r_j` before float rounding. No witness or zero distance means
**abstention**, not an invented coordinate-axis direction.

The observation radius may be stale relative to the current cluster spread;
it is not a certified bound on the current distribution. Selecting and
persisting witnesses would require per-cluster state and backend changes,
including distributed synchronization, quantization, ties, and GPU kernels.
No production FAISS modification has been implemented or validated here.

## Reproduce the bounded diagnostic

```
python benchs/e_means/revival_frame_audit.py --tests
python benchs/e_means/revival_frame_audit.py
```

Requires NumPy only. With a source-matched FAISS package installed, run
`--faiss` to compare the actual NumPy split against the transcribed arithmetic.
The local harness is **not** a FAISS package integration test when FAISS is
missing. It is also not a Lloyd-vs-e-means comparison, quality benchmark, or
convergence proof.

### Frozen validation expectations (prior to fork integration)

- Incumbent translation gap > 5e-4 and rotation gap > 1e-4 in the 3D control.
- Witness similarity residual <= 1e-5 in 64 seeded FP32 frames.
- Remote inactive target does not increase witness-displacement magnitude.
- Explicit abstention on no assigned points, zero local radius, or zero vector.
- No claim of throughput or final MSE improvement from these checks.

The original donor-target split is retained only as a *negative control*:
it was observed to produce very large post-revival MSE in an earlier hostile
synthetic experiment with a remote inactive centroid, and should not be
recommended for production.

## Evidence boundaries

The deterministic CPU control uses source-faithful FP32 split arithmetic and
separate proposed geometry. The optional `--faiss` call checks source parity
only; complete e-means training, BF16, FP8, Triton, CuTe, multi-GPU, and
representative retrieval quality remain **NOT TESTED**. This is not an upstream
bug claim. Review the applicability of rigid-frame symmetry to the intended
embedding representation before proposing any change to the default rule.