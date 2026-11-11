# Post-hoc mathematical note: aggregate statistics cannot reconstruct a witness

**Status:** exact finite counterexample derived after the Stage-6
preregistration. This is a separate, *post-hoc* theorem-target observation,
not an extra preregistered acceptance criterion or a production change.

## Claim

Consider one Euclidean cluster with centroid `c=(0,0)`. Suppose an update
receives only the assigned row count `n`, assigned vector sum `S`, and
scalar objective `Q = sum_i ||x_i-c||^2` (or any function of this same
aggregate tuple and the fixed centroid). There is **no deterministic
function of (c,n,S,Q) alone** that returns the unique farthest assigned
data point for *all* finite inputs.

## Proof by explicit collision of aggregate state

Consider the two batches:

```
A = [(2,0), (-1,0), (-1,0)]
B = [(0,2), (0,-1), (0,-1)]
```

Both have `n=3`, `S=(0,0)`, `c=(0,0)`, and `Q=6`.
All points are assigned to the sole cluster, so even knowing the three
labels would not disambiguate the batches. In `A`, the unique farthest
point is `(2,0)`; in `B`, it is `(0,2)`. A function of the equal
aggregate inputs would have to return the same output for both batches,
contradicting the two required outputs. QED.

This does NOT prove that every useful revival mechanism must retain the
original witness. A covariance operator or other directional statistic
could distinguish these batches, but that requires **additional
information** beyond the present count/sum/scalar-objective tuple.

The counterexample is relevant to the observed FAISS NumPy e-means
`update(stats)` bridge: `_prepare_statistics` extracts assigned sums,
counts and objective for updates. It does not, by itself, reconstruct
the farthest vector. Real adapters must capture a witness during assignment
or pass a separate sufficiently informative geometric statistic.

## Scope and follow-up

The result concerns reconstructing the **unique farthest point** from
aggregate statistics; it does not invalidate ordinary EMA updates, prove
that FAISS lacks other auxiliary information in every backend, or
establish quality/runtime benefits from the prototype.

An independent implementation and realistic vector-stream validation remain
open. This note and its test are exploratory additions to draft PR #2.
