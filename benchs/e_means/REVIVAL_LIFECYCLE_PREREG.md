# Stage 7 preregistration: actual NumPy e-means lifecycle observation

Date: 2026-11-11. This preregistration is committed **before** the adapter
implementation or the first exact-branch CI run.

## Source and scope

Source: `jesusvilela/faiss` branch `research/e-means-stream-witness` at
`ae69d2447babcfce5a8d2e58f3a4a334107f15ff`, with unchanged FAISS NumPy
implementation in `contrib/e_means/e_means.py` and
`contrib/e_means/base.py`. Stage 6's `FixedFrameWitness` is imported from
the parent PR. This is a **stacked experimental PR targeting PR #2's
branch**, not a modification of the algorithm or default splitting behavior.

Observe the actual caller-driven sequence
`begin -> accumulate(chunks) -> update(stats) -> end_epoch -> finalize`.
The private callback `NumpyEmeans._assign_rows` may be overridden in a
**diagnostic-only subclass** to copy exact real assignment labels into the
Stage-6 streaming summary, avoiding a second `faiss.knn` pass. The
overridden callback must return `super()._assign_rows` results unchanged.

Only Euclidean, FP32 NumPy e-means is in scope. Non-NumPy, spherical, GPU,
distributed, auto-selection of donor, and actual replacement revival are
**not** exercised.

## Frozen experiment

- Synthetic: seed `20261111`, `n=100`, `k=3`, `d=3`, two
  50-point Gaussian groups with means near (0,0,0) and (10,0,0), standard
  deviation 0.12. A third, inactive initial centroid at (100,100,100).
  Shuffle once with the same seeded generator.
- Two epochs, logical statistical batch size 20, `alpha0=0.2`, model
  seed 1234, `capture_assignments=False`. Feed identical ordered rows to
  both models and use the same incremental chunk partition [3, 7, 10] in
  every logical batch, a remainder-free 20-row total.
- Pairwise controls:
  (A) `unused_threshold=0.0` (revival disabled);
  (B) `unused_threshold=0.1` (revival enabled at 5-step cadence).
  Both use `NumpyEmeans` from the checked-out code; only the instrumented
  model subclasses `_assign_rows`. Compare every post-update centroid,
  usage vector, split count, running objective, epoch totals and final
  centroids with the uninstrumented control.
- Independent assignment oracle: for each logical batch, call an
  independent `faiss.knn` over a copied pre-update centroid snapshot on
  the same FP32 rows. Compare its labels and unique-farthest witness IDs
  with the callback witness (subject to near-tie abstention). This
  independent oracle is **test-only**, not part of adapter execution.
- Record witness state *before* `update`. After `update`, evaluate
  candidate `summary.split` against new centroids for diagnostics only.
  The incumbent e-means revival still runs unmodified.
- Invalid paths: a missing/short logical batch, overfull logical batch,
  stale generation, and attempting a new capture while a pending
  statistics batch exists must fail closed; reporter must not accidentally
  mutate the training model.

## Frozen success and stop criteria

1. Exact zero divergence in each pair's NumPy centroids, usage, split count,
   objective and finalization at every checked step, including at least one
   **actual default revival** in the enabled control.
2. No additional FAISS nearest-neighbor search in the adapter: each callback
   delegates to `super` once, and records only the returned labels. An
   explicit independent `faiss.knn` is permitted in the test oracle only.
3. Exact equality of winner IDs and label counts with the independent
   oracle where the witness is unambiguous.
4. Post-update stale-generation and excessive-drift abstentions remain
   possible and explicitly reported; no candidate split is committed.
5. No new behavior for ordinary `NumpyEmeans` and no upstream/production
   merge claim. Any violation of 1-3 **kills** the adapter claim.

## Evidence limits

This observer hooks a **private Python method**. Its compatibility is pinned
to the current checkout and must be rechecked after source changes.
Instrumented execution adds per-point Python iteration; parity does not
demonstrate acceptable throughput. The existing Stage-4 negative quality
result and Stage-5 limited centered-coordinate result are retained.
