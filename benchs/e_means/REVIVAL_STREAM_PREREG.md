# Pre-registration — fixed-memory streaming witness state (Stage 6)

Date: 2026-11-11. Commit this protocol **before** producing or executing the
Stage-6 implementation/tests. No change to FAISS default revival, backend,
assignment API, or production kernels is authorized by these tests.

## Question and exact scope

Can a donor-local farthest-witness summary be accumulated **without retaining
all assigned rows**, combined across differently chunked or disjoint shards,
and used safely only while its centroid frame is valid?

Freeze the frame `C_t` for one logical assignment/update step. Each assigned
row has a globally unique stable ID, finite FP32 coordinates, and a label
computed against that same frame. The streaming summary retains for each
cluster: winning row ID/point, largest and second-largest FP64 squared residual
norms, count, and the source centroid snapshot. Storage must scale O(k*d),
not O(number of rows). The summary is valid only for this step. Reuse across
steps must **abstain**, not silently pretend that old and new residual
distances were measured against identical anchors.

The optional bounded candidate uses the same witness- and observed-radius
formula as Stage 3. A centroid drift gate rejects post-update states when
`||C_after[j] - C_before[j]|| > 0.25 * observed_radius[j]`. The 0.25
threshold is a preregistered engineering assumption, not an optimized value.

## Frozen required controls and acceptance conditions

1. For fixed centroids and unique maxima, winner ID and top two squared
   residual scores agree for partitionings by [1, 3, 17, all] rows and
   independently summarized disjoint shards (including reordered merging).
   Winner ID equality is exact; FP64 score differences <= 1e-12.
2. In 64 seeded common similarity transforms (rotation/reflection,
   translations from [-10,10], positive scale [0.5,2]), selected winner
   IDs agree and the bounded proposed split has max absolute FP32
   covariance residual <= 1e-5; reject ambiguous numerical near-ties.
3. Exact ties and nearly tied farthest vectors abstain; no assigned rows,
   absent identifiers, incompatible generations and excessive centroid
   drift abstain; no manufactured coordinate-axis fallback.
4. FP32 split-collapse is detected and abstains. A distant inactive
   target must not change the candidate displacement.
5. State bytes grow linearly in k*d and are independent of number of input
   rows; demonstrate fixed-capacity allocated arrays, not just claim it.
6. On identical synthetic training vectors with non-tied nearest
   centroids, NumPy assignment labels equal compiled FAISS labels in
   an optional installed-FAISS parity job. This does NOT validate
   update/revival integration.

## Synthetic and failure controls

Fixed seed=20261111 for the base sample; k=3, d=3, 64 Gaussian vectors per
cluster with well-separated means, plus one engineered distinct farthest
witness per cluster. Numeric control uses exactly 64 seeded transforms.
Separate hostile cases: equal/opposite maxima, near-equal maxima,
no rows, stale generation, large centroid movement, coordinate 1e6 with
sub-ULP split, and a remote inactive target. Preserve all failures.

## Decision rule

If any required invariant fails, do not claim a composable streaming
witness. If all pass, claim only **standalone fixed-frame accumulator and
merge feasibility**, not better e-means quality. Additional unsolved
bridges are source-version/timing alignment with `update(stats)`,
GPU/low-precision fidelity, sharded global IDs, donor selection vs last
touched batch, persistence of local origins and full training MSE/runtime.

## Evidence governance

Record exact fork commit, Python/NumPy versions, command, test count and
results. Keep this study as an independent draft PR against fork main,
not layered on top of draft PR #1. CI on the exact head is necessary
for artifact verification. Old Stage-4 negative result remains intact.
