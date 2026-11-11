# Stage 4: pre-registered isolated revival-quality control

Date: 2026-11-11. This protocol was committed **before** writing or executing the
new Stage-4 study; do not alter criteria after seeing the results. Changes require
a separately marked exploratory amendment.

## Claim and estimand

Can a bounded donor-local witness revival avoid catastrophic coordinate-origin
sensitivity **without materially worsening held-out nearest-centroid MSE**?
This is a one-revival-plus-one-Lloyd-update **toy control**, not a full FAISS
e-means training evaluation or independent replication of the e-means paper.

## Frozen inputs

- Seeds 0..9; 3-dimensional data; two Gaussian modes centered at +5 and -5
  along coordinate 0, with 256 independent training and 256 held-out points
  per mode.
- Two spread patterns: isotropic std=(0.30,0.30,0.30);
  anisotropic std=(0.15,0.70,0.25).
- Three shared frame translations: 0, 100, 10000 in all coordinates.
  Each generated sample and each centroid is rounded to FP32 *after*
  the translation. Compute MSE distances in FP64 on those FP32 inputs.
- Fixed source donor at positive mode, one live centroid at negative mode,
  and one inactive target at an offset of (+40,+40,+40) from the positive
  mode. Use a single revival, bypassing donor selection.
- Methods: (A) source-matched coordinate-relative revival;
  (B) bounded unique-farthest-training-witness revival with explicit
  abstention on no witness, near-tie or FP32 collapse;
  (C) no-revival control.
- All methods receive **exactly the same** seed, train/holdout split,
  centroids, and evaluation procedure; no method gets test-set information.

## Measured outcomes

For every scenario/method, report:
1. Immediate held-out mean squared nearest-centroid distance.
2. Held-out MSE after one Lloyd-style assignment/mean update computed on
   training points (unassigned centroids stay fixed).
3. Whether witness revival abstained; do not drop abstentions from averages.
4. Revival displacement and count of nonempty clusters in the Lloyd update.
5. Worst-case and median per-regime differences, not only aggregate means.

For abstention, method B uses the unchanged source/target centroids (method C);
this is **coverage loss** and must be reported as such.

## Decision rule

Evidence does NOT support a production replacement if any of:
- Witness abstention rate > 10% overall, or > 20% in any regime;
- Witness median held-out MSE after Lloyd > source by 1% in any regime;
- Benefits are confined to large coordinate translations without
  representative benefit on unshifted controls.

If a condition fails, retain as a mathematical/diagnostic contribution only.
Passing would merely justify a later full FAISS/streaming/backend experiment,
not merging a new splitting rule.

## Known scope and negative controls

The farthest-witness construction is given access to donor-assigned training
rows; actual FAISS update receives aggregated sums/counts and lacks that state.
A near-degenerate witness selection may abstain. Large translations expose
FP32 representability and scaling effects rather than a realistic recommendation
to re-center all datasets. No time/throughput claim should be inferred.

Baseline formula is copied from the currently pinned fork source. Report
version, seed and every failure; no replacing a failed run with a new seed.
