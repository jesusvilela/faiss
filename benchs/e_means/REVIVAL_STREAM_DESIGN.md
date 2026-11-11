# Stage 6 — streaming witness accumulator (CPU research prototype)

**Status:** standalone diagnostic, **not wired into e-means training or revival**.
The experiment is governed by [REVIVAL_STREAM_PREREG.md](REVIVAL_STREAM_PREREG.md).
This branch is independent of [draft PR #1](https://github.com/jesusvilela/faiss/pull/1)
and preserves that PR's negative Stage-4 and positive-but-limited Stage-5 results.

## Contract

For a *fixed* centroid snapshot `C_t` and one assignment step `t`,
receive already-assigned FP32 points `x_i`, cluster labels `j_i`, and stable
**globally unique** row IDs `i`. Retain per cluster only

`W_j = (C_t[j], count_j, max_i ||x_i-C_t[j]||^2, second_max, winning_id, winning_point)`.

All residual scores are computed from those FP32 coordinates using FP64
subtraction and squared norms. The summary does **not** determine whether
the externally supplied labels are correct. It also does not store arbitrary
rows; retained NumPy arrays scale as `8*k*d + 32*k` bytes, not with row count.
A full chunk is still supplied to `observe()` and uses O(batch) temporary
space for ID duplicate checking; Python object overhead is excluded.

Chunk partitions and disjoint shard reductions use the top-two-score monoid
under a common anchor and generation; this is exactly composable for
finite inputs and globally unique row IDs. Cross-shard duplicate IDs outside
the winning candidates are **not** detected: the caller must enforce unique
IDs upstream. The summary's state is **not** composable across changing
centroids or training steps.

`split(j, generation, current_centroids)` requires the same logical step
and a centroid move of no more than `0.25 * observed_radius_j`. A unique
farthest point proposes

```
v = winning_point - current_centroid
delta = epsilon * min(1, observed_radius / ||v||) * v
donor, child = current_centroid - delta, current_centroid + delta
```

with `epsilon=1e-4`. Missing rows, zero residual, close top-two scores,
excessive drift, changing generation and FP32 rounding collapse cause explicit
abstention, not a coordinate-axis fallback.

Under a common similarity transform with assignments and IDs held fixed, a
unique winning point and this formula are similarity-equivariant **in exact
arithmetic**. This is not a global guarantee in FP32 or an affine-equivariant
property of the entire e-means assignment/update system.

## Run

```sh
python -m pip install numpy==2.3.5
python benchs/e_means/revival_stream_test.py
python benchs/e_means/revival_stream_witness.py
# Optional compiled-FAISS *assignment* parity:
python -m pip install faiss-cpu==1.15.1
python benchs/e_means/revival_stream_witness.py --faiss
```

The `--faiss` test checks that the manually supplied labels match nearest
centroids from `faiss.knn` on the seeded, well-separated example. It does
**not** exercise FAISS e-means `accumulate`, `update`, revival or kernels.

## What is missing

- Couple donor witnesses to the *same* assignment frame and logical batch
  as `update(stats)` without stale state or inconsistent chunk IDs
- Enforce globally unique IDs across ranks, handle empty/tied/quantized cases
  and validate server/GPU/low-precision backends
- Record/persist a common floating coordinate origin where appropriate;
  Stage 5 showed this affects FP32 representability
- Compare memory, communication, throughput, end-to-end training MSE,
  retrieval quality and actual revival *coverage* on representative data

No upstream performance claim follows from this experiment; do not merge
into the production clustering path without those controls.

## Post-hoc aggregate non-identifiability result

The additional [exact counterexample](REVIVAL_AGGREGATE_OBSTRUCTION.md) shows
that even equal counts, sums, and scalar squared-distance objectives can hide
different **unique** farthest witness vectors. That result is **derived**,
not a Stage-6 preregistered empirical success. It identifies an information
boundary in the existing update statistics; additional per-batch directional
information must be captured rather than inferred from aggregate counts/sums.
The corresponding unit test is labeled exploratory, not independent proof.

## Fork CI repair (2026-11-11)

The inherited Meta `Build` workflow in `jesusvilela/faiss` had a
pre-job `startup_failure` because two reusable pip-publishing call paths
requested `id-token: write` without the caller providing that permission.
[CI repair PR #3](https://github.com/jesusvilela/faiss/pull/3) was merged
**into this fork only**, preserving historical failure records. On the fork,
`Build` now runs a clearly labeled **limited CPU smoke** (not upstream
C++/GPU/wheel coverage) and the opt-in checks present in the PR checkout.
The dedicated experimental research workflow remains the primary CI for this
PR. Do not interpret a green fork `Build` check as validation of Meta's
private runners, full native build matrix, or a production clustering change.
