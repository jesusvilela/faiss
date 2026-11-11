# Stage 7: observing the real caller-driven NumPy e-means lifecycle

**Experimental and diagnostic only.** This proposal is a stacked extension
of [draft PR #2](https://github.com/jesusvilela/faiss/pull/2), and is
targeted at its `research/e-means-stream-witness` branch. Do not merge
into `main` before its parent. No FAISS production behavior, default
revival rule, or C++/GPU code has been changed.

## Why this is necessary

`NumpyEmeans.accumulate(x)` calls private `_accumulate_core()`, which
freezes the current centroid readout for one logical statistical batch, then
calls `_assign_rows(rows, centroids)` for each assignment chunk. It returns
sums/counts/objective, but ordinarily retains full labels only in the last
epoch when `capture_assignments=True`. Source
`_EmeansBase.update(stats)` applies EMA statistics then may revive unused
clusters on the fixed cadence, **before** the next assignment frame.

`ObservedNumpyEmeans` subclasses `_assign_rows`, delegates exactly once
to `super()._assign_rows`, records the returned labels and rows using the
Stage-6 `FixedFrameWitness`, and returns the untouched result. The
adapter does **not** call a second `faiss.knn` while collecting witnesses.
A separate `faiss.knn` oracle is used only in the test experiment.

The witness is valid only for its stored generation and reference frame.
After `update`, the diagnostic queries the candidate split with the new
centroid positions but **does not apply it**. It also checks that old
generation IDs abstain and that source revival counts and outputs are
identical with and without the observer.

## Reproduction

The installed `faiss-cpu` core must be paired with the *checked-out fork's*
`contrib/e_means` Python sources. CI installs `faiss-cpu==1.15.1`,
copies those sources into the installed package and runs:

```sh
python benchs/e_means/revival_stream_test.py
python benchs/e_means/revival_lifecycle_test.py
python benchs/e_means/revival_lifecycle_observer.py
```

The final script emits per-step witness coverage, abstention reasons and
cumulative **unchanged** source revival counts. Its computation is not
a speed benchmark.

## Measured/assumed boundary

- **Expected parity:** identical source and observed centroid/usage arrays,
  moving objective, epoch totals, and final output for revival enabled and
  disabled. This is the kill criterion in the
  [preregistration](REVIVAL_LIFECYCLE_PREREG.md).
- **Private hook:** stable only under the inspected checkout. The subclass
  does not promise an upstream-supported extension point.
- **Memory:** fixed per-cluster `FixedFrameWitness` arrays within the
  logical batch, with an O(chunk) ID array and per-point Python loop.
  Caller-held run history, when printed, is not bounded.
- **Actual donor and target selection:** still 100% source e-means. A donor
  with no witness in its current batch cannot be justified by last-touch
  statistics; the next experiment must quantify this.
- **No claims:** neither runtime improvement nor representative holdout
  quality nor multi-worker origin transport is demonstrated.
- **Historical evidence:** Stage-4 raw-FP32 witness quality **failed**;
  Stage-5 shared-origin precision **passed its limited controls**. Both
  remain valid findings and are not superseded here.

## Next falsifier

Test whether selected revival donors actually have a usable current-step
witness and whether the post-update drift gate accepts them, under multiple
update orders and class imbalances. Preregister before evaluating; if
coverage is low, do not replace the original split with this witness rule.
