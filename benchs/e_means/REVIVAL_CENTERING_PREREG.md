# Stage 5 preregistration: shared local-coordinate representation

Date: 2026-11-11. Committed before any Stage-5 run. The Stage-4
negative result remains authoritative and must not be overwritten.

## Question

Does a common translation of the entire already-FP32 dataset into a local
coordinate chart restore the representability and coverage of bounded
witness revival at large global offsets? This is a representation control,
NOT a numerical precision theorem and NOT a FAISS runtime implementation.

## Frozen data and algorithms

Reuse the exact Stage-4 60 scenarios, seeds, Gaussian parameters, and frozen
method definitions (original, witness, no revival). No new or selected seeds.

Compare raw and recentered views of the same FP32 arrays:
    o = mean(train, axis=0, dtype=float64).astype(float32)
    centered = (raw.astype(float64) - o.astype(float64)).astype(float32)
Apply the *same* origin to train, holdout and all initial centroids. Do not
estimate any origin from the holdout set. Measure distances in FP64 on the
represented FP32 vectors, as in Stage 4.

## Report

For every scenario, retain original/witness/none immediate and one-Lloyd
held-out MSE, abstentions and occupied centroids in both views.
Report per-regime witness-vs-original MSE ratios, number of abstentions,
and difference in the *no-revival* MSE between representations.

## Frozen decision criteria

- Representation repair: witness abstentions at shift 10000 fall from
  20/20 in raw coordinates to <=2/20 in the centered view.
- Centering must not materially distort the frozen no-revival holdout
  objective: max relative difference <= 0.1% per scenario.
- Candidate remains NONPRODUCTION if centered median witness MSE is
  > original by 1% in any regime, or witness abstains >10% overall.

A pass justifies further FAISS input-space tests; it does not justify changing
backend code or using a moving-frame representation without explicit query
transforms, origin persistence, multi-shard consistency and overflow handling.

## Limitation and falsifiers

Subtracting an origin from already quantized FP32 samples cannot restore
lost input information. The shared frame is estimated from *training* data
and may become stale as distributions drift. This procedure does not promise
equivariance under arbitrary translations; at huge offsets, distinct input
points may have already collapsed before centering. Failed outcomes are kept.
