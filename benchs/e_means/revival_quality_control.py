# Copyright (c) contributors to the FAISS e-means covariance study.
# Licensed under the repository's MIT license.
"""Pre-registered isolated revival-quality control (NOT full e-means training).

See REVIVAL_QUALITY_PREREG.md before interpreting outputs.

Usage:
    python benchs/e_means/revival_quality_control.py
    python benchs/e_means/revival_quality_control.py --output study.json
"""
import argparse
import json
from pathlib import Path
import numpy as np

from revival_frame_audit import farthest_witness, reference_split, witness_split

SEEDS = range(10)
SPREADS = {
    "isotropic": (0.30, 0.30, 0.30),
    "anisotropic": (0.15, 0.70, 0.25),
}
SHIFTS = (0.0, 100.0, 10000.0)
MEANS = np.array([[5., 0., 0.], [-5., 0., 0.]], dtype=np.float64)
INITIAL = np.array([[5., 0., 0.], [-5., 0., 0.], [45., 40., 40.]],
                   dtype=np.float64)


def sample(seed, spread, shift):
    rng = np.random.default_rng(seed)
    scale = np.asarray(spread, dtype=np.float64)
    train = np.vstack([rng.normal(loc=mu, scale=scale, size=(256, 3))
                       for mu in MEANS])
    holdout = np.vstack([rng.normal(loc=mu, scale=scale, size=(256, 3))
                         for mu in MEANS])
    return ((train + shift).astype(np.float32),
            (holdout + shift).astype(np.float32),
            (INITIAL + shift).astype(np.float32))


def squared_distances(x, c):
    delta = x.astype(np.float64)[:, None, :] - c.astype(np.float64)[None, :, :]
    return np.sum(delta * delta, axis=2)


def heldout_mse(x, c):
    return float(squared_distances(x, c).min(axis=1).mean())


def lloyd_once(train, c):
    labels = squared_distances(train, c).argmin(axis=1)
    updated = c.copy()
    counts = np.bincount(labels, minlength=len(c))
    for j, count in enumerate(counts):
        if count:
            updated[j] = train[labels == j].astype(np.float64).mean(axis=0)
    return updated, int(np.count_nonzero(counts))


def candidate_state(train, c, method):
    out = c.copy()
    if method == "none":
        return out, True
    if method == "original":
        pair = reference_split(c[[0, 2]])
    elif method == "witness":
        labels = squared_distances(train, c).argmin(axis=1)
        witness, radius = farthest_witness(train, labels, c[0], donor_id=0)
        if witness is None:
            return out, True
        try:
            pair = witness_split(c[[0, 2]], witness, radius)
        except ValueError:
            return out, True
    else:
        raise ValueError(method)
    out[0], out[2] = pair
    return out, False


def run(seeds=SEEDS):
    rows = []
    for seed in seeds:
        for pattern, spread in SPREADS.items():
            for shift in SHIFTS:
                train, holdout, initial = sample(seed, spread, shift)
                for method in ("original", "witness", "none"):
                    revived, abstained = candidate_state(train, initial, method)
                    immediate = heldout_mse(holdout, revived)
                    after, noccupied = lloyd_once(train, revived)
                    rows.append({
                        "seed": seed,
                        "spread": pattern,
                        "shift": int(shift),
                        "method": method,
                        "abstained": bool(abstained),
                        "mse_immediate": immediate,
                        "mse_after_one_lloyd": heldout_mse(holdout, after),
                        "nonempty_after_assignment": noccupied,
                        "donor_step": float(np.linalg.norm(
                            revived[0].astype(np.float64) -
                            initial[0].astype(np.float64))),
                    })
    return rows


def summarize(rows):
    regimes = {}
    for pattern in SPREADS:
        for shift in SHIFTS:
            subset = [x for x in rows if x["spread"] == pattern and x["shift"] == int(shift)]
            source = {x["seed"]: x for x in subset if x["method"] == "original"}
            witness = {x["seed"]: x for x in subset if x["method"] == "witness"}
            control = {x["seed"]: x for x in subset if x["method"] == "none"}
            ratios = np.asarray([
                witness[s]["mse_after_one_lloyd"] / source[s]["mse_after_one_lloyd"]
                for s in sorted(source)
            ], dtype=float)
            gain_vs_none = np.asarray([
                witness[s]["mse_after_one_lloyd"] / control[s]["mse_after_one_lloyd"]
                for s in sorted(source)
            ], dtype=float)
            key = "%s_shift_%d" % (pattern, int(shift))
            regimes[key] = {
                "cases": len(source),
                "abstentions": int(sum(x["abstained"] for x in witness.values())),
                "median_original_mse_after": float(np.median(
                    [x["mse_after_one_lloyd"] for x in source.values()])),
                "median_witness_mse_after": float(np.median(
                    [x["mse_after_one_lloyd"] for x in witness.values()])),
                "median_no_revival_mse_after": float(np.median(
                    [x["mse_after_one_lloyd"] for x in control.values()])),
                "median_witness_over_original": float(np.median(ratios)),
                "worst_witness_over_original": float(np.max(ratios)),
                "median_witness_over_no_revival": float(np.median(gain_vs_none)),
            }
    witness_rows = [r for r in rows if r["method"] == "witness"]
    failed = []
    for key, result in regimes.items():
        if result["abstentions"] / result["cases"] > .2:
            failed.append(key + ": abstention > 20%")
        if result["median_witness_over_original"] > 1.01:
            failed.append(key + ": witness median MSE > original by 1%")
    if sum(x["abstained"] for x in witness_rows) / len(witness_rows) > .1:
        failed.append("overall abstention > 10%")
    return {
        "protocol": "REVIVAL_QUALITY_PREREG.md",
        "scope": "single revival + one Lloyd update, synthetic holdout, not e-means",
        "scenarios": len(witness_rows),
        "failures_under_frozen_decision_rule": failed,
        "production_replacement_supported": False,
        "regimes": regimes,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rows = run()
    summary = summarize(rows)
    report = {"summary": summary, "rows": rows}
    if args.output:
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
