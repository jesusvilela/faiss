# Copyright (c) contributors to the FAISS e-means covariance study.
# Licensed under the repository's MIT license.
"""Stage-5 preregistered shared-local-frame FP32 representability control.

See REVIVAL_CENTERING_PREREG.md. Not full e-means or an API implementation.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from revival_quality_control import (
    SEEDS, SPREADS, SHIFTS, candidate_state, heldout_mse, lloyd_once, sample,
    run as stage4_run,
)


def center_view(train, holdout, centroids):
    origin = train.astype(np.float64).mean(axis=0).astype(np.float32)
    o = origin.astype(np.float64)
    return tuple((a.astype(np.float64) - o).astype(np.float32)
                 for a in (train, holdout, centroids))


def measure(train, holdout, initial, method):
    revived, abstained = candidate_state(train, initial, method)
    before = heldout_mse(holdout, revived)
    after, occupied = lloyd_once(train, revived)
    return {
        "abstained": bool(abstained),
        "mse_immediate": before,
        "mse_after_one_lloyd": heldout_mse(holdout, after),
        "nonempty_after_assignment": occupied,
    }


def execute():
    rows = []
    for seed in SEEDS:
        for pattern, spread in SPREADS.items():
            for shift in SHIFTS:
                raw = sample(seed, spread, shift)
                centered = center_view(*raw)
                for name, view in (("raw", raw), ("centered", centered)):
                    for method in ("original", "witness", "none"):
                        rows.append({
                            "seed": seed, "spread": pattern,
                            "shift": int(shift), "view": name, "method": method,
                            **measure(*view, method),
                        })
    # Invariant check: the Stage-5 raw view must reproduce every Stage-4
    # final MSE/abstention without a code-path or seed change.
    baseline = {(r["seed"], r["spread"], r["shift"], r["method"]): r
                for r in stage4_run()}
    for r in rows:
        if r["view"] == "raw":
            prev = baseline[(r["seed"], r["spread"], r["shift"], r["method"])]
            if (r["mse_after_one_lloyd"] != prev["mse_after_one_lloyd"]
                    or r["abstained"] != prev["abstained"]):
                raise AssertionError("Stage-4 provenance mismatch")
    return rows


def summarize(rows):
    groups = {}
    largest_baseline_change = 0.0
    shift10000_raw = shift10000_centered = 0
    all_centered_witness = []
    criterion_failures = []
    for pattern in SPREADS:
        for shift in SHIFTS:
            regime = "%s_shift_%d" % (pattern, int(shift))
            groups[regime] = {}
            for view in ("raw", "centered"):
                subset = [r for r in rows if r["view"] == view and
                          r["spread"] == pattern and r["shift"] == int(shift)]
                originals = {r["seed"]: r for r in subset if r["method"] == "original"}
                witnesses = {r["seed"]: r for r in subset if r["method"] == "witness"}
                no_rev = {r["seed"]: r for r in subset if r["method"] == "none"}
                ratios = [witnesses[s]["mse_after_one_lloyd"] /
                          originals[s]["mse_after_one_lloyd"] for s in originals]
                groups[regime][view] = {
                    "abstentions": sum(int(r["abstained"]) for r in witnesses.values()),
                    "median_witness_over_original": float(np.median(ratios)),
                    "worst_witness_over_original": float(np.max(ratios)),
                    "median_witness_mse": float(np.median([
                        r["mse_after_one_lloyd"] for r in witnesses.values()])),
                    "median_original_mse": float(np.median([
                        r["mse_after_one_lloyd"] for r in originals.values()])),
                }
                if view == "centered":
                    all_centered_witness.extend(witnesses.values())
                    if float(np.median(ratios)) > 1.01:
                        criterion_failures.append(regime + ": centered median quality > 1% worse")
                if int(shift) == 10000:
                    n = groups[regime][view]["abstentions"]
                    if view == "raw":
                        shift10000_raw += n
                    else:
                        shift10000_centered += n
            raw_none = {r["seed"]: r for r in rows if r["method"] == "none" and
                        r["view"] == "raw" and r["spread"] == pattern and
                        r["shift"] == int(shift)}
            center_none = {r["seed"]: r for r in rows if r["method"] == "none" and
                           r["view"] == "centered" and r["spread"] == pattern and
                           r["shift"] == int(shift)}
            for seed in raw_none:
                before = raw_none[seed]["mse_after_one_lloyd"]
                after = center_none[seed]["mse_after_one_lloyd"]
                largest_baseline_change = max(largest_baseline_change,
                                              abs(after - before) / before)
    if shift10000_centered > 2:
        criterion_failures.append("centered shift10000 abstentions > 2")
    if largest_baseline_change > .001:
        criterion_failures.append("no-revival MSE representation distortion > 0.1%")
    overall_abst = sum(int(x["abstained"]) for x in all_centered_witness)
    if overall_abst / len(all_centered_witness) > .1:
        criterion_failures.append("centered total abstention > 10%")
    return {
        "preregistration": "REVIVAL_CENTERING_PREREG.md",
        "scope": "shared local chart from training data, single revival + one Lloyd",
        "scenarios_per_view": len(all_centered_witness),
        "raw_abstentions_shift10000": shift10000_raw,
        "centered_abstentions_shift10000": shift10000_centered,
        "centered_abstentions_total": overall_abst,
        "max_rel_change_no_revival_mse": largest_baseline_change,
        "frozen_criteria_failures": criterion_failures,
        "candidate_production_ready": False,
        "regimes": groups,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    rows = execute()
    summary = summarize(rows)
    if args.output:
        args.output.write_text(json.dumps({"summary": summary, "rows": rows},
                                          indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
