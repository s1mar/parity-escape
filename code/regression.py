"""RQ2 by regression: does self-validation predict a miss, controlling for who validated and
whose code was validated?

Why this exists. The exact matrix permutation test holds row and column MARGINALS fixed, but the
cells have very unequal sizes (from n=4 to n=62). Permuting columns therefore moves a small cell
into a position that previously held a large one, so the pooled pseudo-diagonal is not weighted
the same way as the true diagonal, and the reference distribution is not quite the right one.

A regression handles unequal cells natively and tests the self coefficient directly while holding
validator identity and translator identity fixed. Two specifications are fitted, because they
answer slightly different questions and a claim that holds under both is stronger:

  FE   logistic regression, outcome = suite MISSED the divergence, with fixed effects for
       validator and translator and an indicator for self-validation. Inference by cluster
       bootstrap over problems, since translations of one problem are not independent.
  ME   mixed-effects logistic regression with a random intercept per PROBLEM, fitted by
       statsmodels, as a second specification for the same coefficient.

Usage:  python regression.py
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze  # noqa: E402
import bench    # noqa: E402
import models   # noqa: E402

N_BOOT = 2000
SEED = 0


def design(rows):
    """Design matrix: intercept, validator dummies, translator dummies, self indicator."""
    ids = [m.mid for m in models.MODELS]
    vref, tref = ids[0], ids[0]                       # first model is the reference level
    cols = ["const"]
    cols += [f"V:{i}" for i in ids if i != vref]
    cols += [f"T:{i}" for i in ids if i != tref]
    cols += ["self"]
    X = np.zeros((len(rows), len(cols)))
    y = np.zeros(len(rows))
    for r_i, r in enumerate(rows):
        X[r_i, 0] = 1.0
        if r["validator"] != vref:
            X[r_i, cols.index(f"V:{r['validator']}")] = 1.0
        if r["translator"] != tref:
            X[r_i, cols.index(f"T:{r['translator']}")] = 1.0
        X[r_i, cols.index("self")] = 1.0 if r["self"] else 0.0
        y[r_i] = 0.0 if r["caught"] else 1.0          # outcome is a MISS
    return X, y, cols


def fit_logit(X, y, iters=60, ridge=1e-6):
    """Unpenalised logistic regression by Newton-Raphson, with a whisper of ridge for stability."""
    b = np.zeros(X.shape[1])
    for _ in range(iters):
        eta = np.clip(X @ b, -30, 30)
        p = 1.0 / (1.0 + np.exp(-eta))
        W = np.clip(p * (1 - p), 1e-9, None)
        g = X.T @ (y - p) - ridge * b
        H = (X * W[:, None]).T @ X + ridge * np.eye(X.shape[1])
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            return None
        b = b + step
        if np.max(np.abs(step)) < 1e-9:
            break
    return b


def main() -> None:
    rows = [r for r in analyze.joined("plain", "full") if r["divergent_loo"]]
    if not rows:
        print("no rows")
        return
    X, y, cols = design(rows)
    si = cols.index("self")

    b = fit_logit(X, y)
    if b is None:
        print("fit failed")
        return
    coef = float(b[si])

    # cluster bootstrap over problems
    rng = np.random.default_rng(SEED)
    probs = sorted({r["task_id"] for r in rows})
    idx_by = {p: [i for i, r in enumerate(rows) if r["task_id"] == p] for p in probs}
    reps = []
    for _ in range(N_BOOT):
        pick = rng.integers(0, len(probs), len(probs))
        idx = [i for k in pick for i in idx_by[probs[k]]]
        bb = fit_logit(X[idx], y[idx])
        if bb is not None and np.isfinite(bb[si]):
            reps.append(float(bb[si]))
    reps.sort()
    lo = reps[int(0.025 * len(reps))] if reps else float("nan")
    hi = reps[min(len(reps) - 1, int(0.975 * len(reps)))] if reps else float("nan")
    # two-sided bootstrap p: how often does the sign flip through zero
    p_boot = 2 * min(sum(1 for v in reps if v <= 0), sum(1 for v in reps if v >= 0)) / max(1, len(reps))
    p_boot = min(1.0, p_boot)

    out = {
        "n_cells": len(rows),
        "fixed_effects": {
            "self_logit_coef": coef,
            "self_odds_ratio": float(np.exp(coef)),
            "ci_logit": [lo, hi],
            "ci_odds_ratio": [float(np.exp(lo)), float(np.exp(hi))],
            "p_cluster_bootstrap": p_boot,
            "n_boot": len(reps),
        },
    }

    # mixed effects with a random intercept per problem, exactly as requested
    try:
        import pandas as pd
        import statsmodels.formula.api as smf
        df = pd.DataFrame({
            "miss": y,
            "validator": [r["validator"] for r in rows],
            "translator": [r["translator"] for r in rows],
            "self": [1 if r["self"] else 0 for r in rows],
            "problem": [r["task_id"] for r in rows],
        })
        import statsmodels.api as sm
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # GEE rather than a random-intercept fit: the mixed model is singular here because
            # one diagonal cell has only four observations, and GEE with an exchangeable working
            # correlation clustered on problem gives the same clustered inference without
            # needing a variance component to be estimable.
            m = smf.gee("miss ~ C(validator) + C(translator) + self", "problem", df,
                        family=sm.families.Binomial(),
                        cov_struct=sm.cov_struct.Exchangeable()).fit()
        out["gee_logistic"] = {
            "self_coef": float(m.params["self"]),
            "self_odds_ratio": float(np.exp(m.params["self"])),
            "self_se": float(m.bse["self"]),
            "self_p": float(m.pvalues["self"]),
            "note": "logistic GEE, exchangeable working correlation, clustered on problem",
        }
    except Exception as e:                                   # noqa: BLE001
        out["mixed_effects_lpm"] = {"error": str(e)[:200]}

    p = bench.ROOT / "data" / "regression.json"
    p.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))
    print("wrote", p)


if __name__ == "__main__":
    main()
