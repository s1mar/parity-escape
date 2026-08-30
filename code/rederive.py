"""Re-derive the headline numbers from the raw records, WITHOUT importing analyze.py.

analyze.py is the thing under test. If it has a bug, every number it emits is self-consistent and
every gate that traces those numbers is green. The only way to catch that class is to recompute
from the stored per-record files using an independently written path, and compare.

Deliberately duplicates the join logic rather than importing it. Duplication is the point.

Usage:  python rederive.py
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
D = ROOT / "data"
MODELS = ["qwen25", "dscoder", "mistral", "llama3", "gemini"]


def load(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def main() -> int:
    # ---- oracle verdicts, keyed by (translator, task)
    oracle = {}
    for m in MODELS:
        for r in load(D / "oracle" / f"{m}.json"):
            oracle[(m, r["task_id"])] = r

    # ---- every parity suite, every mode
    suites = []
    for mode, sub in (("plain", "parity"), ("targeted", "parity_targeted"),
                      ("random", "parity_random")):
        d = D / sub
        if not d.is_dir():
            continue
        for f in sorted(d.glob("*.json")):
            for r in load(f):
                r["_mode"] = mode
                suites.append(r)

    # ---- who caught what (for the union / leave-one-out ground truth)
    caught_by = defaultdict(set)
    for r in suites:
        if r.get("status") == "ok" and r.get("full", {}).get("caught"):
            caught_by[(r["translator"], r["task_id"])].add(r["validator"])

    # ---- RQ1 escape rate over PLAIN suites, recomputed from scratch
    n_passed = n_escaped = 0
    for r in suites:
        if r["_mode"] != "plain" or r.get("status") != "ok":
            continue
        full = r.get("full") or {}
        if full.get("degenerate"):
            continue
        o = oracle.get((r["translator"], r["task_id"]))
        if o is None or o.get("status") != "ok" or o.get("divergent") is None:
            continue
        if full.get("caught"):
            continue                                   # not a pass
        n_passed += 1
        union_div = bool(o["divergent"]) or bool(caught_by.get(
            (r["translator"], r["task_id"])))
        if union_div:
            n_escaped += 1
    escape = n_escaped / n_passed if n_passed else float("nan")

    # ---- divergence prevalence, straight off the oracle files
    prev = {}
    for m in MODELS:
        rs = [v for (mid, _t), v in oracle.items()
              if mid == m and v.get("status") == "ok"]
        div = sum(1 for v in rs if v.get("divergent"))
        prev[m] = (len(rs), div, 100 * div / len(rs) if rs else float("nan"))

    # ---- suite health, counted directly
    plain = [r for r in suites if r["_mode"] == "plain"]
    health = {
        "total": len(plain),
        "ok": sum(1 for r in plain if r.get("status") == "ok"),
        "unparseable": sum(1 for r in plain if r.get("status") == "unparseable"),
        "call_failed": sum(1 for r in plain if r.get("status") == "call_failed"),
        "degenerate": sum(1 for r in plain if r.get("status") == "ok"
                          and r.get("full", {}).get("degenerate")),
    }

    res = load(D / "results.json")
    print("re-derived from raw records, independent of analyze.py")
    print("=" * 70)
    rows = [
        ("escape rate %", 100 * escape, 100 * res["rq1"]["escape_rate"]),
        ("  n escaped", n_escaped, res["rq1"]["n_escaped"]),
        ("  n passed", n_passed, res["rq1"]["n_passed"]),
        ("usable translations", sum(v[0] for v in prev.values()),
         sum(v["n_usable"] for v in res["divergence_prevalence"].values())),
        ("divergent translations", sum(v[1] for v in prev.values()),
         sum(v["n_divergent"] for v in res["divergence_prevalence"].values())),
        ("suites total", health["total"], res["suite_health"]["n_total"]),
        ("suites ok", health["ok"], res["suite_health"]["n_ok"]),
        ("suites unparseable", health["unparseable"], res["suite_health"]["n_unparseable"]),
        ("suites degenerate", health["degenerate"], res["suite_health"]["n_degenerate"]),
    ]
    bad = 0
    print(f"{'quantity':26s} {'re-derived':>14s} {'results.json':>14s}  match")
    for name, mine, theirs in rows:
        ok = abs(float(mine) - float(theirs)) < 0.05
        bad += 0 if ok else 1
        print(f"{name:26s} {mine:>14.2f} {theirs:>14.2f}  {'yes' if ok else 'NO'}")

    print()
    print("divergence rate by translator (re-derived):")
    for m in MODELS:
        n, d, pct = prev[m]
        ref = 100 * res["divergence_prevalence"][m]["divergence_rate"]
        ok = abs(pct - ref) < 0.05
        bad += 0 if ok else 1
        print(f"  {m:9s} {d:3d}/{n:3d} = {pct:5.1f}%   results.json {ref:5.1f}%  "
              f"{'yes' if ok else 'NO'}")

    print()
    print("ALL RE-DERIVED VALUES MATCH" if bad == 0 else f"{bad} MISMATCH(ES)")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
