"""Miss rate BY divergence category.

The paper asserts that cross-language divergences are "precisely the behaviours a hand-written
test suite is least likely to probe" and that generic testing is "worst precisely there". No detection rate by category was reported anywhere, so the
claim was unsupported as written. It is also computable from data already on disk, so it gets
computed rather than softened on a guess.

For every divergent translation with a taxonomy label, how often did a parity suite miss it?
Split by the cross-language / logic grouping the paper already uses.

Usage:  python category_detection.py
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analyze  # noqa: E402
import bench    # noqa: E402

D = bench.ROOT / "data"


def main() -> None:
    tax = {(t["model"], t["task_id"]): t
           for t in json.loads((D / "taxonomy.json").read_text(encoding="utf-8"))}

    rows = [r for r in analyze.joined("plain", "full") if r["oracle_divergent"]]

    by_cat = defaultdict(lambda: [0, 0])       # category -> [missed, total]
    by_grp = defaultdict(lambda: [0, 0])
    for r in rows:
        t = tax.get((r["translator"], r["task_id"]))
        if not t:
            continue
        miss = 0 if r["caught"] else 1
        by_cat[t["category"]][0] += miss
        by_cat[t["category"]][1] += 1
        by_grp[t.get("group", "?")][0] += miss
        by_grp[t.get("group", "?")][1] += 1

    out = {
        "by_group": {g: {"n": v[1], "miss_rate": v[0] / v[1]}
                     for g, v in by_grp.items() if v[1]},
        "by_category": {c: {"n": v[1], "miss_rate": v[0] / v[1]}
                        for c, v in sorted(by_cat.items(), key=lambda kv: -kv[1][1])
                        if v[1] >= 10},
    }
    (D / "category_detection.json").write_text(json.dumps(out, indent=1), encoding="utf-8")

    print("miss rate by GROUP (on oracle-confirmed divergent cells)")
    for g, v in sorted(out["by_group"].items()):
        print(f"  {g:16s} n={v['n']:4d}  miss {100*v['miss_rate']:5.1f}%")
    print()
    print("miss rate by CATEGORY (n >= 10)")
    for c, v in out["by_category"].items():
        print(f"  {c:46s} n={v['n']:4d}  miss {100*v['miss_rate']:5.1f}%")

    g = out["by_group"]
    if "cross_language" in g and "logic" in g:
        d = 100 * (g["cross_language"]["miss_rate"] - g["logic"]["miss_rate"])
        print()
        print(f"cross-language minus logic: {d:+.1f} percentage points")
        print("  -> supports 'testing is worst precisely there'" if d > 0
              else "  -> does NOT support it; the claim must be softened")


if __name__ == "__main__":
    main()
