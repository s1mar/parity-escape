"""Re-extract parity inputs from stored raw responses and re-score, without regenerating.

Exists because the parser changed twice after generation had already started, and the first time
that happened every suite had to be regenerated purely because only the parsed result had been
kept. Raw responses are now stored, so a parser change costs a re-score (cheap, local execution)
instead of a re-run (expensive, model calls).

Only records whose parsed inputs actually CHANGE are re-scored, so a no-op parser change costs
almost nothing.

Usage:  python reparse.py <python-exe> [--mode plain|targeted]
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench      # noqa: E402
import models     # noqa: E402
import validate as V  # noqa: E402,N812  (reuse score_suite so scoring cannot drift)

argv = sys.argv[1:]
PY = argv.pop(0)
MODE = "plain"
if "--mode" in argv:
    i = argv.index("--mode")
    MODE = argv[i + 1]
    del argv[i:i + 2]

# --force re-scores every record, not only those whose parse changed. Needed when the DOMAIN
# rule changes rather than the parser: the stored verdicts were computed under the old rule and
# the per-input outputs are not kept, so they cannot be recomputed without re-executing.
FORCE = "--force" in argv
if FORCE:
    argv.remove("--force")

D = bench.ROOT / "data" / ("parity" if MODE == "plain" else f"parity_{MODE}")
WORK = bench.HERE / "_build"


def main() -> None:
    problems = {p["task_id"]: p for p in
                json.loads((bench.ROOT / "data" / "reference.json").read_text(encoding="utf-8"))}
    trans = {}
    for m in models.MODELS:
        tp = bench.ROOT / "data" / "translations" / f"{m.mid}.json"
        if tp.exists():
            for t in json.loads(tp.read_text(encoding="utf-8")):
                trans[(m.mid, t["task_id"])] = t

    if not D.is_dir():
        print("no parity data for mode", MODE)
        return

    for path in sorted(D.glob("*.json")):
        recs = json.loads(path.read_text(encoding="utf-8"))
        changed = rescored = became_ok = 0
        t0 = time.time()

        for r in recs:
            raw = r.get("raw")
            if not raw:
                continue
            p = problems.get(r["task_id"])
            if p is None:
                continue
            new = models.extract_inputs(raw, len(p["java_ptags"]), p["param_types"])
            new = new[:bench.K_PARITY] if new else None
            old = r.get("inputs")
            if new == old and not FORCE:
                continue
            changed += 1

            if not new:
                r["status"] = "unparseable"
                r.pop("inputs", None)
                continue

            t = trans.get((r["translator"], r["task_id"]))
            if t is None or not t.get("compile_ok"):
                r["status"] = "no_compile"
                continue
            wd = WORK / r["translator"] / r["task_id"].replace("/", "_")
            if not (wd / "Solution.class").exists():
                ok, _ = bench.compile_java(t["java"], wd)
                if not ok:
                    r["status"] = "no_compile"
                    continue
            was = r.get("status")
            r["status"] = "ok"
            r["inputs"] = new
            r.update(V.score_suite(p, wd, new))
            rescored += 1
            if was != "ok":
                became_ok += 1

        path.write_text(json.dumps(recs), encoding="utf-8")
        ok = [r for r in recs if r.get("status") == "ok"]
        deg = sum(1 for r in ok if r.get("full", {}).get("degenerate"))
        caught = sum(1 for r in ok if r.get("full", {}).get("caught"))
        print(f"{path.name:14s} {len(recs):4d} recs  changed {changed:4d}  rescored {rescored:4d}"
              f"  newly-ok {became_ok:3d}  | ok {len(ok):4d} degenerate {deg:4d} caught {caught:4d}"
              f"  ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
