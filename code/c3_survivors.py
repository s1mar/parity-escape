"""Are the C3 survivors equivalent mutants, or genuine misses by the strong oracle?

C3 injects a known divergence into translations the oracle called clean and requires the oracle to
flag it. Two of twenty survived, both relational boundary flips (>= to >, <= to <). Those are the
classic shape of an EQUIVALENT mutant: if the boundary value is unreachable at that site, the
flip changes no behaviour and no oracle of any strength could catch it.

"Probably equivalent" is a claim, so it gets tested. Each survivor is re-run against a much
larger and differently-seeded input set. If a divergence appears, the oracle genuinely missed it
and C3 should be read as 18/20 with two real misses. If none appears across many more inputs, the
mutant is very likely equivalent and C3 is better read as 18 of 18 non-equivalent injections.
Either way the paper reports the raw 18/20 as well.

Usage:  python c3_survivors.py <python-exe>
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench      # noqa: E402
import controls   # noqa: E402

PY = sys.argv[1]
N_EXTRA = 8000
SEEDS = (101, 202, 303)


def main() -> None:
    c3 = json.loads((bench.ROOT / "data" / "control_c3.json").read_text(encoding="utf-8"))
    survivors = [r for r in c3["rows"] if not r["caught"]]
    if not survivors:
        print("no survivors to investigate")
        return

    problems = {p["task_id"]: p for p in
                json.loads((bench.ROOT / "data" / "reference.json").read_text(encoding="utf-8"))}
    out = []

    for s in survivors:
        tid, mid = s["task_id"], s["model"]
        p = problems[tid]
        tp = bench.ROOT / "data" / "translations" / f"{mid}.json"
        java = next(t["java"] for t in json.loads(tp.read_text(encoding="utf-8"))
                    if t["task_id"] == tid)
        # reproduce the same injection deterministically: controls.run_c3 seeds a shared RNG, so
        # rather than replay it, re-apply the SAME named injection and take the first site
        jsrc, desc = None, s["injection"]
        for pat, rep, d in controls.INJECTIONS:
            if d == desc:
                import re
                m = re.search(pat, java)
                if m:
                    jsrc = java[:m.start()] + rep + java[m.end():]
                break
        if jsrc is None:
            out.append({"task_id": tid, "model": mid, "injection": desc,
                        "verdict": "could not re-apply injection"})
            continue

        wd = bench.HERE / "_ctl" / "c3surv" / f"{mid}_{tid.replace('/', '_')}"
        ok, err = bench.compile_java(jsrc, wd)
        if not ok:
            out.append({"task_id": tid, "model": mid, "injection": desc,
                        "verdict": "injected variant does not compile", "err": err[:200]})
            continue

        prob = bench.Problem(task_id=tid, idx=p["idx"], entry_point=p["entry_point"],
                             param_types=p["param_types"], return_type=p["return_type"],
                             source=p["source"], seed_inputs=[])
        found = 0
        tried = 0
        for seed in SEEDS:
            rng_inputs = bench.fuzz_inputs(prob, n=N_EXTRA // len(SEEDS), seed=seed)
            pr = bench.run_python(PY, p["source"], p["entry_point"], rng_inputs, budget=90.0)
            live = [k for k, x in enumerate(pr) if x.get("ok")]
            jr = bench.run_java(wd, p["entry_point"], p["java_ptags"],
                                [rng_inputs[k] for k in live], budget=90.0)
            tried += len(live)
            for pos, k in enumerate(live):
                if bench.classify(pr[k], jr[pos], p["return_type"]) == "DIVERGE":
                    found += 1
        out.append({
            "task_id": tid, "model": mid, "injection": desc,
            "extra_inputs_tried": tried, "divergences_found": found,
            "verdict": ("GENUINE ORACLE MISS: divergence exists and the oracle's input set did "
                        "not reach it") if found else
                       ("LIKELY EQUIVALENT MUTANT: no divergence over a much larger, differently "
                        "seeded input set"),
        })
        print(f"{mid:8s} {tid:12s} {desc:22s} tried {tried:6d} extra inputs, "
              f"found {found:4d} divergences", flush=True)

    (bench.ROOT / "data" / "control_c3_survivors.json").write_text(
        json.dumps(out, indent=1), encoding="utf-8")
    print()
    for o in out:
        print(f"  {o['task_id']}: {o['verdict']}")


if __name__ == "__main__":
    main()
