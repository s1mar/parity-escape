"""Stage 2: run the strong oracle. For every compiled translation, execute all N_FUZZ inputs and
record whether it diverges from the legacy source, and on which inputs.

This is the ground truth against which every parity suite is later scored. It is deliberately
computed once and reused, so a parity suite can never be evaluated against a differently-seeded
oracle.

Records the first divergent inputs per translation so RQ3's taxonomy is built from real observed
divergences rather than from a guess about what LLMs get wrong.

Usage:  python evaluate.py <python-exe> [model_id ...]
"""
from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench      # noqa: E402
import models     # noqa: E402

PY = sys.argv[1]
WANT = sys.argv[2:] or [m.mid for m in models.MODELS]
OUTDIR = bench.ROOT / "data" / "oracle"
WORK = bench.HERE / "_build"
KEEP_EXAMPLES = 12


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    problems = {p["task_id"]: p for p in
                json.loads((bench.ROOT / "data" / "reference.json").read_text(encoding="utf-8"))}

    for mid in WANT:
        tpath = bench.ROOT / "data" / "translations" / f"{mid}.json"
        if not tpath.exists():
            print(f"!! no translations for {mid}, skipping")
            continue
        trans = json.loads(tpath.read_text(encoding="utf-8"))
        outp = OUTDIR / f"{mid}.json"
        done = {}
        if outp.exists():
            done = {r["task_id"]: r for r in json.loads(outp.read_text(encoding="utf-8"))}

        todo = [t for t in trans if t["task_id"] not in done]
        print(f"\n=== oracle {mid}: {len(done)} done, {len(todo)} to go ===", flush=True)
        t0 = time.time()

        for i, t in enumerate(todo):
            tid = t["task_id"]
            p = problems[tid]
            rec = {"task_id": tid, "model": mid, "compile_ok": bool(t.get("compile_ok"))}

            if not t.get("compile_ok"):
                rec["status"] = "no_compile"
                done[tid] = rec
                continue

            wd = WORK / mid / tid.replace("/", "_")
            if not (wd / "Solution.class").exists():
                ok, err = bench.compile_java(t["java"], wd)
                if not ok:
                    rec["status"] = "no_compile"
                    rec["compile_ok"] = False
                    done[tid] = rec
                    continue

            # Only inputs on which the SOURCE returned a value can inform equivalence, so the
            # target is executed on exactly those. Inputs where the source raised, timed out or
            # exhausted a resource are out of domain by definition and running them would change
            # nothing while costing a JVM call each. On problems whose source hangs on most
            # inputs this is the difference between seconds and minutes, five times over.
            pr_all = p["source_out"]
            live = [k for k, x in enumerate(pr_all) if x.get("ok")]
            jr_live = bench.run_java(wd, p["entry_point"], p["java_ptags"],
                                     [p["inputs"][k] for k in live])

            verdicts = ["OUT_OF_DOMAIN"] * len(pr_all)
            jr = [{"ok": False, "e": "NotRun"} for _ in pr_all]
            for pos, k in enumerate(live):
                jr[k] = jr_live[pos]
                verdicts[k] = bench.classify(pr_all[k], jr_live[pos], p["return_type"])
            # inputs the source rejected outright are recorded as such, not as out of domain
            for k, x in enumerate(pr_all):
                if not x.get("ok") and x.get("e") not in bench.IN_DOMAIN_EXCLUDED:
                    verdicts[k] = "SOURCE_RAISED"

            pr = pr_all
            c = Counter(verdicts)
            div_idx = [k for k, v in enumerate(verdicts) if v == "DIVERGE"]
            comparable = c["AGREE"] + c["DIVERGE"]
            rec["n_live_inputs"] = len(live)
            # Non-termination is excluded from divergence on purpose: a 2 s per-call limit cannot
            # tell "loops forever" from "slower than the source", and counting the latter as a
            # semantic difference would be a performance claim wearing a correctness costume. It
            # is counted separately, because a target that hangs where the source returns is
            # arguably the most severe failure a migration can have, and silently folding it into
            # "out of domain" would hide it.
            rec["n_target_timeout"] = sum(
                1 for k in live if jr[k].get("e") in ("Timeout", "TooManyTimeouts", "Budget"))

            rec.update({
                "status": "ok",
                "n_agree": c["AGREE"], "n_diverge": c["DIVERGE"],
                "n_out_of_domain": c["OUT_OF_DOMAIN"], "n_source_raised": c["SOURCE_RAISED"],
                "n_comparable": comparable,
                "divergent": bool(div_idx),
                "diverge_idx": div_idx[:200],
                "examples": [
                    {"in": p["inputs"][k], "py": pr[k], "jv": jr[k]} for k in div_idx[:KEEP_EXAMPLES]
                ],
            })
            # a translation with no comparable inputs cannot be judged either way
            if comparable == 0:
                rec["status"] = "no_comparable"
                rec["divergent"] = None
            done[tid] = rec

            if (i + 1) % 10 == 0 or i + 1 == len(todo):
                outp.write_text(json.dumps(list(done.values())), encoding="utf-8")
                nd = sum(1 for r in done.values() if r.get("divergent"))
                nok = sum(1 for r in done.values() if r.get("status") == "ok")
                print(f"  [{i + 1:3d}/{len(todo)}] usable={nok} divergent={nd} "
                      f"({time.time() - t0:.0f}s)", flush=True)

        outp.write_text(json.dumps(list(done.values())), encoding="utf-8")
        nok = sum(1 for r in done.values() if r.get("status") == "ok")
        nd = sum(1 for r in done.values() if r.get("divergent"))
        print(f"=== {mid}: usable {nok}, divergent {nd} "
              f"({100 * nd / max(1, nok):.1f}% of usable) ===", flush=True)


if __name__ == "__main__":
    main()
