"""Stage 3: the measurement. For every (validator, translator, problem) cell, ask the validator
for K parity test INPUTS, obtain the expected values by executing the legacy source on them, and
record whether the translation passes.

The validator never supplies an expected value. That is the point of the design: the oracle is
ground truth by construction, so the only thing being measured is the validator's power to pick
inputs that expose a difference.

Batched by validator, because Ollama holds one model in VRAM and interleaving forces a reload.

Usage:  python validate.py <python-exe> [--mode plain|targeted] [validator_id ...]
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench      # noqa: E402
import models     # noqa: E402

argv = sys.argv[1:]
PY = argv.pop(0)
MODE = "plain"
if "--mode" in argv:
    i = argv.index("--mode")
    MODE = argv[i + 1]
    del argv[i:i + 2]
# The mitigation arm (RQ4) is run only on the matrix DIAGONAL. Running divergence-targeted
# prompting over all 25 cells would double the whole experiment's cost for a question that only
# needs the diagonal: the actionable comparison is self+plain (the baseline practice) against
# cross+plain (mitigation A, already measured) and self+targeted (mitigation B).
SELF_ONLY = "--self-only" in argv
if SELF_ONLY:
    argv.remove("--self-only")
WANT = argv or [m.mid for m in models.MODELS]

PARITY_BUDGET_S = 15     # wall clock per side for scoring one K-input parity suite
OUTDIR = bench.ROOT / "data" / ("parity" if MODE == "plain" else f"parity_{MODE}")
WORK = bench.HERE / "_build"
TEMPLATE = models.VALIDATE if MODE == "plain" else models.VALIDATE_TARGETED


def score_suite(p: dict, java_dir: Path, inputs: list[list]) -> dict:
    """Run one generated parity suite and score it, at K and at each sensitivity prefix.

    Tight execution budget (SPEC Amendment 6). A parity suite is only K inputs, but the
    translation it is scoring may hang on several of them, and each hang costs a per-call timeout
    plus a worker restart. Left unbounded that made a single pathological cell cost most of a
    minute, and the matrix has thousands of cells. The budget bounds the cost per side; inputs
    not reached are out of domain, exactly as elsewhere.

    The bound depends only on the TRANSLATION's behaviour, never on which validator supplied the
    inputs, so it applies identically to every validator scoring the same translation and cannot
    move the paired self-versus-cross comparison.
    """
    pr = bench.run_python(PY, p["source"], p["entry_point"], inputs, budget=PARITY_BUDGET_S)
    jr = bench.run_java(java_dir, p["entry_point"], p["java_ptags"], inputs,
                        budget=PARITY_BUDGET_S)
    verdicts = [bench.classify(a, b, p["return_type"]) for a, b in zip(pr, jr)]

    def at(k: int) -> dict:
        v = verdicts[:k]
        indom = [x for x in v if x in ("AGREE", "DIVERGE")]
        caught = any(x == "DIVERGE" for x in v)
        return {"n_indomain": len(indom), "caught": caught,
                "degenerate": len(indom) < bench.MIN_PARITY_INDOMAIN}

    out = {"verdicts": verdicts, "n_inputs": len(inputs)}
    for k in bench.K_SENSITIVITY:
        out[f"k{k}"] = at(k)
    out["full"] = at(len(verdicts))
    return out


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    problems = {p["task_id"]: p for p in
                json.loads((bench.ROOT / "data" / "reference.json").read_text(encoding="utf-8"))}
    translators = [m.mid for m in models.MODELS]

    # seed-input counts, so the random baseline can skip the benchmark's own inputs
    seed_sets = {}
    for e in json.loads((bench.ROOT / "data" / "eligible.json").read_text(encoding="utf-8")):
        seed_sets[e["task_id"]] = {json.dumps(s, sort_keys=True)
                                   for s in (e.get("seed_inputs") or [])}

    def n_seed_prefix(p: dict) -> int:
        s = seed_sets.get(p["task_id"], set())
        n = 0
        for a in p["inputs"]:
            if json.dumps(a, sort_keys=True) in s:
                n += 1
            else:
                break
        return n

    for vid in (["random"] if MODE == "random" else WANT):
        vm = None if MODE == "random" else models.BY_ID[vid]
        outp = OUTDIR / f"{vid}.json"
        done = {}
        if outp.exists():
            done = {f"{r['translator']}|{r['task_id']}": r
                    for r in json.loads(outp.read_text(encoding="utf-8"))}
            # A failed call is not a completed cell. Drop them so a resumed run retries them
            # instead of inheriting a provider outage as though it were data.
            failed = [k for k, r in done.items()
                      if r.get("status") == "call_failed"
                      or (r.get("status") == "unparseable" and not (r.get("raw") or "").strip())]
            for k in failed:
                del done[k]
            if failed:
                print(f"  retrying {len(failed)} cells whose provider call had failed",
                      flush=True)

        jobs = []
        for tid_model in ([vid] if SELF_ONLY else translators):
            tpath = bench.ROOT / "data" / "translations" / f"{tid_model}.json"
            if not tpath.exists():
                continue
            for t in json.loads(tpath.read_text(encoding="utf-8")):
                if not t.get("compile_ok"):
                    continue
                key = f"{tid_model}|{t['task_id']}"
                if key not in done:
                    jobs.append((tid_model, t))

        print(f"\n=== validator {vid} [{MODE}] : {len(done)} done, {len(jobs)} to go ===",
              flush=True)
        t0 = time.time()

        for i, (tmodel, t) in enumerate(jobs):
            tid = t["task_id"]
            p = problems[tid]
            ptypes = ", ".join(p["java_ptags"])

            if MODE == "random":
                # Non-LLM baseline: K type-directed inputs from the same generator that builds
                # the strong oracle, drawn AFTER the benchmark's own seed inputs so the baseline
                # gets no free help from the benchmark. This separates two explanations that
                # would otherwise be confounded: that K = 10 inputs is simply too few to catch
                # anything, and that the model chooses its 10 badly.
                n_seed = n_seed_prefix(p)
                inputs = p["inputs"][n_seed:n_seed + bench.K_PARITY]
            else:
                prompt = TEMPLATE.format(source=p["source"], java=t["java"],
                                         k=bench.K_PARITY, nargs=len(p["java_ptags"]),
                                         ptypes=ptypes)
                raw = models.call(vm, prompt, tag=f"validate:{MODE}:{vid}:{tmodel}:{tid}")
                # declared PYTHON parameter types disambiguate the JSON reading; see
                # models.extract_inputs for why that matters
                inputs = models.extract_inputs(raw, len(p["java_ptags"]), p["param_types"])

            rec = {"validator": vid, "translator": tmodel, "task_id": tid, "mode": MODE,
                   "self": vid == tmodel}
            if MODE != "random":
                # keep the raw response so a future change to the parser can be applied WITHOUT
                # regenerating: the last parser change cost a full re-run of every suite, purely
                # because only the parsed result had been stored
                rec["raw"] = raw[:4000]
            if MODE != "random" and not (raw or "").strip():
                # An empty response is a failed CALL, not an unparseable ANSWER. Conflating the
                # two once recorded a 160-call provider outage as though the model had answered
                # badly, and it removed an entire translator column for one validator.
                rec["status"] = "call_failed"
            elif not inputs:
                rec["status"] = "unparseable"
            else:
                inputs = inputs[:bench.K_PARITY]
                wd = WORK / tmodel / tid.replace("/", "_")
                if not (wd / "Solution.class").exists():
                    ok, _ = bench.compile_java(t["java"], wd)
                    if not ok:
                        rec["status"] = "no_compile"
                        done[f"{tmodel}|{tid}"] = rec
                        continue
                rec["status"] = "ok"
                rec["inputs"] = inputs
                rec.update(score_suite(p, wd, inputs))

            done[f"{tmodel}|{tid}"] = rec
            if (i + 1) % 20 == 0 or i + 1 == len(jobs):
                outp.write_text(json.dumps(list(done.values())), encoding="utf-8")
                nok = sum(1 for r in done.values() if r.get("status") == "ok")
                nc = sum(1 for r in done.values()
                         if r.get("status") == "ok" and r.get("full", {}).get("caught"))
                el = time.time() - t0
                rate = (i + 1) / max(el, 1e-9)
                print(f"  [{i + 1:4d}/{len(jobs)}] ok={nok} caught={nc} "
                      f"{el:.0f}s ({rate * 60:.1f}/min, eta {(len(jobs) - i - 1) / max(rate, 1e-9) / 60:.0f}m)",
                      flush=True)

        outp.write_text(json.dumps(list(done.values())), encoding="utf-8")
        print(f"=== {vid} [{MODE}] done ({time.time() - t0:.0f}s) ===", flush=True)


if __name__ == "__main__":
    main()
