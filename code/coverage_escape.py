"""Do escapes happen on paths the passing suite ALREADY executed?

The sharpest practitioner objection to this paper is that its validators get one shot, while real
migration validators (symbolic execution for COBOL; the Locksmith Loop's agentic search) drive
COVERAGE iteratively. If our escapes are simply unexplored paths, an iterative system fixes them
and the result does not transfer. If they sit on lines the passing suite already ran, then
coverage-driven search does not fix them, because the failure is boundary-VALUE selection inside a
covered path, and the result transfers to iterative systems too.

Method, per escaped translation:
  1. trace the LEGACY SOURCE on the inputs of a parity suite that PASSED, union the executed lines
  2. trace the same source on an input where the translation DIVERGED
  3. if the divergent input's lines are a subset of the suite's lines, the suite already covered
     that path and still missed: a boundary-value failure, not a coverage failure

Coverage is measured on the SOURCE, because that is the artefact a COBOL pipeline instruments and
the only one both sides share. sys.settrace is used rather than a coverage library so the result
has no dependency outside the standard library.

Usage:  python coverage_escape.py <python-exe> [--limit N]
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import bench      # noqa: E402
import models     # noqa: E402

WORKER = r'''
import json, sys
src = json.loads(sys.stdin.readline())
ep  = json.loads(sys.stdin.readline())
runs = json.loads(sys.stdin.readline())     # list of {"tag":..., "inputs":[[...], ...]}

ns = {}
code = compile(src, "<src>", "exec")
exec(code, ns)
fn = ns[ep]

lines = {}
cur = set()
def tracer(frame, event, arg):
    if frame.f_code.co_filename == "<src>":
        if event == "line":
            cur.add(frame.f_lineno)
        return tracer
    return None

out = {}
for r in runs:
    acc = set()
    for args in r["inputs"]:
        cur.clear()
        sys.settrace(tracer)
        try:
            fn(*args)
        except BaseException:
            pass
        finally:
            sys.settrace(None)
        acc |= set(cur)
    out[r["tag"]] = sorted(acc)
print("@@COV@@" + json.dumps(out))
'''


def trace(py: str, src: str, entry: str, runs: list[dict], timeout: float = 60.0):
    wp = HERE / "_worker_cov.py"
    wp.write_text(WORKER, encoding="utf-8")
    payload = (json.dumps(src) + "\n" + json.dumps(entry) + "\n" + json.dumps(runs) + "\n")
    try:
        r = subprocess.run([py, str(wp)], input=payload, capture_output=True, text=True,
                           timeout=timeout, encoding="utf-8", errors="replace")
    except subprocess.TimeoutExpired:
        return None
    for line in r.stdout.splitlines():
        if line.startswith("@@COV@@"):
            return json.loads(line[len("@@COV@@"):])
    return None


def main() -> None:
    py = sys.argv[1]
    limit = 0
    if "--limit" in sys.argv:
        limit = int(sys.argv[sys.argv.index("--limit") + 1])

    problems = {p["task_id"]: p for p in
                json.loads((bench.ROOT / "data" / "reference.json").read_text(encoding="utf-8"))}
    oracle = {}
    for m in models.MODELS:
        for r in json.loads((bench.ROOT / "data" / "oracle" / f"{m.mid}.json")
                            .read_text(encoding="utf-8")):
            oracle[(m.mid, r["task_id"])] = r

    # every ESCAPE: a plain parity suite that PASSED on a translation the oracle calls divergent
    escapes = []
    d = bench.ROOT / "data" / "parity"
    for f in sorted(d.glob("*.json")):
        for r in json.loads(f.read_text(encoding="utf-8")):
            if r.get("status") != "ok":
                continue
            full = r.get("full") or {}
            if full.get("degenerate") or full.get("caught"):
                continue
            o = oracle.get((r["translator"], r["task_id"]))
            if o and o.get("status") == "ok" and o.get("divergent") and o.get("diverge_idx"):
                escapes.append((r, o))
    if limit:
        escapes = escapes[:limit]
    print(f"escapes to analyse: {len(escapes)}")

    covered = uncovered = failed = 0
    for i, (r, o) in enumerate(escapes):
        p = problems[r["task_id"]]
        div_inputs = [p["inputs"][k] for k in o["diverge_idx"][:5] if k < len(p["inputs"])]
        if not div_inputs or not r.get("inputs"):
            failed += 1
            continue
        res = trace(py, p["source"], p["entry_point"],
                    [{"tag": "suite", "inputs": r["inputs"]},
                     {"tag": "div", "inputs": div_inputs}])
        if not res or not res.get("suite"):
            failed += 1
            continue
        s, dv = set(res["suite"]), set(res["div"])
        if dv and dv <= s:
            covered += 1
        else:
            uncovered += 1
        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(escapes)}] covered={covered} uncovered={uncovered} "
                  f"failed={failed}", flush=True)

    n = covered + uncovered
    out = {
        "n_escapes_analysed": n,
        "n_trace_failed": failed,
        "covered_path_escapes": covered,
        "uncovered_path_escapes": uncovered,
        "share_on_already_covered_paths": covered / n if n else float("nan"),
    }
    (bench.ROOT / "data" / "coverage_escape.json").write_text(json.dumps(out, indent=1),
                                                              encoding="utf-8")
    print()
    print(json.dumps(out, indent=1))
    if n:
        print(f"\n{100*covered/n:.1f}% of escapes occur on source lines the PASSING suite "
              f"had already executed.")


if __name__ == "__main__":
    main()
