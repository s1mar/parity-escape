"""Empirically infer each problem's concrete argument and return types.

Most HumanEval signatures carry no return annotation and spell containers as bare `list`, so
static parsing recovers only 26 of 164 problems. Instead we execute the benchmark's own test
block with the entry point replaced by a recording proxy. That yields, per problem: the concrete
Python types actually passed and returned, and the benchmark's own test inputs (which later serve
as the weak "benchmark-provided" oracle baseline).

Runs each problem in its own subprocess with a timeout, because a handful of HumanEval reference
solutions are slow and one hang would stall the whole corpus build.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def source_of(rec: dict) -> str:
    """The legacy source program: imports + helper defs + the entry def + its body."""
    return rec["declaration"] + rec["canonical_solution"]


# ---------------------------------------------------------------- type inference from values

def type_of(v, depth: int = 0) -> str | None:
    """Abstract type tag for one observed value, or None if unsupported."""
    if isinstance(v, bool):
        return "bool"
    if isinstance(v, int):
        return "int"
    if isinstance(v, float):
        return "float"
    if isinstance(v, str):
        return "str"
    if isinstance(v, (list, tuple)) and depth < 2:
        if not v:
            return "list[?]"
        inner = [type_of(x, depth + 1) for x in v]
        if any(i is None for i in inner):
            return None
        return "list[" + join_types(inner) + "]"
    return None


def join_types(ts: list[str]) -> str:
    """Least upper bound over observed element types. `?` is the unknown from an empty list."""
    ts = [t for t in ts if t != "?"]
    if not ts:
        return "?"
    u = set()
    for t in ts:
        u.add(t)
    if len(u) == 1:
        return next(iter(u))
    # widen int to float; bool widens to int then float
    if u <= {"bool", "int"}:
        return "int"
    if u <= {"bool", "int", "float"}:
        return "float"
    # unify list element types recursively
    if all(t.startswith("list[") for t in u):
        inners = [t[5:-1] for t in u]
        return "list[" + join_types(inners) + "]"
    if "?" in u:
        u.discard("?")
        return join_types(sorted(u))
    return "MIXED"


WORKER = r'''
import json, sys, os
os.environ["HF_DATASETS_OFFLINE"] = "1"
sys.setrecursionlimit(10000)
idx = int(sys.argv[1])
from datasets import load_dataset
d = load_dataset("bigcode/humanevalpack", "python", split="test")
rec = d[idx]
src = rec["declaration"] + rec["canonical_solution"]
ep = rec["entry_point"]

ns = {}
exec(src, ns)
real = ns[ep]
calls = []

def proxy(*args, **kw):
    out = real(*args, **kw)
    if not kw:
        try:
            json.dumps([args, out])
            calls.append([list(args), out])
        except (TypeError, ValueError):
            pass
    return out

ns[ep] = proxy
try:
    exec(rec["test"], ns)
    status = "ok"
except BaseException as e:
    status = "test_error:" + type(e).__name__
print("@@RESULT@@" + json.dumps({"status": status, "calls": calls[:400]}))
'''


def probe_one(py: str, idx: int, timeout: int = 60) -> dict:
    wp = HERE / "_worker_probe.py"
    if not wp.exists():
        wp.write_text(WORKER, encoding="utf-8")
    try:
        r = subprocess.run(
            [py, str(wp), str(idx)],
            capture_output=True, text=True, timeout=timeout,
            cwd=str(HERE),
        )
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "calls": []}
    for line in r.stdout.splitlines():
        if line.startswith("@@RESULT@@"):
            return json.loads(line[len("@@RESULT@@"):])
    return {"status": "no_result", "calls": [], "stderr": r.stderr[-400:]}


def main() -> None:
    py = sys.argv[1] if len(sys.argv) > 1 else sys.executable
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    from datasets import load_dataset

    d = load_dataset("bigcode/humanevalpack", "python", split="test")
    out = []
    for i, rec in enumerate(d):
        res = probe_one(py, i)
        entry = {
            "idx": i,
            "task_id": rec["task_id"],
            "entry_point": rec["entry_point"],
            "signature": rec["signature"],
            "status": res["status"],
            "n_calls": len(res.get("calls", [])),
        }
        calls = res.get("calls", [])
        if calls:
            arity = {len(c[0]) for c in calls}
            if len(arity) == 1:
                n = next(iter(arity))
                ptypes = []
                for k in range(n):
                    ts = [type_of(c[0][k]) for c in calls]
                    ptypes.append(None if any(t is None for t in ts) else join_types(ts))
                rts = [type_of(c[1]) for c in calls]
                rtype = None if any(t is None for t in rts) else join_types(rts)
                entry["param_types"] = ptypes
                entry["return_type"] = rtype
                entry["seed_inputs"] = [c[0] for c in calls[:64]]
            else:
                entry["param_types"] = None
                entry["return_type"] = None
                entry["note"] = f"variable arity {sorted(arity)}"
        out.append(entry)
        print(f"[{i:3d}/164] {rec['task_id']:12s} {res['status']:16s} calls={len(calls)}",
              flush=True)

    (ROOT / "data" / "probe.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("wrote", ROOT / "data" / "probe.json")


if __name__ == "__main__":
    main()
