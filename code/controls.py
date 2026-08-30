"""Controls C2 and C3: measure the strong oracle's own detection power.

The whole paper turns on the strong oracle being able to see divergences that a small generated
parity suite cannot. An oracle that says "equivalent" to everything would produce an escape rate
of zero and look like a clean null result, so its power is measured rather than assumed.

C2 (mutation adequacy, source side): seeded single-token mutants of the legacy Python source are
run against the source itself over the strong oracle's input set. Reports the kill rate, and for
contrast the kill rate of the benchmark's own test inputs on the same mutants. Mutants that no
input set kills are either semantically equivalent or unreachable, and are reported as such
rather than being quietly dropped.

C3 (injection, target side): a known divergence is injected into Java translations that passed
both the parity suite and the strong oracle. Every injection the oracle fails to catch is a
false negative in the instrument.

Usage:  python controls.py <python-exe> [--c2] [--c3] [--limit N]
"""
from __future__ import annotations

import ast
import json
import random
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench      # noqa: E402
import models     # noqa: E402

WORK = bench.HERE / "_ctl"


# ------------------------------------------------------------------------------ C2: Python mutants

_CMP_SWAP = {
    ast.Lt: ast.LtE, ast.LtE: ast.Lt, ast.Gt: ast.GtE, ast.GtE: ast.Gt,
    ast.Eq: ast.NotEq, ast.NotEq: ast.Eq,
}
_BIN_SWAP = {
    ast.Add: ast.Sub, ast.Sub: ast.Add, ast.Mult: ast.FloorDiv,
    ast.FloorDiv: ast.Mult, ast.Mod: ast.FloorDiv,
}


class _Mutator(ast.NodeTransformer):
    """Applies exactly the n-th eligible mutation and leaves everything else alone."""

    def __init__(self, target: int):
        self.target = target
        self.seen = 0
        self.applied: str | None = None

    def _hit(self) -> bool:
        hit = self.seen == self.target
        self.seen += 1
        return hit

    def visit_Compare(self, node):                      # noqa: N802
        self.generic_visit(node)
        if len(node.ops) == 1 and type(node.ops[0]) in _CMP_SWAP:
            if self._hit():
                old = type(node.ops[0]).__name__
                node.ops[0] = _CMP_SWAP[type(node.ops[0])]()
                self.applied = f"cmp:{old}->{type(node.ops[0]).__name__}"
        return node

    def visit_BinOp(self, node):                        # noqa: N802
        self.generic_visit(node)
        if type(node.op) in _BIN_SWAP:
            if self._hit():
                old = type(node.op).__name__
                node.op = _BIN_SWAP[type(node.op)]()
                self.applied = f"bin:{old}->{type(node.op).__name__}"
        return node

    def visit_Constant(self, node):                     # noqa: N802
        if isinstance(node.value, int) and not isinstance(node.value, bool):
            if self._hit():
                self.applied = f"const:{node.value}->{node.value + 1}"
                return ast.copy_location(ast.Constant(value=node.value + 1), node)
        return node


def count_sites(src: str) -> int:
    m = _Mutator(-1)
    m.visit(ast.parse(src))
    return m.seen


def make_mutant(src: str, k: int) -> tuple[str, str] | None:
    tree = ast.parse(src)
    m = _Mutator(k)
    new = m.visit(tree)
    if m.applied is None:
        return None
    ast.fix_missing_locations(new)
    try:
        return ast.unparse(new), m.applied
    except Exception:                                   # noqa: BLE001
        return None


def run_c2(py: str, limit: int, per_problem: int = 4, seed: int = 0) -> dict:
    problems = json.loads((bench.ROOT / "data" / "reference.json").read_text(encoding="utf-8"))
    eligible = json.loads((bench.ROOT / "data" / "eligible.json").read_text(encoding="utf-8"))
    bench_n_by_task = bench.benchmark_seed_prefix_counts(problems, eligible)
    rng = random.Random(seed)
    rows = []
    t0 = time.time()
    sel = problems[:limit] if limit else problems

    for i, p in enumerate(sel):
        n = count_sites(p["source"])
        if n == 0:
            continue
        picks = rng.sample(range(n), min(per_problem, n))
        base = p["source_out"]
        strong_inputs = p["inputs"]
        # The benchmark's own inputs ARE the leading prefix of the strong set by construction
        # (bench.fuzz_inputs prepends them), but they are not a fixed-length prefix: this used to
        # be min(64, len(strong_inputs)), which overcounts for the 141 of 147 problems with fewer
        # than 64 real seed inputs. bench.benchmark_seed_prefix_counts is the true count, shared
        # with analyze.py's identical computation so the two cannot state two different numbers
        # for the same quantity again.
        bench_n = bench_n_by_task.get(p["task_id"], 0)

        for k in picks:
            mk = make_mutant(p["source"], k)
            if mk is None:
                continue
            msrc, desc = mk
            # Progressive chunks with early exit. A mutant is killed by its FIRST divergent
            # input, so running all 1000 is wasted work for the ones that die immediately, and
            # the ones that do not die are frequently mutants that broke a loop guard and now
            # hang on every input: those would each burn a full execution budget. Chunking pays
            # the budget only for as long as the mutant is still alive.
            killed_at = None
            start = 0
            for size in (25, 75, 200, 700):
                if killed_at is not None or start >= len(strong_inputs):
                    break
                chunk = strong_inputs[start:start + size]
                mo = bench.run_python(py, msrc, p["entry_point"], chunk, budget=25.0)
                for j, b in enumerate(mo):
                    # same domain rule as the main analysis: an input on which the SOURCE steps
                    # outside its declared return type is out of domain here too, so the
                    # mutation score is measured over exactly the inputs the study can use
                    if bench.classify(base[start + j], b, p["return_type"]) == "DIVERGE":
                        killed_at = start + j
                        break
                start += size
            rows.append({
                "task_id": p["task_id"], "site": k, "mutation": desc,
                "killed": killed_at is not None,
                "killed_at": killed_at,
                "killed_by_benchmark_inputs": killed_at is not None and killed_at < bench_n,
                "bench_n": bench_n,
            })
        if (i + 1) % 10 == 0:
            print(f"  C2 [{i + 1}/{len(sel)}] mutants={len(rows)} ({time.time() - t0:.0f}s)",
                  flush=True)

    killed = sum(1 for r in rows if r["killed"])
    bench_killed = sum(1 for r in rows if r["killed_by_benchmark_inputs"])
    out = {
        "n_mutants": len(rows),
        "n_killed": killed,
        "kill_rate": killed / max(1, len(rows)),
        "n_killed_by_benchmark_inputs": bench_killed,
        "benchmark_kill_rate": bench_killed / max(1, len(rows)),
        "rows": rows,
    }
    (bench.ROOT / "data" / "control_c2.json").write_text(json.dumps(out), encoding="utf-8")
    print(f"\nC2 mutation adequacy: strong oracle killed {killed}/{len(rows)} "
          f"({100 * out['kill_rate']:.1f}%); benchmark's own inputs killed {bench_killed} "
          f"({100 * out['benchmark_kill_rate']:.1f}%)")
    return out


def reprocess_c2_bench_n() -> dict:
    """Recompute bench_n and the benchmark kill rate from the STORED C2 rows.

    Same pattern as reparse.py for parity data: a downstream definition changed, and every
    already-collected record has everything needed to re-score under the corrected one without
    repeating the expensive part. Mutation execution (killed_at) does not depend on bench_n, so
    nothing here re-runs a single mutant; only the boundary that turns killed_at into "caught by
    the benchmark's own inputs" changes.
    """
    p = bench.ROOT / "data" / "control_c2.json"
    out = json.loads(p.read_text(encoding="utf-8"))
    problems = json.loads((bench.ROOT / "data" / "reference.json").read_text(encoding="utf-8"))
    eligible = json.loads((bench.ROOT / "data" / "eligible.json").read_text(encoding="utf-8"))
    true_n = bench.benchmark_seed_prefix_counts(problems, eligible)

    for r in out["rows"]:
        n = true_n.get(r["task_id"], 0)
        r["bench_n"] = n
        r["killed_by_benchmark_inputs"] = r["killed_at"] is not None and r["killed_at"] < n

    bench_killed = sum(1 for r in out["rows"] if r["killed_by_benchmark_inputs"])
    out["n_killed_by_benchmark_inputs"] = bench_killed
    out["benchmark_kill_rate"] = bench_killed / max(1, len(out["rows"]))
    p.write_text(json.dumps(out), encoding="utf-8")
    print(f"C2 reprocessed: benchmark's own inputs killed {bench_killed}/{len(out['rows'])} "
          f"({100 * out['benchmark_kill_rate']:.1f}%) under the true per-problem seed count")
    return out


# ------------------------------------------------------------------------------ C3: Java injection

INJECTIONS = [
    (r"(?<![<>=!])>=(?!=)", ">", "relational >= to >"),
    (r"(?<![<>=!])<=(?!=)", "<", "relational <= to <"),
    (r"(?<![<>=!+\-*/])\+ 1\b", "+ 2", "off-by-one +1 to +2"),
    (r"\bi\+\+", "i+=2", "loop stride 1 to 2"),
    (r"(?<![<>=!])==(?!=)", "!=", "equality == to !="),
]


def inject(java: str, rng: random.Random) -> tuple[str, str] | None:
    order = list(range(len(INJECTIONS)))
    rng.shuffle(order)
    for oi in order:
        pat, rep, desc = INJECTIONS[oi]
        hits = list(re.finditer(pat, java))
        if not hits:
            continue
        h = rng.choice(hits)
        return java[:h.start()] + rep + java[h.end():], desc
    return None


def run_c3(py: str, n_target: int = 20, seed: int = 0) -> dict:
    problems = {p["task_id"]: p for p in
                json.loads((bench.ROOT / "data" / "reference.json").read_text(encoding="utf-8"))}
    rng = random.Random(seed)

    clean = []
    for m in models.MODELS:
        op = bench.ROOT / "data" / "oracle" / f"{m.mid}.json"
        tp = bench.ROOT / "data" / "translations" / f"{m.mid}.json"
        if not (op.exists() and tp.exists()):
            continue
        tj = {t["task_id"]: t for t in json.loads(tp.read_text(encoding="utf-8"))}
        for r in json.loads(op.read_text(encoding="utf-8")):
            if r.get("status") == "ok" and r.get("divergent") is False:
                clean.append((m.mid, r["task_id"], tj[r["task_id"]]["java"]))
    rng.shuffle(clean)
    print(f"C3: {len(clean)} translations agree with the source on every comparable input")

    rows = []
    for mid, tid, java in clean:
        if len(rows) >= n_target:
            break
        inj = inject(java, rng)
        if inj is None:
            continue
        jsrc, desc = inj
        p = problems[tid]
        wd = WORK / "c3" / f"{mid}_{tid.replace('/', '_')}"
        ok, _ = bench.compile_java(jsrc, wd)
        if not ok:
            continue
        jr = bench.run_java(wd, p["entry_point"], p["java_ptags"], p["inputs"])
        verdicts = [bench.classify(a, b, p["return_type"])
                    for a, b in zip(p["source_out"], jr)]
        caught = any(v == "DIVERGE" for v in verdicts)
        rows.append({"model": mid, "task_id": tid, "injection": desc, "caught": caught,
                     "n_diverge": sum(1 for v in verdicts if v == "DIVERGE")})
        print(f"  C3 {len(rows):2d}/{n_target} {mid:8s} {tid:12s} {desc:22s} "
              f"{'CAUGHT' if caught else 'MISSED'}", flush=True)

    caught = sum(1 for r in rows if r["caught"])
    out = {"n": len(rows), "n_caught": caught,
           "catch_rate": caught / max(1, len(rows)), "rows": rows}
    (bench.ROOT / "data" / "control_c3.json").write_text(json.dumps(out), encoding="utf-8")
    print(f"\nC3 injection: strong oracle caught {caught}/{len(rows)}")
    return out


def main() -> None:
    argv = sys.argv[1:]
    py = argv.pop(0)
    limit = 0
    if "--limit" in argv:
        i = argv.index("--limit")
        limit = int(argv[i + 1])
        del argv[i:i + 2]
    if "--reprocess-c2" in argv:
        reprocess_c2_bench_n()
        return
    if "--c2" in argv or not argv:
        run_c2(py, limit)
    if "--c3" in argv or not argv:
        run_c3(py)


if __name__ == "__main__":
    main()
