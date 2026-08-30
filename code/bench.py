"""Corpus, fuzzer, dual execution and divergence comparison.

Everything here implements SPEC.md literally. Where SPEC.md names a constant, it is defined once
at the top of this file with the same name, so `crosscheck.py` can diff the two mechanically.
"""
from __future__ import annotations

import json
import random
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import winlimit  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
JAVA_DIR = HERE / "java"

# ---------------------------------------------------------------- constants declared in SPEC.md
N_FUZZ = 1000              # type-directed inputs per problem
FUZZ_SEED = 0
BOUNDARY_FRACTION = 0.30   # remainder is random
K_PARITY = 10              # test inputs requested from a validator
K_SENSITIVITY = (3, 5, 10)
FLOAT_RTOL = 1e-6
PER_CALL_MS = 2000         # per-input timeout, both sides
MAX_TIMEOUTS = 25          # abort a program after this many timed-out inputs
MIN_INDOMAIN = 100         # a problem needs this many in-domain fuzz inputs to be eligible
MIN_PARITY_INDOMAIN = 3    # fewer than this makes a generated parity suite DEGENERATE
MEM_CAP_BYTES = 2 * 1024 ** 3    # BEST-EFFORT per-child memory cap. Measured NOT to take effect
                                 # in this sandbox (SPEC Amendment 2); nothing relies on it.
JVM_HEAP_MB = 1024
MAX_RESULT_CHARS = 1_000_000     # a larger serialised result is treated as resource exhaustion
EXEC_BUDGET_S = 120              # wall-clock budget for executing ONE program over one input set
JAVA_LONG_MAX = 2 ** 63 - 1      # the declared Java integer type; see conforms() (Amendment 10)
MUTANT_KILL_FLOOR = 0.90   # C2: strong oracle must kill at least this share of live mutants
CONSERVATIVE_INT_CAP = 10 ** 6   # robustness split: escape rate restricted to |int| <= this

# Two integer boundary catalogues, split by POSITION (Amendment 2).
#
# A scalar integer parameter is frequently used as an allocation size or a loop bound, so an
# input of 10**18 asks the source program for a structure of that many elements: the first two
# reference runs died at 41 GB and 21 GB on `string_sequence(n)`. The same value appearing as a
# LIST ELEMENT cannot drive allocation, because the generator bounds list length independently.
#
# Overflow probing is not lost by this split, and is arguably improved: Java `long` only wraps
# near 2**63, and a scalar argument of that size makes the source time out long before it
# computes anything, whereas a list of large elements summed or multiplied inside the function
# reaches the wrap point while both programs still terminate.
SCALAR_INT_BOUNDARIES = [0, 1, -1, 2, -2, 3, -3, 10, -10, 100, -100, 1000, -1000,
                         10 ** 4, -(10 ** 4)]
ELEM_INT_BOUNDARIES = SCALAR_INT_BOUNDARIES + [
    2 ** 31 - 1, -(2 ** 31), 2 ** 31, 2 ** 62, 2 ** 63 - 1, -(2 ** 63), 10 ** 18]
FLOAT_BOUNDARIES = [0.0, -0.0, 1.0, -1.0, 0.5, -0.5, 1e-9, -1e-9, 1e18, -1e18,
                    2.0, 3.0, 0.1, 1e9]
STR_BOUNDARIES = ["", " ", "a", "A", "0", "ab", "aa", "Aa", "  ", "\t", "abc123",
                  "The quick brown fox", "aAbBcC", "x" * 200, "éüñ", "()", "[]"]

# Python abstract tag -> Java type tag
J = {
    "int": "long", "float": "double", "bool": "boolean", "str": "String",
    "list[int]": "long[]", "list[float]": "double[]", "list[bool]": "boolean[]",
    "list[str]": "String[]", "list[list[int]]": "long[][]",
    "list[list[float]]": "double[][]", "list[list[str]]": "String[][]",
}
# Java type tag -> a readable Java type for the prompt (identical here; kept explicit)
JAVA_TYPE = dict((v, v) for v in J.values())


@dataclass
class Problem:
    task_id: str
    idx: int
    entry_point: str
    param_types: list[str]
    return_type: str
    source: str                      # the legacy Python program
    seed_inputs: list = field(default_factory=list)

    @property
    def java_ptags(self) -> list[str]:
        return [J[t] for t in self.param_types]

    @property
    def java_rtag(self) -> str:
        return J[self.return_type]

    @property
    def java_decl(self) -> str:
        args = ", ".join(f"{J[t]} a{i}" for i, t in enumerate(self.param_types))
        return f"public static {J[self.return_type]} {self.entry_point}({args})"

    @property
    def slug(self) -> str:
        return self.task_id.replace("/", "_")


def load_problems() -> list[Problem]:
    """Eligible problems, with the legacy source text attached."""
    import os
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    from datasets import load_dataset

    d = load_dataset("bigcode/humanevalpack", "python", split="test")
    elig = json.loads((ROOT / "data" / "eligible.json").read_text(encoding="utf-8"))
    out = []
    for e in elig:
        rec = d[e["idx"]]
        out.append(Problem(
            task_id=e["task_id"], idx=e["idx"], entry_point=e["entry_point"],
            param_types=list(e["param_types"]), return_type=e["return_type"],
            source=rec["declaration"] + rec["canonical_solution"],
            seed_inputs=e.get("seed_inputs", []),
        ))
    return out


# ---------------------------------------------------------------------------------- the fuzzer

def _rand_int(rng: random.Random, depth: int = 0) -> int:
    """Random integer. Scalar argument positions (depth 0) stay small; list elements may be large.

    A scalar integer argument is very often an allocation size or a loop bound, so drawing 20% of
    them from +/-10^6 made the reference run spend most of its time building multi-megabyte
    strings that were then discarded as oversize. It bought nothing: a Java `long` cannot overflow
    from the magnitude of a scalar argument alone. List elements keep the wide range, because
    there the magnitude feeds the function's own arithmetic, which is where a wrap is observable.
    """
    r = rng.random()
    if depth == 0:
        if r < 0.45:
            return rng.randint(-10, 10)
        if r < 0.85:
            return rng.randint(-1000, 1000)
        return rng.randint(-10 ** 4, 10 ** 4)
    if r < 0.45:
        return rng.randint(-10, 10)
    if r < 0.75:
        return rng.randint(-1000, 1000)
    return rng.randint(-CONSERVATIVE_INT_CAP, CONSERVATIVE_INT_CAP)


def _rand_float(rng: random.Random) -> float:
    r = rng.random()
    if r < 0.5:
        return round(rng.uniform(-10, 10), 4)
    if r < 0.8:
        return round(rng.uniform(-1000, 1000), 4)
    return rng.uniform(-1e6, 1e6)


def _rand_str(rng: random.Random) -> str:
    alpha = rng.choice(["abc", "abcABC", "abc xyz", "a1b2c3", "()[]{}", "aeiou",
                        "abcdefghijklmnopqrstuvwxyz "])
    n = rng.choice([0, 1, 2, 3, 5, 8, 13, 25])
    return "".join(rng.choice(alpha) for _ in range(n))


def gen_value(tag: str, rng: random.Random, boundary: bool, depth: int = 0):
    """Generate one value. `depth` 0 means a top-level scalar argument position, where large
    integer magnitudes can drive an allocation; depth > 0 means inside a list, where they cannot."""
    if tag == "int":
        cat = ELEM_INT_BOUNDARIES if depth > 0 else SCALAR_INT_BOUNDARIES
        return rng.choice(cat) if boundary else _rand_int(rng, depth)
    if tag == "float":
        return rng.choice(FLOAT_BOUNDARIES) if boundary else _rand_float(rng)
    if tag == "bool":
        return rng.choice([True, False])
    if tag == "str":
        return rng.choice(STR_BOUNDARIES) if boundary else _rand_str(rng)
    if tag.startswith("list["):
        inner = tag[5:-1]
        if boundary:
            n = rng.choice([0, 0, 1, 1, 2, 100])
        else:
            n = rng.choice([0, 1, 2, 3, 4, 5, 8, 12, 20])
        vals = [gen_value(inner, rng, boundary, depth + 1) for _ in range(n)]
        if boundary and n > 1 and rng.random() < 0.4:
            shape = rng.random()
            if shape < 0.34:
                vals = [vals[0]] * n                       # all duplicates
            elif shape < 0.67:
                try:
                    vals = sorted(vals)                    # already sorted
                except TypeError:
                    pass
            else:
                try:
                    vals = sorted(vals, reverse=True)      # reverse sorted
                except TypeError:
                    pass
        return vals
    raise ValueError(f"no generator for {tag}")


def benchmark_seed_prefix_counts(reference_records: list[dict],
                                 eligible_records: list[dict]) -> dict[str, int]:
    """How many of each problem's LEADING fuzz inputs are the benchmark's own seed inputs.

    fuzz_inputs() below prepends a problem's deduplicated seed_inputs before generating the rest,
    so they are always a prefix of the stored "inputs" list, but a SHORT one (median 7 in this
    corpus, max 64) and one that varies per problem. controls.py once counted this as a fixed
    min(64, len(inputs)) instead of the true prefix length, which is a different quantity: for
    141 of 147 problems that overcounts, because most have fewer than 64 real seed inputs, and it
    silently changed what "the benchmark's own inputs" measured. This is the single definition;
    analyze.py and controls.py both call it, so the two can no longer drift apart.
    """
    seed_sets = {e["task_id"]: {json.dumps(s, sort_keys=True)
                                for s in (e.get("seed_inputs") or [])}
                 for e in eligible_records}
    out = {}
    for p in reference_records:
        s = seed_sets.get(p["task_id"], set())
        n = 0
        for a in p["inputs"]:
            if json.dumps(a, sort_keys=True) in s:
                n += 1
            else:
                break
        out[p["task_id"]] = n
    return out


def fuzz_inputs(p: Problem, n: int = N_FUZZ, seed: int = FUZZ_SEED) -> list[list]:
    """Type-directed inputs: BOUNDARY_FRACTION drawn from the boundary catalogue, rest random.

    The problem's own benchmark inputs are prepended, so the strong oracle is a strict superset
    of the benchmark-provided oracle and can never be weaker than it.
    """
    rng = random.Random(seed)
    out: list[list] = []
    seen: set[str] = set()
    for si in p.seed_inputs:
        k = json.dumps(si, sort_keys=True)
        if k not in seen:
            seen.add(k)
            out.append(si)
    guard = 0
    while len(out) < n and guard < n * 40:
        guard += 1
        boundary = rng.random() < BOUNDARY_FRACTION
        args = [gen_value(t, rng, boundary) for t in p.param_types]
        k = json.dumps(args, sort_keys=True)
        if k in seen:
            continue
        seen.add(k)
        out.append(args)
    return out


# ------------------------------------------------------------------------------ Python execution

PY_WORKER = r'''
import json, os, sys, threading
sys.setrecursionlimit(20000)
MAX_RESULT_CHARS = 1_000_000

# Raw DAEMON threads, not a ThreadPoolExecutor. A timed-out call cannot be killed in Python, so
# the worker has to be able to exit while it is still running. ThreadPoolExecutor threads are
# non-daemon and are joined by an interpreter atexit hook, so the first non-terminating input
# would hang the worker until the outer subprocess timeout fired: 600 seconds per problem
# instead of 2. Every exit below is os._exit, which skips atexit entirely.
def call_with_timeout(fn, args, secs):
    box = {}
    def run():
        try:
            box["v"] = fn(*args)
        except BaseException as ex:
            box["e"] = type(ex).__name__
    t = threading.Thread(target=run, daemon=True)
    t.start()
    t.join(secs)
    if t.is_alive():
        return "timeout", None
    if "e" in box:
        return "error", box["e"]
    return "ok", box.get("v")
src = sys.stdin.readline()
src = json.loads(src)
ep = json.loads(sys.stdin.readline())
per_ms = json.loads(sys.stdin.readline())
max_to = json.loads(sys.stdin.readline())
ns = {}
exec(src, ns)
fn = ns[ep]
timeouts = 0
for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    try:
        args = json.loads(line)
    except Exception:
        print(json.dumps({"ok": False, "e": "BadInput"}), flush=True); continue

    kind, payload = call_with_timeout(fn, args, per_ms / 1000.0)
    if kind == "timeout":
        timeouts += 1
        print(json.dumps({"ok": False, "e": "Timeout"}), flush=True)
        if timeouts >= max_to:
            # exit reporting ONLY the timeout, so the driver sees a trailing Timeout line and
            # knows exactly which input to resume after
            sys.stdout.flush()
            os._exit(3)
        continue
    if kind == "error":
        print(json.dumps({"ok": False, "e": payload}), flush=True)
        continue
    try:
        enc = json.dumps({"ok": True, "v": payload})
    except (TypeError, ValueError):
        enc = json.dumps({"ok": False, "e": "Unserializable"})
    except (OverflowError, MemoryError):
        enc = json.dumps({"ok": False, "e": "Oversize"})
    if len(enc) > MAX_RESULT_CHARS:
        enc = json.dumps({"ok": False, "e": "Oversize"})
    print(enc, flush=True)
sys.stdout.flush()
os._exit(0)
'''


def _parse_lines(stdout: str) -> list[dict]:
    out = []
    for l in stdout.splitlines():
        if not l.strip():
            continue
        try:
            out.append(json.loads(l))
        except Exception:                               # noqa: BLE001
            out.append({"ok": False, "e": "BadOutput"})
    return out


def _run_with_restarts(spawn, inputs: list[list], timeout: float,
                       budget: float = EXEC_BUDGET_S) -> list[dict]:
    """Execute `inputs`, restarting the worker process after every timed-out call.

    A timed-out call cannot be killed inside the worker: Python cannot kill a thread and Java
    cannot kill a Future's thread either. Left running, those threads keep consuming CPU and
    contend for the interpreter lock, so every SUBSEQUENT call slows down. That is what turned
    one non-terminating problem into a stall of many minutes. Killing the whole worker after a
    timeout and resuming from the next input keeps the cost of a hang at one timeout plus one
    process start, and keeps it local to the input that caused it.

    The total number of timeouts is capped at MAX_TIMEOUTS across the whole call; beyond that the
    remaining inputs are marked TooManyTimeouts, which is out of domain, so a program that hangs
    on a large share of its inputs simply fails the in-domain threshold and is dropped.
    """
    results: list[dict] = []
    timeouts = 0
    t0 = time.time()
    while len(results) < len(inputs):
        remaining = inputs[len(results):]
        if timeouts >= MAX_TIMEOUTS:
            results.extend({"ok": False, "e": "TooManyTimeouts"} for _ in remaining)
            break
        left = budget - (time.time() - t0)
        if left <= 0:
            # A per-call timeout cannot catch a program that is merely SLOW: one HumanEval
            # reference runs about 10^6 inner iterations per call, so every call finishes just
            # inside the 2 s limit and 1000 of them take half an hour. The wall-clock budget is
            # what bounds that case. Unspent inputs are out of domain, so a truncated problem
            # simply has a smaller comparable set, or falls below the in-domain floor and is
            # dropped.
            results.extend({"ok": False, "e": "Budget"} for _ in remaining)
            break
        stdout = spawn(remaining, min(timeout, left))
        got = _parse_lines(stdout)
        if not got:
            results.extend({"ok": False, "e": "NoResult"} for _ in remaining)
            break
        results.extend(got[:len(remaining)])
        if got and got[-1].get("e") == "Timeout" and len(results) < len(inputs):
            timeouts += 1
            continue
        if len(got) < len(remaining):
            # the worker died without reporting a timeout; do not spin on the same input
            results.append({"ok": False, "e": "NoResult"})
    while len(results) < len(inputs):
        results.append({"ok": False, "e": "NoResult"})
    return results[:len(inputs)]


def run_python(py: str, source: str, entry: str, inputs: list[list],
               timeout: float = 600.0, budget: float = EXEC_BUDGET_S) -> list[dict]:
    wp = HERE / "_worker_py.py"
    if not wp.exists():
        wp.write_text(PY_WORKER, encoding="utf-8")

    def spawn(chunk: list[list], tmo: float) -> str:
        payload = (json.dumps(source) + "\n" + json.dumps(entry) + "\n"
                   + json.dumps(PER_CALL_MS) + "\n" + json.dumps(1) + "\n"
                   + "\n".join(json.dumps(a) for a in chunk) + "\n")
        stdout, _e, _rc = winlimit.run_capped([py, str(wp)], payload,
                                              timeout=tmo, mem_bytes=MEM_CAP_BYTES)
        return stdout

    return _run_with_restarts(spawn, inputs, timeout, budget)


# -------------------------------------------------------------------------------- Java execution

def compile_java(solution_src: str, workdir: Path) -> tuple[bool, str]:
    workdir.mkdir(parents=True, exist_ok=True)
    (workdir / "Solution.java").write_text(solution_src, encoding="utf-8")
    harness = (JAVA_DIR / "Harness.java").read_text(encoding="utf-8")
    (workdir / "Harness.java").write_text(harness, encoding="utf-8")
    r = subprocess.run(["javac", "-nowarn", "-d", str(workdir),
                        str(workdir / "Solution.java"), str(workdir / "Harness.java")],
                       capture_output=True, text=True, timeout=180)
    return r.returncode == 0, (r.stderr or "")[-1500:]


def run_java(workdir: Path, method: str, ptags: list[str], inputs: list[list],
             timeout: float = 600.0, budget: float = EXEC_BUDGET_S) -> list[dict]:
    def spawn(chunk: list[list], tmo: float) -> str:
        payload = "\n".join(json.dumps(a) for a in chunk) + "\n"
        cmd = ["java", "-Xss64m", f"-Xmx{JVM_HEAP_MB}m", "-cp", str(workdir), "Harness",
               method, ",".join(ptags), str(PER_CALL_MS), "1"]
        stdout, _e, _rc = winlimit.run_capped(cmd, payload, timeout=tmo,
                                              mem_bytes=MEM_CAP_BYTES)
        return stdout

    return _run_with_restarts(spawn, inputs, timeout, budget)


# ------------------------------------------------------------------------------- comparison rules

def _num_eq(a, b) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return bool(a) == bool(b)
    # Integers are compared exactly and BEFORE any float conversion. Python integers are
    # arbitrary precision, so a source program can legitimately return a value that float()
    # cannot represent; converting first raised OverflowError on the reference run.
    if isinstance(a, int) and isinstance(b, int):
        return a == b
    try:
        fa, fb = float(a), float(b)
    except (OverflowError, ValueError):
        # one side does not fit in a float, so the two cannot be equal within a relative
        # tolerance unless they were both integers, which the branch above already handled
        return False
    if fa != fa and fb != fb:          # both NaN
        return True
    if fa in (float("inf"), float("-inf")) or fb in (float("inf"), float("-inf")):
        return fa == fb
    return abs(fa - fb) <= FLOAT_RTOL * max(1.0, abs(fa), abs(fb))


def values_equal(a, b) -> bool:
    """SPEC.md comparison rules: integers exact, floats to a relative tolerance, strings exact,
    lists elementwise in order."""
    if isinstance(a, str) or isinstance(b, str):
        return isinstance(a, str) and isinstance(b, str) and a == b
    if isinstance(a, list) or isinstance(b, list):
        if not (isinstance(a, list) and isinstance(b, list)) or len(a) != len(b):
            return False
        return all(values_equal(x, y) for x, y in zip(a, b))
    if a is None or b is None:
        return a is None and b is None
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return _num_eq(a, b)
    return a == b


# An input is informative about equivalence only if the source produced a value and neither side
# hit a RESOURCE limit. Resource exhaustion is not semantic divergence: a Java OutOfMemoryError
# against a Python run that happened to fit says something about heap settings, not about whether
# the migration preserved behaviour. Every name here is excluded on BOTH sides, and excluding
# them can only lower the measured escape rate, so the headline number stays a lower bound.
IN_DOMAIN_EXCLUDED = {
    "Timeout", "TooManyTimeouts", "NoResult", "BadOutput", "Unserializable",
    "Oversize", "MemoryError", "OutOfMemoryError", "StackOverflowError",
    "RecursionError", "MemoryAbort", "NegativeArraySizeException", "Budget", "NotRun",
}


# An input is ORDINARY if nothing about it is extreme: no large integer, no long or non-ASCII
# string, no long list. The strongest objection to this study is that its divergences live only
# on inputs no real caller would produce, so the escape rate is also reported restricted to
# divergences that have at least one ordinary witness. The thresholds are deliberately strict.
ORDINARY_INT = 1000
ORDINARY_STR = 25
ORDINARY_LIST = 20


def is_ordinary(v) -> bool:
    if isinstance(v, bool):
        return True
    if isinstance(v, int):
        return abs(v) <= ORDINARY_INT
    if isinstance(v, float):
        return abs(v) <= ORDINARY_INT
    if isinstance(v, str):
        return len(v) <= ORDINARY_STR and all(ord(c) < 128 for c in v)
    if isinstance(v, list):
        return len(v) <= ORDINARY_LIST and all(is_ordinary(x) for x in v)
    return False


def conforms(v, tag: str) -> bool:
    """Does a source return value fit the declared return type?

    The Java signature is derived from types OBSERVED on the benchmark's own test inputs, so a
    fuzz input can take the source somewhere those observations never went. `largest_divisor`
    returns an `int` on every benchmark input and `None` on a negative one; `find_closest_elements`
    returns a list of floats normally and `None` on a list shorter than two. No Java method
    returning a primitive `long` can reproduce `None`, so on those inputs the translation is
    forced to differ no matter what the model writes.

    Counting that as a semantic divergence would measure OUR signature choice rather than the
    model's fidelity, and it would do so in a way that hits every model on the same two problems.
    Such inputs are therefore out of domain: they take the source outside the typed contract the
    migration was specified against. This can only REMOVE divergences, never add one.
    """
    if tag == "int":
        # REPRESENTABILITY, not just Python type. Python integers are unbounded and the declared
        # Java return type is `long`, so a source that legitimately returns a value beyond
        # 2^63-1 forces the translation to differ no matter what the model writes: we fixed the
        # signature, so we, not the model, made that outcome unavoidable. Counting it as a
        # semantic divergence would measure our own harness. 2098 of 60819 integer-returning
        # source outputs, over 25 problems, fall here.
        return isinstance(v, int) and not isinstance(v, bool) and abs(v) <= JAVA_LONG_MAX
    if tag == "float":
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            return False
        try:
            float(v)                       # a Python int too large for a double is not either
        except (OverflowError, ValueError):
            return False
        return True
    if tag == "bool":
        return isinstance(v, bool)
    if tag == "str":
        return isinstance(v, str)
    if tag.startswith("list["):
        inner = tag[5:-1]
        return isinstance(v, list) and all(conforms(x, inner) for x in v)
    return False


def classify(py_r: dict, jv_r: dict, ret_tag: str | None = None) -> str:
    """One input's verdict.

    OUT_OF_DOMAIN  the source did not produce a value, or either side timed out or was lost;
                   such an input carries no information about equivalence and is discarded.
    SOURCE_RAISED  the source raised a normal exception: also out of domain by SPEC.md.
    AGREE          both produced values and they match.
    DIVERGE        the source produced a value and the target did not match it, including the
                   case where the target raised.
    """
    if not py_r.get("ok"):
        if py_r.get("e") in IN_DOMAIN_EXCLUDED:
            return "OUT_OF_DOMAIN"
        return "SOURCE_RAISED"
    if ret_tag is not None and not conforms(py_r.get("v"), ret_tag):
        return "OUT_OF_DOMAIN"
    if not jv_r.get("ok"):
        if jv_r.get("e") in IN_DOMAIN_EXCLUDED:
            return "OUT_OF_DOMAIN"
        return "DIVERGE"
    return "AGREE" if values_equal(py_r.get("v"), jv_r.get("v")) else "DIVERGE"
