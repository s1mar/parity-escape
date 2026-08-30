"""C1b: validate the marshalling harness itself, independently of any problem or any model.

For every supported Java type tag, compile an echo method and check that a value round-trips
Python -> JSON -> Java -> JSON -> Python unchanged. A marshalling bug would otherwise appear
later as a "divergence" and be attributed to a translation, which is precisely the false
positive this paper exists to warn about.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench  # noqa: E402

RESULT = Path(bench.ROOT) / "data" / "marshal_selftest.json"

PY = sys.argv[1] if len(sys.argv) > 1 else sys.executable

PY_TAGS = ["int", "float", "bool", "str",
           "list[int]", "list[float]", "list[bool]", "list[str]",
           "list[list[int]]", "list[list[float]]", "list[list[str]]"]


def main() -> int:
    root = Path(bench.HERE) / "_selftest"
    failures = []
    total_checked = total_mismatches = 0
    for tag in PY_TAGS:
        jtag = bench.J[tag]
        src = (
            "public class Solution {\n"
            f"    public static {jtag} echo({jtag} a0) {{ return a0; }}\n"
            "}\n"
        )
        wd = root / tag.replace("[", "_").replace("]", "")
        ok, err = bench.compile_java(src, wd)
        if not ok:
            failures.append((tag, "COMPILE", err[:200]))
            continue

        rng = random.Random(7)
        vals = []
        for boundary in (True, False):
            for _ in range(60):
                vals.append([bench.gen_value(tag, rng, boundary)])
        pysrc = "def echo(a0):\n    return a0\n"
        pr = bench.run_python(PY, pysrc, "echo", vals)
        jr = bench.run_java(wd, "echo", [jtag], vals)

        bad = 0
        first = None
        for v, p, j in zip(vals, pr, jr):
            verdict = bench.classify(p, j)
            if verdict == "DIVERGE":
                bad += 1
                if first is None:
                    first = (v, p, j)
        status = "ok" if bad == 0 else f"FAIL {bad}/{len(vals)}"
        print(f"{tag:20s} -> {jtag:12s} {status}")
        total_checked += len(vals)
        total_mismatches += bad
        if bad:
            failures.append((tag, "ROUNDTRIP", str(first)[:300]))

    # This file is the record analyze.py reads for \ConeMismatch. Writing it is what makes that
    # macro a MEASUREMENT rather than the literal 0 it used to be: analyze.py previously asserted
    # zero mismatches unconditionally and never opened any output of this script, so a real
    # failure here would not have changed a single printed number.
    RESULT.parent.mkdir(parents=True, exist_ok=True)
    RESULT.write_text(json.dumps({
        "n_type_tags": len(PY_TAGS), "n_checked": total_checked,
        "n_mismatches": total_mismatches, "ok": not failures,
    }), encoding="utf-8")

    print()
    if failures:
        print("MARSHALLING SELF-TEST FAILED")
        for t, kind, detail in failures:
            print(f"  {t:20s} {kind}: {detail}")
        return 1
    print("MARSHALLING SELF-TEST PASSED: every supported type round-trips exactly.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
