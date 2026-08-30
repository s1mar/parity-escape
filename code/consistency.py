"""This paper's claim gate: one entry point that runs both halves.

Deliberately a thin entry point rather than a copy of either gate. A gate duplicated into a second
file diverges from the original, and a check that has silently stopped running looks exactly like a
check that passes.

Runs both of this paper's claim gates:
  gates.py         venue facts, numbers-via-macros, prose glosses, range claims over macro
                   families, retired claims, citations, house style, source corruption
  traceability.py  every number the manuscript prints is present in the shipped ground-truth dump

Exit code is the number of failing gates, so a caller can branch on it.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import gates          # noqa: E402
import traceability   # noqa: E402


def main() -> int:
    failed = 0

    print("=" * 74)
    print("consistency: gates.py")
    print("=" * 74)
    if gates.main() != 0:
        failed += 1

    print()
    print("=" * 74)
    print("consistency: traceability.py")
    print("=" * 74)
    # strict: a number the paper prints but the shipped dump does not contain is a failure,
    # because it is a number no reviewer can check from the artifact
    sys.argv = [sys.argv[0], "--strict"]
    if traceability.main() != 0:
        failed += 1

    print()
    if failed:
        print(f"CONSISTENCY FAILED: {failed} gate(s)")
    else:
        print("CONSISTENCY PASSED: both gates green")
    return failed


if __name__ == "__main__":
    raise SystemExit(main())
