"""Every number in the manuscript must be checkable by a reviewer from the shipped ground truth.

`gates.py` proves each number reaches the page through a macro. That is only half the chain. This
proves the other half: that the macro's VALUE is present in `paper/verify_output.txt`, the file a
reviewer is given to audit against. A number that exists only inside `results.json` is one a
reader cannot check at all, which is a real gap and not a cosmetic one.

Reports three sets:
  USED       macros referenced by main.tex
  UNBACKED   used macros whose value does not appear in verify_output.txt
  UNUSED     macros generated but never referenced (harmless, listed for hygiene)

Usage:  python traceability.py [--strict]
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench  # noqa: E402

TEX = bench.ROOT / "paper" / "main.tex"
NUMS = bench.ROOT / "paper" / "numbers.tex"
VO = bench.ROOT / "paper" / "verify_output.txt"

# Values that are design constants or corpus facts rather than analysis outputs. They are checked
# by gates.py against SPEC.md and the code, not against the results dump.
STRUCTURAL = {
    "NFuzz", "KParity", "NModels", "MutantFloor", "NAmendments",
    "NHumanEval", "NSigEligible", "NProblems", "NDropped", "DiagPerms",
}


def norm(s: str) -> str:
    return s.replace(",", "").replace(" ", "")


def main() -> int:
    tex = TEX.read_text(encoding="utf-8")
    nums = NUMS.read_text(encoding="utf-8")
    vo = norm(VO.read_text(encoding="utf-8")) if VO.exists() else ""

    defined = dict(re.findall(r"\\newcommand\{\\(\w+)\}\{([^}]*)\}", nums))
    used = set(re.findall(r"\\([A-Z][A-Za-z]+)", tex))
    used_defined = sorted(m for m in used if m in defined)

    unbacked, unused = [], []
    for m in used_defined:
        if m in STRUCTURAL:
            continue
        v = norm(defined[m])
        if not v:
            continue
        if v not in vo:
            unbacked.append((m, defined[m]))
    for m in sorted(defined):
        if m not in used:
            unused.append(m)

    print(f"macros defined            : {len(defined)}")
    print(f"macros used in main.tex   : {len(used_defined)}")
    print(f"structural (checked by gates.py, not the dump): {len(STRUCTURAL)}")
    print(f"UNBACKED by verify_output : {len(unbacked)}")
    for m, v in unbacked:
        print(f"    {m} = {v}")
    print(f"unused macros (hygiene)   : {len(unused)}")

    if unbacked:
        print("\nA reviewer cannot check the values above from the shipped ground truth.")
        return 1 if "--strict" in sys.argv else 0
    print("\nTRACEABILITY OK: every reported number is present in verify_output.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
