"""Build paper/verify_output.txt, the file a reviewer audits the manuscript against.

It must contain BOTH:
  (a) the raw analysis blocks, so a reviewer can recompute from the numbers we actually stored;
  (b) the exact rendered value of every macro the paper prints, so a reviewer can reconcile the
      page cell by cell without doing the rounding themselves.

(b) is the part that was missing. 38 of the paper's numbers were uncheckable from the artifact,
because the dump held 0.503 while the page printed 50.3.
That is a real gap in the evidence a reviewer is handed, not a cosmetic one.

Usage:  python make_verify.py <python-exe>
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench  # noqa: E402

HERE = bench.HERE
OUT = bench.ROOT / "paper" / "verify_output.txt"

BLOCKS = [
    "rq1", "rq1_ordinary", "rq2", "rq2_sample", "pairing", "target_timeouts",
    "rq1_by_validator", "rq1_by_translator", "divergence_prevalence",
    "rq4_targeted", "two_factor", "validator_skill_offdiag",
    "translation_subtlety_offdiag", "matrix_miss_rate",
    "within_validator", "within_validator_summary",
    "within_validator_union", "within_validator_union_summary",
    "selector_comparison_loo", "selector_comparison_biased_views",
    "baseline_random", "complementarity", "oracle_supplemented", "benchmark_suite",
    "suite_health", "k_sensitivity",
    "control_c2", "control_c3", "control_c3surv", "control_c5",
]


def main() -> None:
    py = sys.argv[1] if len(sys.argv) > 1 else sys.executable
    parts: list[str] = []

    def run(script: str, *args: str) -> str:
        r = subprocess.run([py, str(HERE / script), *args],
                           capture_output=True, text=True, encoding="utf-8",
                           errors="replace", cwd=str(HERE))
        return r.stdout

    parts.append("=" * 78)
    parts.append("ANALYSIS SUMMARY (code/analyze.py)")
    parts.append("=" * 78)
    parts.append(run("analyze.py"))

    parts.append("=" * 78)
    parts.append("TAXONOMY (code/taxonomy.py)")
    parts.append("=" * 78)
    parts.append(run("taxonomy.py"))

    r = json.loads((bench.ROOT / "data" / "results.json").read_text(encoding="utf-8"))
    parts.append("=" * 78)
    parts.append("RAW RESULT BLOCKS (data/results.json)")
    parts.append("=" * 78)
    for k in BLOCKS:
        if k in r:
            parts.append(f"{k} = {json.dumps(r[k])}")

    # (b) the rendered values, exactly as the manuscript prints them
    nums = (bench.ROOT / "paper" / "numbers.tex").read_text(encoding="utf-8")
    tex = (bench.ROOT / "paper" / "main.tex").read_text(encoding="utf-8")
    defined = re.findall(r"\\newcommand\{\\(\w+)\}\{([^}]*)\}", nums)
    used = set(re.findall(r"\\([A-Z][A-Za-z]+)", tex))
    parts.append("")
    parts.append("=" * 78)
    parts.append("RENDERED VALUES OF EVERY MACRO THE PAPER PRINTS")
    parts.append("Generated from data/results.json by code/make_macros.py. Percentages are")
    parts.append("shown exactly as they appear on the page, so no rounding is left to the reader.")
    parts.append("=" * 78)
    for name, val in defined:
        mark = " " if name in used else "."          # '.' = generated but not printed
        parts.append(f"{mark} {name:26s} = {val}")
    parts.append("")
    parts.append("legend: leading '.' means the macro is generated but not referenced by the "
                 "manuscript.")

    # Strip absolute paths before the dump ships. The scripts print the paths they read, which is
    # a deliberate anti-footgun against reporting on the wrong file, but an absolute path also
    # carries the author's home directory and username, which has no place in a public artifact.
    text = "\n".join(parts) + "\n"
    # A path redaction has to consume the WHOLE path, and a regex cannot be trusted to find the
    # end of one. Two attempts failed here for the same reason, each stopping at a different
    # boundary: a character class excluding path separators stopped just after the username, and
    # one excluding whitespace stopped at a space inside a directory name. Both removed the
    # username, left the machine layout behind, and looked like they had worked. The directory
    # this script is running in is known exactly, so substitute it literally rather than
    # describing its shape, in both separator styles and longest first. The regexes below stay as
    # defence in depth for a path that is not under this root.
    for base in sorted({str(bench.ROOT), str(bench.ROOT.parent), str(Path.home())},
                       key=len, reverse=True):
        for variant in (base, base.replace("\\", "/")):
            text = text.replace(variant, "<PATH>")
    text = re.sub(r"[A-Za-z]:[\\/][^\s\"']*", "<PATH>", text)
    text = re.sub(r"/(?:home|Users)/[^\s\"']*", "<PATH>", text)
    text = re.sub(r"[A-Za-z]:\\\\?Users\\\\?[^\\\\\s\"']+", "<HOME>", text)
    text = re.sub(r"[A-Za-z]:[\\/](?:Users)[\\/][^\\/\s\"']+", "<HOME>", text)
    text = re.sub(r"/(?:home|Users)/[^/\s\"']+", "<HOME>", text)
    OUT.write_text(text, encoding="utf-8")
    n_used = sum(1 for n, _ in defined if n in used)
    print(f"wrote {OUT}")
    print(f"  {len(defined)} macros, {n_used} printed by the manuscript")


if __name__ == "__main__":
    main()
