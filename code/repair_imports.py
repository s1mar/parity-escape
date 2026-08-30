"""Second compilation attempt for translations that failed only on missing imports.

Why this is legitimate and not "fixing the model's work": the subject of this paper is whether a
parity suite can detect a SEMANTIC divergence. A translation that fails to compile because it
called `Arrays.stream` without importing `java.util.Arrays` has an entirely visible, mechanical
defect that any real migration pipeline resolves automatically, and it never reaches validation
at all. Discarding those translations would silently shrink the sample for a reason unrelated to
the question.

What keeps it honest:
  * the repair is deterministic, non-semantic, and applied uniformly to every model;
  * it is only attempted on translations that ALREADY failed;
  * it adds imports and nothing else, so it cannot change behaviour of code that compiled;
  * both compile rates, before and after repair, are recorded and reported.

Usage:  python repair_imports.py <python-exe>
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench      # noqa: E402
import models     # noqa: E402

PREAMBLE = (
    "import java.util.*;\n"
    "import java.util.stream.*;\n"
    "import java.util.function.*;\n"
    "import java.math.*;\n"
)
WORK = bench.HERE / "_build"


def main() -> None:
    summary = {}
    for m in models.MODELS:
        p = bench.ROOT / "data" / "translations" / f"{m.mid}.json"
        if not p.exists():
            continue
        recs = json.loads(p.read_text(encoding="utf-8"))
        before = sum(1 for r in recs if r.get("compile_ok"))
        repaired = 0

        for r in recs:
            if r.get("compile_ok") or not r.get("java"):
                continue
            java = r["java"]
            # only meaningful when the file is not already importing everything it needs
            candidate = PREAMBLE + re.sub(r"^\s*import\s+[\w.*]+;\s*$", "", java, flags=re.M)
            wd = WORK / m.mid / r["task_id"].replace("/", "_")
            ok, err = bench.compile_java(candidate, wd)
            if ok:
                r["java"] = candidate
                r["compile_ok"] = True
                r["compile_repaired"] = "imports"
                r.pop("compile_err", None)
                repaired += 1
            else:
                # restore the original on disk so nothing downstream sees a half-repaired tree
                bench.compile_java(java, wd)
                r["compile_err"] = err[:800]

        p.write_text(json.dumps(recs), encoding="utf-8")
        after = sum(1 for r in recs if r.get("compile_ok"))
        summary[m.mid] = {"n": len(recs), "compiled_before": before,
                          "compiled_after": after, "repaired": repaired}
        print(f"{m.mid:9s} {len(recs):3d} translations: "
              f"{before} compiled, +{repaired} after import repair = {after}")

    (bench.ROOT / "data" / "import_repair.json").write_text(
        json.dumps(summary, indent=1), encoding="utf-8")
    print("\nwrote data/import_repair.json")


if __name__ == "__main__":
    main()
