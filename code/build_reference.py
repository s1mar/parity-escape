"""Build the strong-oracle input set and the ground-truth outputs of the legacy source.

Also runs two of the four pre-registered controls:
  C4 determinism: every problem's source is executed TWICE on the same inputs; any problem whose
     outputs differ between runs is nondeterministic and is excluded, because parity against a
     nondeterministic source is meaningless.
  eligibility:    a problem needs at least MIN_INDOMAIN in-domain inputs out of N_FUZZ.

Output: data/reference.json  (one record per surviving problem, with inputs and source outputs)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench  # noqa: E402

PY = sys.argv[1] if len(sys.argv) > 1 else sys.executable
OUT = bench.ROOT / "data" / "reference.json"


def main() -> None:
    problems = bench.load_problems()
    print(f"loaded {len(problems)} signature-eligible problems")
    kept, dropped = [], []
    t0 = time.time()

    for i, p in enumerate(problems):
        inputs = bench.fuzz_inputs(p)
        r1 = bench.run_python(PY, p.source, p.entry_point, inputs)
        r2 = bench.run_python(PY, p.source, p.entry_point, inputs)

        nondet = 0
        for a, b in zip(r1, r2):
            if a.get("ok") != b.get("ok"):
                nondet += 1
            elif a.get("ok") and not bench.values_equal(a.get("v"), b.get("v")):
                nondet += 1

        indomain = [k for k, r in enumerate(r1)
                    if r.get("ok") or r.get("e") not in bench.IN_DOMAIN_EXCLUDED]
        # in-domain for the oracle means the SOURCE returned a value
        returned = [k for k, r in enumerate(r1) if r.get("ok")]

        rec = {
            "task_id": p.task_id, "idx": p.idx, "entry_point": p.entry_point,
            "param_types": p.param_types, "return_type": p.return_type,
            "java_decl": p.java_decl, "java_ptags": p.java_ptags,
            "n_inputs": len(inputs), "n_returned": len(returned), "n_nondet": nondet,
        }

        if nondet > 0:
            rec["drop"] = f"nondeterministic ({nondet} inputs)"
            dropped.append(rec)
        elif len(returned) < bench.MIN_INDOMAIN:
            rec["drop"] = f"only {len(returned)} in-domain inputs (< {bench.MIN_INDOMAIN})"
            dropped.append(rec)
        else:
            rec["inputs"] = inputs
            rec["source_out"] = r1
            rec["source"] = p.source
            kept.append(rec)

        el = time.time() - t0
        print(f"[{i + 1:3d}/{len(problems)}] {p.task_id:12s} "
              f"returned={len(returned):4d} nondet={nondet:3d} "
              f"{'DROP: ' + rec['drop'] if 'drop' in rec else 'keep'}  ({el:.0f}s)",
              flush=True)

    OUT.write_text(json.dumps(kept), encoding="utf-8")
    (bench.ROOT / "data" / "reference_dropped.json").write_text(
        json.dumps(dropped, indent=1), encoding="utf-8")
    print(f"\nKEPT {len(kept)}  DROPPED {len(dropped)}")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
