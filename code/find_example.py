"""Enumerate escape cells suitable as worked examples in the paper.

An escape cell is a (validator, translator, problem) row whose parity suite PASSED while the
translation is divergent. For a worked example the paper wants one with an ORDINARY witness
input (so the divergence is not an artifact of an outlandish input) in a cross-language
category other than integer_width, which the existing median example already covers.

Prints, for each candidate: the cell, the category, and the concrete (input, source output,
target output) triple of an ordinary witness, straight from the oracle record. Nothing here is
typed by hand; the paper's example must quote this output.
"""
from __future__ import annotations

import json
from pathlib import Path

import analyze
import bench
import models

D = Path(__file__).resolve().parent.parent / "data"


def main() -> None:
    tax = {}
    for t in json.loads((D / "taxonomy.json").read_text(encoding="utf-8")):
        tax[(t["model"], t["task_id"])] = t

    ref = {p["task_id"]: p for p in json.loads((D / "reference.json").read_text(encoding="utf-8"))}

    orc = {}
    for m in models.MODELS:
        p = D / "oracle" / f"{m.mid}.json"
        if p.exists():
            for r in json.loads(p.read_text(encoding="utf-8")):
                orc[(m.mid, r["task_id"])] = r

    rows = analyze.joined()
    cands = [r for r in rows
             if not r["caught"] and r["divergent"] and r["ordinary_witness"]]
    print(f"{len(cands)} escape cells with an ordinary witness\n")

    seen = set()
    for r in sorted(cands, key=lambda r: (r["translator"], r["task_id"], r["validator"])):
        key = (r["translator"], r["task_id"])
        t = tax.get(key)
        if t is None or t.get("group") != "cross_language":
            continue
        o = orc.get(key)
        if not o:
            continue
        ins = ref.get(r["task_id"], {}).get("inputs", [])
        wit = None
        for k in o.get("diverge_idx", []):
            if k < len(ins) and bench.is_ordinary(ins[k]):
                wit = k
                break
        if wit is None:
            continue
        if key in seen:
            continue
        seen.add(key)
        exp = (o.get("expected") or [])
        got = (o.get("got") or [])
        py = exp[wit] if wit < len(exp) else t.get("first_example", {}).get("py")
        jv = got[wit] if wit < len(got) else t.get("first_example", {}).get("jv")
        print(f"translator={r['translator']} task={r['task_id']} "
              f"entry={t.get('entry_point')} category={t.get('category')}")
        print(f"  passing validators: "
              f"{sorted(c['validator'] for c in cands if (c['translator'], c['task_id']) == key)}")
        print(f"  witness input #{wit}: {json.dumps(ins[wit])}")
        print(f"  source -> {json.dumps(py)}   target -> {json.dumps(jv)}")
        print(f"  diverges on {t.get('n_diverge')} of {t.get('n_comparable')} comparable inputs")
        print()


if __name__ == "__main__":
    main()
