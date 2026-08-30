"""RQ3: classify each observed divergence into a semantic category.

Categories are assigned by rule from the (input, source output, target output) triple, not by an
LLM, so the classification is deterministic and auditable. The rules are ordered and the FIRST
match wins; the order encodes specificity, most specific first.

`--audit N` prints a random sample of classified divergences with their evidence, so the
labelling can be checked by hand rather than trusted. Every category that the paper names must
survive that check.
"""
from __future__ import annotations

import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench      # noqa: E402
import models     # noqa: E402

D = bench.ROOT / "data"
INT32 = 2 ** 31
INT64 = 2 ** 63


def _flat(v, out=None):
    """Every scalar inside a nested value."""
    if out is None:
        out = []
    if isinstance(v, list):
        for x in v:
            _flat(x, out)
    else:
        out.append(v)
    return out


def _has_big_int(v) -> bool:
    return any(isinstance(x, int) and not isinstance(x, bool) and abs(x) >= INT32
               for x in _flat(v))


def _has_nonascii(v) -> bool:
    return any(isinstance(x, str) and any(ord(c) > 127 for c in x) for x in _flat(v))


def _has_empty(v) -> bool:
    if isinstance(v, list):
        return len(v) == 0 or any(_has_empty(x) for x in v)
    return isinstance(v, str) and len(v) == 0


def _numeric_like(v) -> bool:
    """A number or boolean, or a (possibly nested) list whose scalars are all numbers.

    Booleans count: `will_it_fly([2^62, 2^62], -1)` returns a bool, and the reason the bool
    differs is that the sum overflowed inside the comparison. Excluding bools sent exactly that
    case to the residual bucket.
    """
    if isinstance(v, bool):
        return True
    if isinstance(v, (int, float)):
        return True
    if isinstance(v, list):
        flat = _flat(v)
        return bool(flat) and all(isinstance(x, (int, float)) and not isinstance(x, bool)
                                  for x in flat)
    return False


def _has_negative(v) -> bool:
    return any(isinstance(x, (int, float)) and not isinstance(x, bool) and x < 0
               for x in _flat(v))


def _wraps_to(py, jv) -> bool:
    """Is the target value the source value reduced into signed 64-bit range?"""
    if not (isinstance(py, int) and isinstance(jv, int)):
        return False
    if isinstance(py, bool) or isinstance(jv, bool):
        return False
    if abs(py) < INT64:
        return False
    w = ((py + INT64) % (2 * INT64)) - INT64
    return w == jv


def _saturated(py, jv) -> bool:
    """Did the target clamp at a 64-bit limit where the source kept going?

    Not every width failure is a two's-complement wrap. `sum_squares` over large doubles gives
    Python an exact 38-digit integer and Java exactly Long.MAX_VALUE, which is saturation from a
    `(long)` cast of an out-of-range double. Requiring an exact wrap sent every such case to the
    residual bucket.
    """
    lim = {2 ** 63 - 1, -(2 ** 63)}
    if isinstance(jv, int) and not isinstance(jv, bool) and jv in lim:
        return isinstance(py, (int, float)) and not isinstance(py, bool) and abs(py) >= 2 ** 62
    if isinstance(py, list) and isinstance(jv, list) and len(py) == len(jv):
        return any(_saturated(a, b) for a, b in zip(py, jv))
    return False


def _has_str(v) -> bool:
    return any(isinstance(x, str) for x in _flat(v))


def classify_one(inp, py_r: dict, jv_r: dict) -> str:
    pv = py_r.get("v")
    if not jv_r.get("ok"):
        return f"target_exception:{jv_r.get('e', 'Unknown')}"
    jv = jv_r.get("v")

    # numeric wraparound, checked directly rather than inferred from input magnitude
    if _wraps_to(pv, jv):
        return "integer_overflow"
    if isinstance(pv, list) and isinstance(jv, list) and len(pv) == len(jv):
        if any(_wraps_to(a, b) for a, b in zip(pv, jv)):
            return "integer_overflow"

    # Integer-width divergence is not always visible as a clean two's-complement wrap of the
    # whole return value: `compare([-2, MAX], [-1, -2])` differs elementwise inside a list, and
    # Java's Math.abs(Long.MIN_VALUE) is negative, so the arithmetic that produced the difference
    # is not a single wrap. Any numeric disagreement on an input carrying a value at or beyond
    # 32-bit range is attributed to width rather than left in the residual bucket.
    if _saturated(pv, jv):
        return "integer_width"
    if _has_big_int(inp) and _numeric_like(pv) and _numeric_like(jv):
        return "integer_width"

    # same elements, different order
    if isinstance(pv, list) and isinstance(jv, list) and len(pv) == len(jv) and pv:
        try:
            if sorted(map(repr, pv)) == sorted(map(repr, jv)):
                return "collection_ordering"
        except TypeError:
            pass

    # integer division and rounding of negative operands
    if _has_negative(inp) and isinstance(pv, (int, float)) and isinstance(jv, (int, float)):
        if not isinstance(pv, bool) and not isinstance(jv, bool):
            try:
                if abs(float(pv) - float(jv)) <= 2.0:
                    return "negative_division_rounding"
            except (OverflowError, ValueError):
                pass

    if isinstance(pv, float) or isinstance(jv, float):
        return "float_precision_or_formatting"

    if _has_nonascii(inp):
        return "unicode_or_charset"

    if _has_empty(inp):
        return "empty_or_boundary_input"

    if isinstance(pv, str) or isinstance(jv, str):
        return "string_handling"
    # A string-valued INPUT with a non-string result is still string handling: the case-folding
    # difference in `count_distinct_characters` shows up as 5 versus 9, not as text.
    if _has_str(inp):
        return "string_handling"

    return "other_logic"


# Two groups, because the distinction is the one a modernization team actually acts on.
# CROSS_LANGUAGE divergences arise from Python and Java disagreeing about what an operation
# means, and no amount of careful reimplementation removes them; they need targeted checking.
# LOGIC divergences are the translation simply doing something else, which better models fix.
CROSS_LANGUAGE = {
    "integer_overflow", "integer_width", "negative_division_rounding",
    "float_precision_or_formatting", "string_handling", "unicode_or_charset",
    "collection_ordering", "empty_or_boundary_input",
}


def group_of(category: str) -> str:
    if category.startswith("target_exception:"):
        return "cross_language"
    return "cross_language" if category in CROSS_LANGUAGE else "logic"


def collect() -> list[dict]:
    problems = {p["task_id"]: p for p in
                json.loads((D / "reference.json").read_text(encoding="utf-8"))}
    rows = []
    for m in models.MODELS:
        p = D / "oracle" / f"{m.mid}.json"
        if not p.exists():
            continue
        for r in json.loads(p.read_text(encoding="utf-8")):
            if r.get("status") != "ok" or not r.get("divergent"):
                continue
            prob = problems[r["task_id"]]
            cats = []
            for ex in r.get("examples", []):
                cats.append(classify_one(ex["in"], ex["py"], ex["jv"]))
            if not cats:
                continue
            # a translation's category is its most frequent example category, ties broken by
            # first occurrence, so one translation contributes exactly one label
            top = Counter(cats).most_common(1)[0][0]
            rows.append({
                "model": m.mid, "task_id": r["task_id"], "category": top,
                "group": group_of(top),
                "example_categories": cats,
                "n_diverge": r["n_diverge"], "n_comparable": r["n_comparable"],
                "first_example": r["examples"][0] if r.get("examples") else None,
                "entry_point": prob["entry_point"],
            })
    return rows


def main() -> None:
    audit = 0
    if "--audit" in sys.argv:
        audit = int(sys.argv[sys.argv.index("--audit") + 1])

    rows = collect()
    (D / "taxonomy.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")

    c = Counter(r["category"] for r in rows)
    tot = sum(c.values())
    print(f"{tot} divergent translations classified\n")
    print(f"{'category':36s} {'n':>5s} {'share':>7s}")
    for k, v in c.most_common():
        print(f"{k:36s} {v:5d} {100 * v / max(1, tot):6.1f}%")

    g = Counter(r["group"] for r in rows)
    print(f"\ngroups: cross-language {g['cross_language']} "
          f"({100 * g['cross_language'] / max(1, tot):.1f}%), "
          f"logic {g['logic']} ({100 * g['logic'] / max(1, tot):.1f}%)")

    by_model = defaultdict(Counter)
    gm = defaultdict(Counter)
    for r in rows:
        by_model[r["model"]][r["category"]] += 1
        gm[r["model"]][r["group"]] += 1
    print("\nby translator model (cross-language share of its divergences):")
    for m in models.MODELS:
        if m.mid in gm:
            c = gm[m.mid]
            n = c["cross_language"] + c["logic"]
            print(f"  {m.mid:9s} n={n:3d}  cross-language "
                  f"{100 * c['cross_language'] / max(1, n):5.1f}%   top: "
                  + ", ".join(f"{k}={v}" for k, v in by_model[m.mid].most_common(2)))

    if audit:
        rng = random.Random(0)
        sample = rng.sample(rows, min(audit, len(rows)))
        print("\n" + "=" * 74)
        print("AUDIT SAMPLE (check each label by hand)")
        for r in sample:
            ex = r["first_example"]
            print("-" * 74)
            print(f"{r['model']:9s} {r['task_id']:12s} {r['entry_point']:26s} -> {r['category']}")
            if ex:
                print(f"  in : {json.dumps(ex['in'])[:150]}")
                print(f"  py : {json.dumps(ex['py'])[:150]}")
                print(f"  jv : {json.dumps(ex['jv'])[:150]}")


if __name__ == "__main__":
    main()
