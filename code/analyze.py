"""Stage 4: compute every number the paper reports, and write them to data/results.json.

Two metrics, deliberately both:

  ESCAPE RATE = P(translation is divergent | its parity suite passed). This is the number a
  practitioner cares about: of the migrations my validation signed off, how many were wrong.
  Its denominator moves with translator quality, so it is a headline, not a contrast.

  MISS RATE   = P(parity suite did not catch it | the translation IS divergent). This conditions
  on the translation, so the self-versus-cross comparison for RQ2 is PAIRED: the same divergent
  translation is scored by all five validators, exactly one of which produced it. That removes
  translator quality from the contrast entirely, which the escape rate cannot do.

The RQ2 permutation test exploits that pairing: under the null that validator identity does not
matter, which of the five validators carries the "self" label is exchangeable within a
translation, so the label is reshuffled within each translation and the difference recomputed.

Usage:  python analyze.py
"""
from __future__ import annotations

import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench      # noqa: E402
import models     # noqa: E402

D = bench.ROOT / "data"
N_BOOT = 10000
N_PERM = 10000
RNG_SEED = 0


def load_oracle() -> dict:
    """(model, task_id) -> record, with an `ordinary_witness` flag added.

    `ordinary_witness` is true when at least one of the inputs on which the translation diverged
    is unremarkable: no large integer, no long or non-ASCII string, no long list. It exists to
    answer the strongest objection to the study, which is that divergences might live only on
    inputs no caller would ever produce.
    """
    inputs_by_task = {}
    ref = D / "reference.json"
    if ref.exists():
        for p in json.loads(ref.read_text(encoding="utf-8")):
            inputs_by_task[p["task_id"]] = p["inputs"]

    out = {}
    for m in models.MODELS:
        p = D / "oracle" / f"{m.mid}.json"
        if not p.exists():
            continue
        for r in json.loads(p.read_text(encoding="utf-8")):
            ins = inputs_by_task.get(r["task_id"])
            r["ordinary_witness"] = bool(
                ins and any(bench.is_ordinary(ins[k]) for k in r.get("diverge_idx", [])
                            if k < len(ins))
            )
            out[(m.mid, r["task_id"])] = r
    return out


def load_parity(mode: str = "plain") -> list[dict]:
    """Every parity-suite record for one mode.

    Globs rather than iterating model ids on purpose: the random baseline writes `random.json`,
    which is not a model id, and a loop over models.MODELS would skip it silently. A missing
    baseline that reads as "no baseline data" rather than as an error is exactly the kind of
    clean-looking failure this project keeps being bitten by.
    """
    d = D / ("parity" if mode == "plain" else f"parity_{mode}")
    rows = []
    if not d.exists():
        return rows
    for p in sorted(d.glob("*.json")):
        rows.extend(json.loads(p.read_text(encoding="utf-8")))
    return rows


def witnessed_by() -> dict[tuple[str, str], set[str]]:
    """(translator, task) -> set of validator ids whose suite caught a divergence there.

    Needed for LEAVE-ONE-OUT ground truth. Comparing two selectors on the union of everything
    everyone found is unfair in whichever direction the union was built: a validator's own
    catches help define the set it is then scored against, which deflates its miss rate. Scoring
    only on oracle-confirmed divergences has the mirror problem, because the random baseline
    draws its inputs FROM the oracle's input list and so inherits an advantage there.
    Leave-one-out removes both: each selector is scored against divergences established by the
    oracle plus every OTHER selector.
    """
    out: dict[tuple[str, str], set[str]] = {}
    for mode in ("plain", "targeted", "random"):
        for r in load_parity(mode):
            if r.get("status") == "ok" and r.get("full", {}).get("caught"):
                out.setdefault((r["translator"], r["task_id"]), set()).add(r["validator"])
    return out


def witnessed_divergent() -> set[tuple[str, str]]:
    """Translations shown to be divergent by ANY parity suite, in any mode.

    The strong oracle's input set is finite, so a parity suite can occasionally catch a
    divergence on an input the oracle never tried. Without this union the row would read
    `caught = True, divergent = False`, which is not merely a lost observation: it is a
    contradiction, and it would understate how many translations are actually divergent. A
    translation is divergent if ANY input distinguishes it from the source, whoever supplied it.
    """
    out = set()
    for mode in ("plain", "targeted", "random"):
        for r in load_parity(mode):
            if r.get("status") == "ok" and r.get("full", {}).get("caught"):
                out.add((r["translator"], r["task_id"]))
    return out


def joined(mode: str = "plain", k: str = "full") -> list[dict]:
    """One row per (validator, translator, problem) that is scorable on both sides."""
    orc = load_oracle()
    extra = witnessed_divergent()
    wby = witnessed_by()
    out = []
    for r in load_parity(mode):
        if r.get("status") != "ok":
            continue
        o = orc.get((r["translator"], r["task_id"]))
        if o is None or o.get("status") != "ok" or o.get("divergent") is None:
            continue
        kk = r.get(k)
        if not kk or kk.get("degenerate"):
            continue
        out.append({
            "validator": r["validator"], "translator": r["translator"],
            "task_id": r["task_id"], "self": r["validator"] == r["translator"],
            "divergent": bool(o["divergent"]) or (r["translator"], r["task_id"]) in extra,
            "oracle_divergent": bool(o["divergent"]),
            # LEAVE-ONE-OUT: divergent according to the oracle plus every validator EXCEPT this
            # one. This is the unbiased ground truth for comparing selectors, because a
            # selector's own catches no longer help define the set it is scored against.
            "divergent_loo": bool(o["divergent"]) or bool(
                wby.get((r["translator"], r["task_id"]), set()) - {r["validator"]}),
            "ordinary_witness": bool(o.get("ordinary_witness")),
            "caught": bool(kk["caught"]),
            "n_indomain": kk["n_indomain"],
            "mode": mode,
        })
    return out


def boot_ci(items: list[tuple[str, int, int]], rng: random.Random,
            n: int = N_BOOT) -> tuple[float, float, float]:
    """Cluster bootstrap over problems. `items` is (cluster, numerator, denominator)."""
    by = defaultdict(list)
    for c, num, den in items:
        by[c].append((num, den))
    clusters = list(by)
    if not clusters:
        return (float("nan"),) * 3
    tot_n = sum(x for c in clusters for x, _ in by[c])
    tot_d = sum(y for c in clusters for _, y in by[c])
    point = tot_n / tot_d if tot_d else float("nan")
    reps = []
    for _ in range(n):
        num = den = 0
        for _ in clusters:
            c = clusters[rng.randrange(len(clusters))]
            for x, y in by[c]:
                num += x
                den += y
        if den:
            reps.append(num / den)
    reps.sort()
    if not reps:
        return point, float("nan"), float("nan")
    lo = reps[int(0.025 * len(reps))]
    hi = reps[min(len(reps) - 1, int(0.975 * len(reps)))]
    return point, lo, hi


def rate(rows: list[dict], num_key, den_key) -> tuple[int, int, float]:
    den = [r for r in rows if den_key(r)]
    num = [r for r in den if num_key(r)]
    return len(num), len(den), (len(num) / len(den) if den else float("nan"))


def perm_test_self(rows: list[dict], rng: random.Random, n: int = N_PERM) -> dict:
    """Paired permutation test on MISS RATE, self versus cross, within each divergent translation.

    Each translation contributes one observation per validator. Under the null, which validator
    is the 'self' one is exchangeable within the translation, so the label is reshuffled there.
    Only translations that actually have a self observation and at least one cross observation
    can carry information, and only those are used.
    """
    by = defaultdict(list)
    for r in rows:
        # oracle-confirmed only: see the note at the RQ2 call site. Selecting on the union would
        # select on the very behaviour being tested.
        if r.get("divergent_loo"):
            by[(r["translator"], r["task_id"])].append(r)

    groups = []
    for _key, rs in by.items():
        selves = [r for r in rs if r["self"]]
        cross = [r for r in rs if not r["self"]]
        if len(selves) == 1 and cross:
            groups.append((selves[0], cross))
    if not groups:
        return {"n_groups": 0}

    def stat(assign: list[int]) -> float:
        sm = sc = cm = cc = 0
        for gi, (s, c) in enumerate(groups):
            pool = [s] + c
            si = assign[gi]
            for j, r in enumerate(pool):
                miss = 0 if r["caught"] else 1
                if j == si:
                    sm += miss
                    sc += 1
                else:
                    cm += miss
                    cc += 1
        return (sm / sc if sc else float("nan")) - (cm / cc if cc else float("nan"))

    obs = stat([0] * len(groups))
    ge = 0
    for _ in range(n):
        assign = [rng.randrange(1 + len(c)) for _s, c in groups]
        if abs(stat(assign)) >= abs(obs) - 1e-12:
            ge += 1
    self_miss = sum(0 if s["caught"] else 1 for s, _ in groups) / len(groups)
    cross_obs = [r for _s, c in groups for r in c]
    cross_miss = sum(0 if r["caught"] else 1 for r in cross_obs) / len(cross_obs)
    return {
        "n_groups": len(groups), "n_cross_obs": len(cross_obs),
        "self_miss_rate": self_miss, "cross_miss_rate": cross_miss,
        "diff_pp": 100 * (self_miss - cross_miss),
        "p_value": (ge + 1) / (n + 1),
    }


def diagonal_permutation_test(rows: list[dict]) -> dict:
    """EXACT test for a diagonal effect in the validator x translator matrix.

    Why the pre-registered test is not usable, discovered during the run and reported alongside
    this one: it permuted the "self" label among the five validators scoring a translation, whose
    null is that validators are exchangeable. They are emphatically not. On partial data the
    Gemini validator caught 96 of 157 cells and the Qwen validator 4 of 53. Under that inequality,
    "self" and "cross" differ in WHICH validators populate them as well as in self-ness, and the
    naive contrast confounds validator skill and translator difficulty with the effect of
    interest.

    This test removes both. The matrix has one miss rate per (validator, translator) cell, and the
    diagonal is self-validation. For every permutation pi of the model set we read off the
    pseudo-diagonal {(V, pi(V))} and compute its mean miss rate. Permuting columns preserves every
    row marginal and every column marginal exactly, so validator skill and translator difficulty
    are held fixed by construction, and the only thing that varies is which cells are called
    diagonal. With five models all 120 permutations are enumerated, so the p-value is exact rather
    than sampled.
    """
    from itertools import permutations

    ids = [m.mid for m in models.MODELS]
    cell = {}
    for r in rows:
        if not r.get("divergent_loo"):
            continue
        k = (r["validator"], r["translator"])
        n, miss = cell.get(k, (0, 0))
        cell[k] = (n + 1, miss + (0 if r["caught"] else 1))

    present = [i for i in ids if any((i, j) in cell for j in ids)]
    if len(present) < 3:
        return {"n_models": len(present)}

    def diag_mean(perm) -> float | None:
        num = den = 0
        for v, t in zip(present, perm):
            if (v, t) in cell:
                n, miss = cell[(v, t)]
                num += miss
                den += n
        return num / den if den else None

    obs = diag_mean(present)
    if obs is None:
        return {"n_models": len(present)}
    stats = [d for p in permutations(present) if (d := diag_mean(p)) is not None]
    ge = sum(1 for s in stats if s >= obs - 1e-12)
    off_num = off_den = 0
    for (v, t), (n, miss) in cell.items():
        if v != t:
            off_num += miss
            off_den += n
    ss = sorted(stats)
    lo = ss[int(0.025 * len(ss))]
    hi = ss[min(len(ss) - 1, int(0.975 * len(ss)))]
    return {
        "n_models": len(present),
        "n_permutations": len(stats),
        "diagonal_miss_rate": obs,
        "offdiagonal_miss_rate": off_num / off_den if off_den else float("nan"),
        "mean_pseudodiagonal": sum(stats) / len(stats),
        # The null spread of the pseudo-diagonal, which is the margin-preserving reference. The
        # naive self-versus-cross bootstrap CI is NOT a valid bound on the effect: it is the
        # confounded comparison, and it can exclude zero while this test does not, precisely
        # because it lets row and column composition vary.
        "null_range": [lo, hi],
        "diagonal_excess_pp": 100 * (obs - sum(stats) / len(stats)),
        "p_value_one_sided_diag_worse": ge / len(stats),
    }


def main() -> None:
    rng = random.Random(RNG_SEED)
    res: dict = {"n_boot": N_BOOT, "n_perm": N_PERM}

    rows = joined("plain", "full")
    res["n_cells"] = len(rows)
    if not rows:
        print("no joined rows yet")
        (D / "results.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
        return

    # ---- RQ1: escape rate, pooled and per validator
    passed = lambda r: not r["caught"]                                   # noqa: E731
    esc = lambda r: r["divergent"]                                       # noqa: E731
    n, d, pt = rate(rows, esc, passed)
    _p, lo, hi = boot_ci([(r["task_id"], int(esc(r)), 1) for r in rows if passed(r)], rng)
    res["rq1"] = {"n_escaped": n, "n_passed": d, "escape_rate": pt, "ci": [lo, hi]}

    # Ground-truth sensitivity for RQ1: the escape rate under the ORACLE-ONLY divergence
    # indicator, beside the union D above. Deliberately no CI: it is a sensitivity bound, and a
    # bootstrap here would consume rng draws and silently shift every CI computed after it.
    esc_orc = lambda r: r["oracle_divergent"]                            # noqa: E731
    n_oo, d_oo, pt_oo = rate(rows, esc_orc, passed)
    res["rq1_oracle_only"] = {"n_escaped": n_oo, "n_passed": d_oo, "escape_rate": pt_oo}

    # Robustness for the disclosed import-repair pass: the escape rate with every repaired
    # translation excluded. The per-translation `compile_repaired` flags in the translations
    # files are the authoritative record (the repair stage ran more than once and
    # import_repair.json keeps only the last run's summary). No rng consumed.
    repaired = set()
    per_model_rep = {}
    for m in models.MODELS:
        p = D / "translations" / f"{m.mid}.json"
        cnt = 0
        if p.exists():
            for t in json.loads(p.read_text(encoding="utf-8")):
                if t.get("compile_repaired"):
                    repaired.add((m.mid, t["task_id"]))
                    cnt += 1
        per_model_rep[m.mid] = cnt
    rx = [r for r in rows if (r["translator"], r["task_id"]) not in repaired]
    n_xr, d_xr, pt_xr = rate(rx, esc, passed)
    res["rq1_excl_repaired"] = {"n_repaired_translations": len(repaired),
                                "repaired_by_model": per_model_rep,
                                "n_escaped": n_xr, "n_passed": d_xr, "escape_rate": pt_xr}

    res["rq1_by_validator"] = {}
    for m in models.MODELS:
        sub = [r for r in rows if r["validator"] == m.mid]
        if sub:
            nn, dd, pp = rate(sub, esc, passed)
            res["rq1_by_validator"][m.mid] = {"n_escaped": nn, "n_passed": dd, "escape_rate": pp}

    res["rq1_by_translator"] = {}
    for m in models.MODELS:
        sub = [r for r in rows if r["translator"] == m.mid]
        if sub:
            nn, dd, pp = rate(sub, esc, passed)
            res["rq1_by_translator"][m.mid] = {"n_escaped": nn, "n_passed": dd, "escape_rate": pp}

    # ---- robustness: divergences that have at least one ORDINARY witness input
    esc_ord = lambda r: r["divergent"] and r["ordinary_witness"]              # noqa: E731
    n_o, d_o, pt_o = rate(rows, esc_ord, passed)
    _p, lo_o, hi_o = boot_ci([(r["task_id"], int(esc_ord(r)), 1) for r in rows if passed(r)], rng)
    res["rq1_ordinary"] = {"n_escaped": n_o, "n_passed": d_o,
                           "escape_rate": pt_o, "ci": [lo_o, hi_o]}
    div_all = [r for r in rows if r["divergent"]]
    res["ordinary_share_of_divergences"] = (
        sum(1 for r in div_all if r["ordinary_witness"]) / len(div_all) if div_all else float("nan")
    )

    # ---- divergence prevalence (denominator context)
    orc = load_oracle()
    prev = {}
    for m in models.MODELS:
        rs = [v for (mid, _t), v in orc.items() if mid == m.mid and v.get("status") == "ok"]
        if rs:
            prev[m.mid] = {
                "n_usable": len(rs),
                "n_divergent": sum(1 for v in rs if v.get("divergent")),
                "divergence_rate": sum(1 for v in rs if v.get("divergent")) / len(rs),
            }
    res["divergence_prevalence"] = prev

    # ---- RQ2: miss rate on divergent translations, self vs cross
    #
    # RQ2 uses ORACLE-CONFIRMED divergences only, not the union. This matters. Under the union, a
    # translation enters the divergent set partly BECAUSE some validator caught it, so
    # conditioning on membership conditions on validator behaviour: exactly the quantity being
    # compared. The oracle is an independent criterion, decided before any parity suite ran, so
    # selection into the RQ2 sample cannot depend on which validator is being scored.
    # RQ1 keeps the union, where it is correct and desirable: a translation that validator A
    # passed and validator B caught genuinely IS an escape for A.
    # PRIMARY sample is LEAVE-ONE-OUT: divergent according to the oracle plus every validator
    # except the one being scored. It is the only one of the three that is clean in both
    # directions. Oracle-only keeps selection independent of validator behaviour but inherits the
    # fuzzer's blind spot for structured inputs, which is where model validators are strongest.
    # The union fixes that coverage gap and reintroduces selection, because a cell can enter the
    # set by being caught by the very validator under test. Both are reported as robustness.
    div = [r for r in rows if r["divergent_loo"]]
    div_oracle = [r for r in rows if r["oracle_divergent"]]
    div_union = [r for r in rows if r["divergent"]]
    res["rq2_sample"] = {"leave_one_out_cells": len(div),
                         "oracle_confirmed_cells": len(div_oracle),
                         "union_cells": len(div_union)}

    # How complete is the pairing, actually? A divergent translation is scored by however many
    # validators produced a USABLE suite for it, which is often fewer than five: degenerate and
    # unparseable suites drop out. Claiming "scored by all five" would be false, so the pairing
    # depth is measured and reported rather than assumed.
    per = Counter((r["translator"], r["task_id"]) for r in div)
    if per:
        counts = sorted(per.values())
        res["pairing"] = {
            "n_divergent_translations": len(per),
            "n_scored_by_all_validators": sum(1 for v in counts if v == len(models.MODELS)),
            "median_validators_per_translation": counts[len(counts) // 2],
            "distribution": dict(Counter(counts)),
        }

    # Target-timeout total, promised in Threats and previously never surfaced.
    res["target_timeouts"] = sum(
        (o.get("n_target_timeout") or 0) for o in orc.values() if o.get("status") == "ok")

    # C1 and C4 as NUMBERS, not prose. Neither claim was checkable from the results dump, and
    # C4 read as being in tension with the corpus section, which lists nondeterminism as a
    # possible drop reason. It is a possible reason that
    # never fired: every drop was for too few in-domain inputs.
    dropped = json.loads((D / "reference_dropped.json").read_text(encoding="utf-8"))
    res["control_c4"] = {
        "n_dropped": len(dropped),
        "n_dropped_nondeterministic": sum(1 for x in dropped if "nondet" in x.get("drop", "")),
        "n_dropped_too_few_indomain": sum(1 for x in dropped if "in-domain" in x.get("drop", "")),
    }
    # Read the ACTUAL result of code/selftest_marshal.py rather than asserting it. This block used
    # to write n_mismatches: 0 unconditionally without ever opening that script's output, which
    # is exactly the "hardcoded measurement instead of a macro" defect class this paper's own
    # gate self-test re-injects: a real failure there would not have changed a single printed
    # number. Refuse rather than guess if the file is absent, same as check_results_present does
    # for results.json.
    marshal_path = D / "marshal_selftest.json"
    if not marshal_path.exists():
        raise SystemExit(
            "data/marshal_selftest.json is missing: run `python selftest_marshal.py <py>` "
            "before analyze.py, or \\ConeMismatch has no measurement behind it")
    ms = json.loads(marshal_path.read_text(encoding="utf-8"))
    if ms.get("n_type_tags") != len(bench.J) or not ms.get("ok"):
        raise SystemExit(
            f"marshal_selftest.json reports {ms.get('n_mismatches')} mismatch(es) over "
            f"{ms.get('n_type_tags')} type tags (expected {len(bench.J)} tags, all passing); "
            f"the marshalling harness is not trustworthy, so nothing downstream is either")
    res["control_c1"] = {"n_type_tags_round_tripped": ms["n_type_tags"],
                         "n_mismatches": ms["n_mismatches"],
                         "source": "code/selftest_marshal.py"}

    # The two divergent-translation counts differ by one and both are correct for their scope.
    # Stated explicitly so a reader does not have to reconcile them.
    res["divergent_translation_counts"] = {
        "all_usable_translations": sum(v["n_divergent"] for v in prev.values()),
        "with_at_least_one_scorable_parity_cell": len(
            {(r["translator"], r["task_id"]) for r in rows if r["oracle_divergent"]}),
        "note": "the difference is translations that are divergent but for which no validator "
                "produced a scorable suite, so they cannot enter any miss-rate denominator",
    }
    sm, sd, sp = rate(div, lambda r: not r["caught"], lambda r: r["self"])
    cm, cd, cp = rate(div, lambda r: not r["caught"], lambda r: not r["self"])
    _p1, slo, shi = boot_ci([(r["task_id"], int(not r["caught"]), 1) for r in div if r["self"]], rng)
    _p2, clo, chi = boot_ci([(r["task_id"], int(not r["caught"]), 1)
                             for r in div if not r["self"]], rng)
    res["rq2"] = {
        "n_divergent_cells": len(div),
        "self": {"n_missed": sm, "n": sd, "miss_rate": sp, "ci": [slo, shi]},
        "cross": {"n_missed": cm, "n": cd, "miss_rate": cp, "ci": [clo, chi]},
        "diff_pp": 100 * (sp - cp) if sd and cd else float("nan"),
        "permutation_preregistered": perm_test_self(rows, rng),
        "permutation_diagonal_exact": diagonal_permutation_test(rows),
    }

    # ---- What size of self-penalty could this study have detected?
    # A null result is not evidence of absence unless the study could have seen the effect. The
    # bootstrap CI on the self-minus-cross difference states the range the data is consistent
    # with, which is the honest way to report a failure to reject.
    diff_items = []
    for r in div:
        diff_items.append((r["task_id"], 1 if not r["caught"] else 0, 1, r["self"]))
    if diff_items:
        by_prob = defaultdict(list)
        for t, miss, _one, is_self in diff_items:
            by_prob[t].append((miss, is_self))
        probs_l = list(by_prob)
        reps = []
        for _ in range(N_BOOT):
            sm = sc = cm = cc = 0
            for _ in probs_l:
                t = probs_l[rng.randrange(len(probs_l))]
                for miss, is_self in by_prob[t]:
                    if is_self:
                        sm += miss
                        sc += 1
                    else:
                        cm += miss
                        cc += 1
            if sc and cc:
                reps.append(sm / sc - cm / cc)
        reps.sort()
        if reps:
            res["rq2"]["diff_ci_pp"] = [100 * reps[int(0.025 * len(reps))],
                                        100 * reps[min(len(reps) - 1, int(0.975 * len(reps)))]]

    # ---- RQ2 on the UNION sample, as a robustness check against the opposite bias.
    #
    # The two samples are biased in opposite directions and neither dominates. Restricting to
    # ORACLE-confirmed divergences keeps selection independent of validator behaviour, but the
    # oracle is a type-directed fuzzer, so it under-represents divergences that only structured
    # inputs reveal, which is exactly where a model validator is strongest; that could bias the
    # RQ2 sample toward boundary and width errors. Using the UNION fixes the coverage problem and
    # creates a selection problem, because a translation can enter the set by being caught. If
    # the two samples agree, the conclusion does not rest on either bias.
    wv_union = {}
    for m in models.MODELS:
        own = [r for r in div_union if r["validator"] == m.mid and r["translator"] == m.mid]
        oth = [r for r in div_union if r["validator"] == m.mid and r["translator"] != m.mid]
        if own and oth:
            mo = sum(1 for r in own if not r["caught"]) / len(own)
            mt = sum(1 for r in oth if not r["caught"]) / len(oth)
            wv_union[m.mid] = {"n_own": len(own), "miss_own": mo,
                               "n_other": len(oth), "miss_other": mt,
                               "diff_pp": 100 * (mo - mt)}
    res["within_validator_union"] = wv_union
    if wv_union:
        res["within_validator_union_summary"] = {
            "n_validators": len(wv_union),
            "n_worse_on_own": sum(1 for v in wv_union.values() if v["diff_pp"] > 0),
            "mean_diff_pp": sum(v["diff_pp"] for v in wv_union.values()) / len(wv_union),
        }

    # ---- within-validator breakdown: each validator on ITS OWN translations versus on others'.
    # Holds validator skill fixed by construction, which the pooled self-versus-cross contrast
    # cannot do. Reported per validator because pooling would reintroduce the same confound.
    wv = {}
    for m in models.MODELS:
        own = [r for r in div if r["validator"] == m.mid and r["translator"] == m.mid]
        oth = [r for r in div if r["validator"] == m.mid and r["translator"] != m.mid]
        if own and oth:
            wv[m.mid] = {
                "n_own": len(own),
                "miss_own": sum(1 for r in own if not r["caught"]) / len(own),
                "n_other": len(oth),
                "miss_other": sum(1 for r in oth if not r["caught"]) / len(oth),
            }
            wv[m.mid]["diff_pp"] = 100 * (wv[m.mid]["miss_own"] - wv[m.mid]["miss_other"])
    res["within_validator"] = wv
    if wv:
        worse = sum(1 for v in wv.values() if v["diff_pp"] > 0)
        res["within_validator_summary"] = {
            "n_validators": len(wv), "n_worse_on_own": worse,
            "mean_diff_pp": sum(v["diff_pp"] for v in wv.values()) / len(wv),
        }

    # ---- escape rate on the diagonal vs off it (the SPEC-declared framing)
    ns, ds, ps = rate([r for r in rows if r["self"]], esc, passed)
    nc, dc, pc = rate([r for r in rows if not r["self"]], esc, passed)
    res["rq2_escape"] = {
        "self": {"n_escaped": ns, "n_passed": ds, "escape_rate": ps},
        "cross": {"n_escaped": nc, "n_passed": dc, "escape_rate": pc},
        "diff_pp": 100 * (ps - pc) if ds and dc else float("nan"),
    }

    # ---- full 5x5 matrix of miss rates on divergent translations
    mat = {}
    for v in models.MODELS:
        mat[v.mid] = {}
        for t in models.MODELS:
            sub = [r for r in div if r["validator"] == v.mid and r["translator"] == t.mid]
            if sub:
                mat[v.mid][t.mid] = {
                    "n": len(sub),
                    "miss_rate": sum(1 for r in sub if not r["caught"]) / len(sub),
                }
    res["matrix_miss_rate"] = mat

    # ---- the two factors the matrix actually shows, separated.
    # ROW effect: how good a validator is, measured only on OTHER models' translations.
    # COLUMN effect: how hard a translator's divergences are, measured only by OTHER validators.
    # Both exclude the diagonal, so neither is contaminated by self-validation.
    rows_eff, cols_eff = {}, {}
    for m in models.MODELS:
        n_r = miss_r = n_c = miss_c = 0
        for o in models.MODELS:
            if o.mid == m.mid:
                continue
            c = mat.get(m.mid, {}).get(o.mid)
            if c:
                n_r += c["n"]
                miss_r += c["miss_rate"] * c["n"]
            c = mat.get(o.mid, {}).get(m.mid)
            if c:
                n_c += c["n"]
                miss_c += c["miss_rate"] * c["n"]
        if n_r:
            rows_eff[m.mid] = {"n": n_r, "miss_rate": miss_r / n_r}
        if n_c:
            cols_eff[m.mid] = {"n": n_c, "miss_rate": miss_c / n_c}
    res["validator_skill_offdiag"] = rows_eff
    res["translation_subtlety_offdiag"] = cols_eff
    if rows_eff and cols_eff:
        best_v = min(rows_eff.items(), key=lambda kv: kv[1]["miss_rate"])
        hard_t = max(cols_eff.items(), key=lambda kv: kv[1]["miss_rate"])
        worst = None
        for v in models.MODELS:
            for t in models.MODELS:
                c = mat.get(v.mid, {}).get(t.mid)
                if c and v.mid != t.mid and (worst is None or c["miss_rate"] > worst[2]):
                    worst = (v.mid, t.mid, c["miss_rate"], c["n"])
        res["two_factor"] = {
            "best_validator": best_v[0], "best_validator_miss": best_v[1]["miss_rate"],
            "hardest_translator": hard_t[0], "hardest_translator_miss": hard_t[1]["miss_rate"],
            "worst_cell": {"validator": worst[0], "translator": worst[1],
                           "miss_rate": worst[2], "n": worst[3]} if worst else None,
        }

    # ---- K sensitivity
    res["k_sensitivity"] = {}
    for k in bench.K_SENSITIVITY:
        rk = joined("plain", f"k{k}")
        if rk:
            nn, dd, pp = rate(rk, esc, passed)
            dv = [r for r in rk if r["divergent"]]
            _a, _b, msp = rate(dv, lambda r: not r["caught"], lambda r: r["self"])
            _c, _d2, mcp = rate(dv, lambda r: not r["caught"], lambda r: not r["self"])
            res["k_sensitivity"][f"k{k}"] = {
                "n_cells": len(rk), "escape_rate": pp, "n_passed": dd,
                "self_miss": msp, "cross_miss": mcp,
            }

    # ---- baseline: K random type-directed inputs, no model involved
    rnd = joined("random", "full")
    if rnd:
        nn, dd, pp = rate(rnd, esc, passed)
        res["baseline_random"] = {
            "n_cells": len(rnd), "escape_rate": pp, "n_passed": dd, "n_escaped": nn,
        }
        # Miss rates compared on LEAVE-ONE-OUT ground truth, restricted to the cells both
        # selectors actually scored, so neither is helped by having defined the target set and
        # neither is scored on translations the other never saw.
        rnd_by = {(r["translator"], r["task_id"]): r for r in rnd}
        llm_by: dict[tuple[str, str], list[dict]] = {}
        for r in rows:
            llm_by.setdefault((r["translator"], r["task_id"]), []).append(r)

        r_miss = r_n = l_miss = l_n = 0
        for key, rr in rnd_by.items():
            ll = llm_by.get(key)
            if not ll:
                continue
            if rr["divergent_loo"]:
                r_n += 1
                r_miss += 0 if rr["caught"] else 1
            for x in ll:
                if x["divergent_loo"]:
                    l_n += 1
                    l_miss += 0 if x["caught"] else 1
        res["selector_comparison_loo"] = {
            "random": {"n": r_n, "miss_rate": r_miss / max(1, r_n)},
            "llm": {"n": l_n, "miss_rate": l_miss / max(1, l_n)},
            "note": "leave-one-out ground truth: oracle plus every OTHER selector",
        }
        # Also report the two biased views, with the direction of each bias named, so a reader
        # can see that the conclusion does not depend on the choice.
        dvr_u = [r for r in rnd if r["divergent"]]
        dvr_o = [r for r in rnd if r["oracle_divergent"]]
        _a, _b, rm_u = rate(dvr_u, lambda r: not r["caught"], lambda r: True)
        _c, _d, rm_o = rate(dvr_o, lambda r: not r["caught"], lambda r: True)
        _e, _f, lm_u = rate(div_union, lambda r: not r["caught"], lambda r: True)
        # div_oracle, not div. The random arm above is scored on ORACLE-CONFIRMED cells, so the
        # LLM arm must be too or the two sides of a view named "oracle_only" are computed over
        # different populations. Using the leave-one-out set here scored the LLM on 854 cells
        # against the random selector's 162 and shipped a dump reading as though the random
        # selector wins under oracle-only scoring, which is the reverse of the truth: like for
        # like the LLM misses fewer. No macro from this block reaches the manuscript, so no
        # printed number was ever wrong, but the replication package contradicted the paper.
        _g, _h, lm_o = rate(div_oracle, lambda r: not r["caught"], lambda r: True)
        res["selector_comparison_biased_views"] = {
            "union_favours_llm": {"random_miss": rm_u, "llm_miss": lm_u},
            "oracle_only_favours_random": {"random_miss": rm_o, "llm_miss": lm_o},
        }

    # ---- complementarity: an LLM suite AND K random type-directed inputs, together
    # Costs nothing extra: both sets already exist. The two are expected to be complementary
    # rather than redundant, because the failures they see are different in kind. The model
    # probes structured, meaning-carrying inputs (a paren matcher gets " ( ( a ) ) ( b ) "); the
    # generator probes magnitudes and empties that no one would think to write down.
    rnd_caught = {}
    for r in load_parity("random"):
        if r.get("status") == "ok":
            rnd_caught[(r["translator"], r["task_id"])] = bool(r["full"]["caught"])
    if rnd_caught:
        pool = [r for r in rows if (r["translator"], r["task_id"]) in rnd_caught
                and r["divergent_loo"]]
        if pool:
            def combo(r):
                return r["caught"] or rnd_caught[(r["translator"], r["task_id"])]
            miss_llm = sum(1 for r in pool if not r["caught"]) / len(pool)
            miss_rnd = sum(1 for r in pool
                           if not rnd_caught[(r["translator"], r["task_id"])]) / len(pool)
            miss_both = sum(1 for r in pool if not combo(r)) / len(pool)
            # how much of each one's catching is unique to it
            only_llm = sum(1 for r in pool
                           if r["caught"] and not rnd_caught[(r["translator"], r["task_id"])])
            only_rnd = sum(1 for r in pool
                           if not r["caught"] and rnd_caught[(r["translator"], r["task_id"])])
            res["complementarity"] = {
                "n_cells": len(pool),
                "miss_rate_llm_only": miss_llm,
                "miss_rate_random_only": miss_rnd,
                "miss_rate_combined": miss_both,
                "caught_only_by_llm": only_llm,
                "caught_only_by_random": only_rnd,
                "gain_over_llm_pp": 100 * (miss_llm - miss_both),
            }

    # ---- how often a parity suite found a divergence the strong oracle missed
    # Counted over unique TRANSLATIONS, not cells: each translation appears once per validator,
    # so a cell count would multiply the same fact by five.
    tr_union, tr_oracle = set(), set()
    for r in rows:
        key = (r["translator"], r["task_id"])
        if r["divergent"]:
            tr_union.add(key)
        if r["oracle_divergent"]:
            tr_oracle.add(key)
    only_suite = tr_union - tr_oracle
    res["oracle_supplemented"] = {
        "n_divergent_translations_union": len(tr_union),
        "n_divergent_translations_oracle": len(tr_oracle),
        "n_found_only_by_a_suite": len(only_suite),
        "share_of_divergent_translations": len(only_suite) / max(1, len(tr_union)),
    }

    # ---- RQ4: divergence-targeted prompting, PAIRED on the diagonal.
    # The targeted arm was run on self-validation cells only, so it must be compared against the
    # PLAIN DIAGONAL, cell for cell, not against the whole plain matrix. Comparing it to the full
    # matrix would mix in every cross-validation cell and silently change what is being measured.
    tro = joined("targeted", "full")
    if tro:
        plain_diag = {(r["validator"], r["task_id"]): r for r in rows if r["self"]}
        pairs = []
        for r in tro:
            if r["validator"] != r["translator"]:
                continue
            p0 = plain_diag.get((r["validator"], r["task_id"]))
            if p0 is not None and (p0["divergent_loo"] or r["divergent_loo"]):
                pairs.append((p0, r))
        if pairs:
            mp = sum(1 for a, _b in pairs if not a["caught"]) / len(pairs)
            mt = sum(1 for _a, b in pairs if not b["caught"]) / len(pairs)
            gained = sum(1 for a, b in pairs if b["caught"] and not a["caught"])
            lost = sum(1 for a, b in pairs if a["caught"] and not b["caught"])
            # McNemar exact, two-sided, on the discordant pairs
            n_d = gained + lost
            pv = float("nan")
            if n_d:
                from math import comb
                k = min(gained, lost)
                tail = sum(comb(n_d, i) for i in range(0, k + 1)) / (2 ** n_d)
                pv = min(1.0, 2 * tail)
            res["rq4_targeted"] = {
                "n_paired_cells": len(pairs),
                "miss_rate_plain_diag": mp, "miss_rate_targeted": mt,
                "miss_rate_reduction_pp": 100 * (mp - mt),
                "caught_only_targeted": gained, "caught_only_plain": lost,
                "mcnemar_p": pv,
            }
        nn, dd, pp = rate(tro, esc, passed)
        res.setdefault("rq4_targeted", {}).update(
            {"n_cells": len(tro), "escape_rate": pp, "n_passed": dd})

    # ---- degenerate and unparseable suites (reported, not hidden)
    raw = load_parity("plain")
    # A degenerate suite is NOT a pass, and neither is an unparseable one: both are excluded from
    # the escape-rate denominator by `joined()` and reported here instead, so that a validator
    # cannot look safe by failing to produce a usable suite at all.
    res["suite_health"] = {
        "n_total": len(raw),
        "n_ok": sum(1 for r in raw if r.get("status") == "ok"),
        "n_unparseable": sum(1 for r in raw if r.get("status") == "unparseable"),
        "n_call_failed": sum(1 for r in raw if r.get("status") == "call_failed"),
        "n_degenerate": sum(1 for r in raw
                            if r.get("status") == "ok" and r.get("full", {}).get("degenerate")),
        "by_validator": {
            m.mid: {
                "n": sum(1 for r in raw if r["validator"] == m.mid),
                "unparseable": sum(1 for r in raw if r["validator"] == m.mid
                                   and r.get("status") == "unparseable"),
                "degenerate": sum(1 for r in raw if r["validator"] == m.mid
                                  and r.get("status") == "ok"
                                  and r.get("full", {}).get("degenerate")),
            } for m in models.MODELS
        },
    }

    # Counterfactual denominator, so the direction of the exclusion is visible rather than merely
    # disclosed. A reader who notices that 38% of suites are dropped cannot tell from the counts
    # alone whether dropping them flatters the headline or depresses it. It depresses it: counting
    # every parseable suite as a pass unless it caught something raises the escape rate. Computed
    # exactly as joined() does, minus the degeneracy filter, and it draws no random numbers, so no
    # bootstrap or permutation result can shift because this block exists.
    _orc = load_oracle()
    _extra = witnessed_divergent()
    _esc_d = _pas_d = 0
    for r in raw:
        if r.get("status") != "ok":
            continue
        o = _orc.get((r["translator"], r["task_id"]))
        if o is None or o.get("status") != "ok" or o.get("divergent") is None:
            continue
        f = r.get("full")
        if not f or f["caught"]:
            continue
        _pas_d += 1
        if bool(o["divergent"]) or (r["translator"], r["task_id"]) in _extra:
            _esc_d += 1
    res["suite_health"]["escape_rate_incl_degenerate"] = _esc_d / max(1, _pas_d)
    res["suite_health"]["n_passed_incl_degenerate"] = _pas_d

    # ---- C5: subset consistency between the random baseline and the strong oracle
    # The baseline's inputs come FROM the oracle's input list, so "baseline caught it, oracle
    # called the translation clean" is impossible. Asserts membership rather than relying on a
    # dict default, because the first version of this check reported 43 phantom violations that
    # were only a missing oracle file reading as "clean".
    orc_full = load_oracle()
    checked = viol = 0
    for r in load_parity("random"):
        if r.get("status") != "ok":
            continue
        key = (r["translator"], r["task_id"])
        if key not in orc_full:
            continue
        o = orc_full[key]
        if o.get("status") != "ok" or o.get("divergent") is None:
            continue
        checked += 1
        if r["full"]["caught"] and not o["divergent"]:
            viol += 1
    res["control_c5"] = {"n_checked": checked, "n_violations": viol}

    # ---- would the BENCHMARK's own test inputs have caught these divergences?
    # C2 shows the benchmark's own inputs are 92.7% mutation-adequate against classical
    # single-token mutants, which is close to what 1000 fuzz inputs achieve. If those same inputs
    # nevertheless miss most real translation divergences, then mutation adequacy is the wrong
    # instrument for migration validation, and that is worth stating with a number.
    ref_all = json.loads((D / "reference.json").read_text(encoding="utf-8"))
    elig = json.loads((D / "eligible.json").read_text(encoding="utf-8"))
    # bench.benchmark_seed_prefix_counts is the single definition of this quantity, shared with
    # controls.py's C2 benchmark-kill computation. The two used to disagree: this block always
    # counted the true prefix, but controls.py counted a fixed min(64, len(inputs)) that overcounts
    # for most problems, so \CtwoBenchKill and \BenchInputsMean were printed side by side in the
    # manuscript from two different definitions of "the benchmark's own inputs".
    nseed = bench.benchmark_seed_prefix_counts(ref_all, elig)

    det = miss = 0
    for (mid, tid), o in orc.items():
        if o.get("status") != "ok" or not o.get("divergent"):
            continue
        idx = o.get("diverge_idx") or []
        if any(k < nseed.get(tid, 0) for k in idx):
            det += 1
        else:
            miss += 1
    res["benchmark_suite"] = {
        "n_divergent": det + miss,
        "n_detected_by_benchmark_inputs": det,
        "detection_rate": det / max(1, det + miss),
        "mean_benchmark_inputs_per_problem": sum(nseed.values()) / max(1, len(nseed)),
    }

    # ---- Is suite dropout ASYMMETRIC between the self and cross conditions?
    # An objection worth measuring rather than arguing: if a validator is likelier to
    # emit a degenerate or unparseable suite for a FOREIGN translation, the surviving cells are
    # not comparable between the diagonal and the off-diagonal, and the contrast is confounded by
    # which cells survived rather than by what the survivors show.
    raw_all = load_parity("plain")
    d_self = [r for r in raw_all if r["validator"] == r["translator"]]
    d_cross = [r for r in raw_all if r["validator"] != r["translator"]]

    def _drop(rs):
        if not rs:
            return None
        bad = sum(1 for r in rs
                  if r.get("status") != "ok"
                  or r.get("full", {}).get("degenerate"))
        return {"n": len(rs), "n_dropped": bad, "dropout_rate": bad / len(rs)}

    res["dropout_symmetry"] = {"self": _drop(d_self), "cross": _drop(d_cross)}
    if res["dropout_symmetry"]["self"] and res["dropout_symmetry"]["cross"]:
        res["dropout_symmetry"]["diff_pp"] = 100 * (
            res["dropout_symmetry"]["self"]["dropout_rate"]
            - res["dropout_symmetry"]["cross"]["dropout_rate"])

    # Per-validator breakdown of the same quantity. The pooled diff above is a composition
    # effect, not a per-validator tendency: it mixes validators with very different dropout
    # rates (Gemini 1-2%, Mistral 85-91%) whose self/cross cell counts are also very different
    # (Gemini supplies 136 self cells, Mistral only 65), which is exactly the confound this
    # section exists to check for. Reported so the manuscript states the composition fact rather
    # than a validator behaviour the pooled number does not actually establish.
    per_v = {}
    worse_on_self = 0
    for m in models.MODELS:
        s = _drop([r for r in d_self if r["validator"] == m.mid])
        c = _drop([r for r in d_cross if r["validator"] == m.mid])
        if not s or not c:
            continue
        d = 100 * (s["dropout_rate"] - c["dropout_rate"])
        per_v[m.mid] = {"self": s, "cross": c, "diff_pp": d}
        if d > 0:
            worse_on_self += 1
    res["dropout_symmetry"]["per_validator"] = per_v
    res["dropout_symmetry"]["n_worse_on_self"] = worse_on_self
    res["dropout_symmetry"]["n_validators_checked"] = len(per_v)

    # ---- RQ2 diagonal test on the UNION sample as well.
    # Leave-one-out discards a catch that only the scored validator made, which is what a genuine
    # self-recognition ADVANTAGE would look like; the union keeps those. Reporting the exact test
    # under both settles whether the null depends on that choice.
    rows_union = [dict(r, divergent_loo=r["divergent"]) for r in rows]
    res["rq2"]["permutation_diagonal_union"] = diagonal_permutation_test(rows_union)

    # ---- MATCHED subsample: translations with BOTH a usable self suite and at least one usable
    # cross suite. Differential dropout cannot bias this, because every translation in it
    # contributes to both conditions by construction. Running the exact matrix test here answers
    # the attrition objection and the validator-skill objection at the same time, which neither
    # the pre-registered paired test (matched but skill-confounded) nor the full matrix test
    # (skill-controlled but unmatched) does alone.
    by_tr = defaultdict(list)
    for r in rows:
        if r["divergent_loo"]:
            by_tr[(r["translator"], r["task_id"])].append(r)
    matched_keys = {k for k, rs in by_tr.items()
                    if any(x["self"] for x in rs) and any(not x["self"] for x in rs)}
    matched = [r for r in div if (r["translator"], r["task_id"]) in matched_keys]
    if matched:
        ms = [r for r in matched if r["self"]]
        mc = [r for r in matched if not r["self"]]
        res["matched_subsample"] = {
            "n_translations": len(matched_keys),
            "n_cells": len(matched),
            "self": {"n": len(ms),
                     "miss_rate": sum(1 for r in ms if not r["caught"]) / len(ms)},
            "cross": {"n": len(mc),
                      "miss_rate": sum(1 for r in mc if not r["caught"]) / len(mc)},
            "permutation_diagonal": diagonal_permutation_test(matched),
        }
        res["matched_subsample"]["diff_pp"] = 100 * (
            res["matched_subsample"]["self"]["miss_rate"]
            - res["matched_subsample"]["cross"]["miss_rate"])

    # ---- Two-way bootstrap: resample PROBLEMS and MODELS.
    # The problem-clustered interval treats the five models as fixed, so it describes this
    # ensemble rather than models in general. Resampling the model set as well widens it to
    # include between-model variance, which is large here (escape rate by validator spans roughly
    # 15% to 51%). With five models the model dimension is coarse, so this is a rough upper bound
    # on uncertainty rather than a precise interval, and we say so.
    passed_rows = [r for r in rows if passed(r)]
    if passed_rows:
        ids = [m.mid for m in models.MODELS]
        by_pm = defaultdict(list)
        for r in passed_rows:
            by_pm[(r["task_id"], r["validator"], r["translator"])].append(int(esc(r)))
        probs_all = sorted({r["task_id"] for r in passed_rows})
        reps = []
        for _ in range(N_BOOT):
            pr_s = [probs_all[rng.randrange(len(probs_all))] for _ in probs_all]
            md_s = [ids[rng.randrange(len(ids))] for _ in ids]
            num = den = 0
            for p_ in pr_s:
                for v_ in md_s:
                    for t_ in md_s:
                        for e in by_pm.get((p_, v_, t_), ()):
                            num += e
                            den += 1
            if den:
                reps.append(num / den)
        reps.sort()
        if reps:
            res["rq1_twoway_ci"] = [reps[int(0.025 * len(reps))],
                                    reps[min(len(reps) - 1, int(0.975 * len(reps)))]]

    # ---- Leave-one-MODEL-out jackknife on the headline.
    # The cluster bootstrap clusters on problems, which treats the five models as fixed. With
    # only five, one model could carry the result. Dropping each in turn, as translator AND as
    # validator, bounds that directly.
    jack = {}
    for m in models.MODELS:
        sub = [r for r in rows if r["validator"] != m.mid and r["translator"] != m.mid]
        n, d, pt = rate(sub, esc, passed)
        if d:
            jack[m.mid] = {"n_passed": d, "escape_rate": pt}
    if jack:
        vals = [v["escape_rate"] for v in jack.values()]
        res["jackknife_escape"] = {"by_dropped_model": jack,
                                   "min": min(vals), "max": max(vals)}

    # ---- The translation funnel, end to end.
    # The paper reported "510 translations were executable" with no denominator, so a reader could
    # not see that 680 were attempted and 141 never compiled. A coverage audit flagged it: 21% of
    # all attempts were invisible. Every stage is now counted here so the funnel is reportable.
    n_attempted = n_compiled = 0
    for m in models.MODELS:
        tp = D / "translations" / f"{m.mid}.json"
        if tp.exists():
            ts = json.loads(tp.read_text(encoding="utf-8"))
            n_attempted += len(ts)
            n_compiled += sum(1 for t in ts if t.get("compile_ok"))
    st = Counter(o.get("status") for o in orc.values())
    res["translation_funnel"] = {
        "attempted": n_attempted,
        "compiled": n_compiled,
        "compile_failures": n_attempted - n_compiled,
        "no_comparable_inputs": st.get("no_comparable", 0),
        "usable": st.get("ok", 0),
    }

    # ---- controls
    for tag, f in (("c2", "control_c2.json"), ("c3", "control_c3.json"),
                   ("c3surv", "control_c3_survivors.json")):
        p = D / f
        if not p.exists():
            continue
        j = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(j, list):
            # the survivor investigation is a list of per-survivor verdicts
            res[f"control_{tag}"] = {
                "n": len(j),
                "n_likely_equivalent": sum(1 for x in j if "EQUIVALENT" in x.get("verdict", "")),
                "n_genuine_miss": sum(1 for x in j if "GENUINE" in x.get("verdict", "")),
                "extra_inputs_each": max((x.get("extra_inputs_tried", 0) for x in j), default=0),
            }
        else:
            j.pop("rows", None)
            res[f"control_{tag}"] = j

    (D / "results.json").write_text(json.dumps(res, indent=1), encoding="utf-8")

    # ---------------------------------------------------------------- console summary
    print("=" * 74)
    print(f"cells (validator x translator x problem, scorable): {res['n_cells']}")
    r1 = res["rq1"]
    print(f"\nRQ1  escape rate = {100 * r1['escape_rate']:.1f}%  "
          f"[{100 * r1['ci'][0]:.1f}, {100 * r1['ci'][1]:.1f}]  "
          f"({r1['n_escaped']}/{r1['n_passed']} parity-passing translations were divergent)")
    r2 = res["rq2"]
    print(f"\nRQ2  miss rate on DIVERGENT translations")
    print(f"     self  {100 * r2['self']['miss_rate']:.1f}%  "
          f"[{100 * r2['self']['ci'][0]:.1f}, {100 * r2['self']['ci'][1]:.1f}]  "
          f"(n={r2['self']['n']})")
    print(f"     cross {100 * r2['cross']['miss_rate']:.1f}%  "
          f"[{100 * r2['cross']['ci'][0]:.1f}, {100 * r2['cross']['ci'][1]:.1f}]  "
          f"(n={r2['cross']['n']})")
    pm = r2.get("permutation_preregistered", {})
    if pm.get("n_groups"):
        print(f"     pre-registered paired test (CONFOUNDED, reported for transparency): "
              f"self {100 * pm['self_miss_rate']:.1f}% vs cross "
              f"{100 * pm['cross_miss_rate']:.1f}%, diff {pm['diff_pp']:+.1f}pp, "
              f"p={pm['p_value']:.4f}, n={pm['n_groups']}")
    dx = r2.get("permutation_diagonal_exact", {})
    if dx.get("n_permutations"):
        print(f"     EXACT diagonal test ({dx['n_permutations']} permutations of the "
              f"{dx['n_models']}-model matrix, margins fixed):")
        print(f"       diagonal {100 * dx['diagonal_miss_rate']:.1f}%  "
              f"off-diagonal {100 * dx['offdiagonal_miss_rate']:.1f}%  "
              f"mean pseudo-diagonal {100 * dx['mean_pseudodiagonal']:.1f}%  "
              f"p={dx['p_value_one_sided_diag_worse']:.4f}")
    wv = res.get("within_validator", {})
    if wv:
        print("\n     within-validator (own translations vs others'):")
        for mid, v in wv.items():
            print(f"       {mid:9s} own {100 * v['miss_own']:5.1f}% (n={v['n_own']:3d})  "
                  f"other {100 * v['miss_other']:5.1f}% (n={v['n_other']:3d})  "
                  f"{v['diff_pp']:+.1f}pp")
    for k in ("baseline_random", "complementarity", "oracle_supplemented", "benchmark_suite"):
        if k in res:
            print(f"\n{k}:", json.dumps(res[k], indent=None)[:300])
    print("\nsuite health:", res["suite_health"])
    for t in ("control_c2", "control_c3"):
        if t in res:
            print(f"{t}:", res[t])
    print("\nwrote", D / "results.json")


if __name__ == "__main__":
    main()
