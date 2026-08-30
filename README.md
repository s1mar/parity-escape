# Replication package: Passing Parity Is Not Preserving Behaviour

Everything needed to reproduce every number in the paper, on one workstation, with no paid API
and no special access. The models are four locally hosted open checkpoints plus one hosted
frontier model reached through a free CLI.

## What is here, and why

| path | what it is |
|---|---|
| `SPEC.md` | The **frozen pre-registration**, written before any model was called, with every amendment numbered, dated, justified, and annotated with the direction of its effect on the headline number. |
| `code/` | The pipeline. Each stage is resumable; rerunning skips completed work. |
| `data/` | All generated artifacts: translations, parity suites, oracle verdicts, control results, and `results.json`, from which the paper's macros are generated. |
| `paper/` | `main.tex`, `refs.bib`, the built `main.pdf`, and three generated files never edited by hand: `numbers.tex` (macro values), `tables.tex` (table bodies) and `verify_output.txt` (the ground-truth dump to reconcile the paper against). |

## Pipeline

Stages run in order. `PY` is the Python 3.13 interpreter.

```
python code/probe.py           $PY   # observe concrete argument/return types by executing tests
python code/build_reference.py $PY   # fuzz inputs + ground-truth source outputs (controls C1,C4)
python code/translate.py       $PY   # one Java translation per (model, problem)
python code/repair_imports.py  $PY   # one deterministic retry for import-only compile failures
python code/evaluate.py        $PY   # the oracle: differential execution, ground-truth divergence
python code/validate.py        $PY                      # the 5x5 parity-suite matrix
python code/validate.py        $PY --mode random        # non-LLM baseline, no model calls
python code/validate.py        $PY --mode targeted --self-only   # RQ4 mitigation arm
python code/reparse.py         $PY --force              # re-score suites from stored raw responses
python code/controls.py        $PY --c2 --c3            # oracle power: mutation and injection
python code/c3_survivors.py    $PY   # are the C3 survivors equivalent mutants, or real misses?
python code/analyze.py                                  # every reported number -> data/results.json
python code/regression.py                               # RQ2 PRIMARY test -> data/regression.json
python code/taxonomy.py --audit 20                      # RQ3 categories, with a hand-audit sample
python code/make_macros.py                              # results.json -> numbers.tex + tables.tex
python code/make_figures.py                             # results.json -> paper/fig_matrix.pdf
python code/make_verify.py     $PY                      # -> paper/verify_output.txt
python code/consistency.py                              # this paper's claim gate (gates + traceability)
python code/gates.py --selftest                         # validate the gate by re-injecting defects
python code/package_artifact.py                         # rebuild this bundle
```

`regression.py` is not optional. It produces the odds ratio the paper reports as the **primary**
RQ2 result; the permutation tests in `analyze.py` are reported alongside it, not instead of it.
An earlier version of this list omitted it, along with `repair_imports.py`, `reparse.py`,
`make_verify.py` and `c3_survivors.py`, so a reader following it could not have reproduced the
headline RQ2 number.

## The two checks worth running first

`python code/selftest_marshal.py $PY` round-trips every supported type through the full
Python/JSON/Java marshalling harness. If it does not report an exact round-trip for all eleven
type tags, no divergence reported downstream can be trusted, because a marshalling bug and a
translation bug are indistinguishable at the harness boundary.

`python code/gates.py --selftest` re-injects twelve historical defects (among them an invented
venue city, a double-blind claim at a single-blind venue, a hardcoded measurement, a retired
claim, an em-dash, a citation with no bibliography entry, and a macro silently corrupted into
prose) and requires the gate to catch each. A gate written in the same session as the change it
enforces has not been validated.

## Reading the numbers

`data/results.json` is the single source of truth. `paper/numbers.tex` is generated from it, and
`code/gates.py` fails the build if a numeral appears in the manuscript without arriving through a
macro. There is exactly one path from measurement to rendered page.

Rates in the paper are **lower bounds** by construction. Resource exhaustion (timeouts,
out-of-memory, oversize results, unreached inputs) is excluded on both sides, which can only
remove candidate divergences, never add one.
