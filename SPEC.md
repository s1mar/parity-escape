# FROZEN PRE-REGISTRATION SPEC

Frozen before any translation or validation call was made. Every method threshold in
the manuscript must appear here. `code/gates.py` reads this file and fails the build if the
declared value of K or N_fuzz disagrees with the code, or if the amendment count stated in the
manuscript disagrees with the number of entries below. Amendments go in the AMENDMENTS section at
the bottom, are numbered, and are counted.

*Corrected: this paragraph previously attributed the enforcement to a
`code/crosscheck.py` that was never written for this study, and claimed a broader check than the
one that runs. The sentence above describes what `code/gates.py` actually does. No design
decision, threshold or measurement is affected, which is why this is a correction to the document
and not a numbered amendment: the amendment log records changes to the study, and padding it with
a documentation fix would devalue the ten entries that are real.*

## Research questions

- **RQ1 (escape rate).** Among LLM Python-to-Java translations that PASS an LLM-generated parity
  test suite whose expected values are ground truth (obtained by executing the source program),
  what fraction are nevertheless semantically divergent from the source?
- **RQ2 (correlated blindness).** Is the escape rate higher when the model that generated the
  parity inputs is the SAME model that produced the translation (self-validation, matrix diagonal)
  than when it is a different model (cross-validation, off-diagonal)?
- **RQ3 (where).** Which semantic categories do the escaping divergences fall into, and is the
  distribution different on the diagonal than off it?
- **RQ4 (mitigation).** Do (a) cross-family validation and (b) divergence-targeted prompting
  reduce the escape rate, and by how much?

## Pre-registered predictions

Recorded now so a failed prediction cannot be retro-fitted. Both directions are publishable and
the paper is written to report whichever occurs.

- **P1.** Escape rate under self-validation is strictly greater than zero. (If it is zero the
  paper has no result and will be withdrawn rather than reframed.)
- **P2.** Escape rate on the diagonal exceeds the off-diagonal escape rate.
  **If P2 fails**, the reported finding becomes "validator blindness is a property of the language
  pair, not of model identity", which is stated in the paper as the alternative outcome, not as
  the hypothesis I held. The abstract is written AFTER this number is known.
- **P3.** Escaping divergences concentrate in a small number of cross-language semantic categories
  (integer width/overflow, integer division and rounding, floating-point formatting, string and
  character handling, collection ordering and stability, empty/boundary inputs) rather than being
  uniformly spread over program logic.

## Corpus

- Source: HumanEvalPack `python` split, 164 problems, loaded OFFLINE from the local HuggingFace
  cache. The `canonical_solution` field prefixed by `prompt` is the legacy source program.
- A problem is ELIGIBLE if and only if all of:
  1. its signature type hints parse into the supported type set (below);
  2. the Python source executes without exception on at least 100 of the 1000 generated inputs;
  3. the reference-vs-reference positive control shows zero divergence.
- Supported type set (declared in advance, not tuned to results):
  `int, float, bool, str, List[int], List[float], List[str], List[bool], List[List[int]],
  Tuple[...], Dict[str,int], Optional[...]` for parameters; the same set plus `None` for returns.
- Ineligible problems are EXCLUDED and the count and reasons are reported in the paper. The
  eligible count is whatever it is; no target count is set, because setting one would invite
  tuning the type set until the count is reached.

## Models

Translator set == validator set, giving a square matrix. The four local weights are pinned by
digest below, captured from the running Ollama daemon after the run and confirmed unchanged
since (`ollama list` / `GET /api/tags`); the hosted arm has no local weight to digest.

| id | route | family | quantisation | digest (sha256, first 16) |
|---|---|---|---|---|
| `qwen2.5:7b-instruct` | Ollama, local | Qwen | Q4_K_M | `845dbda0ea48ed74` |
| `deepseek-coder:6.7b-instruct` | Ollama, local | DeepSeek | Q4_0 | `ce298d984115b93b` |
| `mistral:7b-instruct-v0.2-q4_0` | Ollama, local | Mistral | Q4_0 | `61e88e884507ba5e` |
| `llama3:latest` (8B) | Ollama, local | Llama | Q4_0 | `365c0bd3c000a25d` |
| `gemini-3.6-flash-high` | agy CLI | Gemini (frontier arm) | n/a, hosted | n/a, hosted |

Decoding: temperature 0.2, top_p 0.95, seed 0, max 1024 new tokens, ONE sample per cell. Fixed
before the run. Single-sample is a stated limitation, not a hidden one.

## Translation task

The translator receives the Python source and an EXACT Java method signature derived
mechanically from the Python type hints, and must implement `public class Solution` with that
static method. Fixing the signature removes signature ambiguity as a confound and makes
marshalling deterministic. A translation is USABLE if it compiles with `javac 17` and returns a
parseable result on at least one input; unusable translations are excluded from escape-rate
denominators and counted separately (a translation that does not compile is a visible failure,
not a silent one, and is not what this paper is about).

## Parity validation (the thing under test)

- The validator model receives the Python source AND the candidate Java translation, and is asked
  for **K = 10 test INPUTS ONLY**, as JSON argument tuples. It never supplies expected values.
- Expected values are computed by EXECUTING THE PYTHON SOURCE on those inputs. This is the
  ground-truth oracle and is the industrial record-and-replay parity setup.
- Inputs on which the Python source raises are OUT OF DOMAIN and discarded before scoring; a
  validator that returns fewer than 3 in-domain inputs is recorded as a DEGENERATE suite and
  reported separately rather than silently counted as a pass.
- The translation PASSES parity validation if the Java output equals the Python output on every
  in-domain input.
- Sensitivity to K is reported at K in {3, 5, 10} by taking prefixes of the same generated list,
  so no extra generation is needed and the comparison is within-suite.

## Strong oracle (ground truth for divergence)

- **N_fuzz = 1000** type-directed inputs per problem, generated from the signature by a fixed
  generator with seed 0. Composition fixed in advance: 30% boundary values, 70% random.
  Boundary catalogue: `0, 1, -1, 2, -2, 2**31-1, -2**31, 2**63-1, 10**18` for ints; `0.0, -0.0,
  1e-9, -1e-9, 1e18, 0.5, -0.5` for floats; `"", " ", "a", "A", "0", unicode, 200-char` for
  strings; `[], [x], duplicates, sorted, reverse-sorted, length-100` for lists.
- A translation is **DIVERGENT** if there exists at least one in-domain input on which the Java
  output differs from the Python output.
- Comparison rules, fixed in advance: integers exact; floats equal if
  `abs(a-b) <= 1e-6 * max(1, abs(a), abs(b))`; strings exact; lists elementwise in order;
  a Java exception where Python returned a value is a divergence; both raising is not.
- The **escape set** = translations that PASS parity validation but are DIVERGENT.
- **Escape rate** = |escape set| / |translations that passed parity validation|. Reported per
  cell, and pooled with a cluster bootstrap over problems (10,000 resamples, percentile 95% CI),
  because translations of the same problem are not independent.

## Controls (both sides required; a one-sided control is not a control)

- **C1 positive / identity.** Python source vs itself through the full marshalling harness.
  MUST report zero divergence on every eligible problem. Any nonzero result means the harness,
  not the translation, is divergent, and the run is invalid.
- **C2 negative / mutation.** Seeded single-token mutants of the Python source (comparison
  operator swap, off-by-one on a constant, boundary inequality flip, arithmetic operator swap),
  marshalled through the same harness. The strong oracle MUST detect at least **90%** of mutants
  that are not semantically equivalent. This measures the oracle's own power and is reported as a
  number, not asserted. If it falls below 90%, N_fuzz is raised and every downstream number is
  recomputed, and that would be recorded as a numbered amendment like any other change.
- **C3 injection.** For a random sample of 20 translations that passed BOTH parity and the strong
  oracle, hand-inject one known divergence each; the strong oracle must flag all 20. A harness
  that says "equivalent" to everything is exactly the harness this paper warns about.
- **C4 determinism.** Every eligible problem's Python source is executed twice on the same inputs;
  any problem whose output differs between runs is nondeterministic and is EXCLUDED, because a
  nondeterministic source makes parity meaningless.

## Statistics

- Diagonal vs off-diagonal escape rate: two-sided permutation test, 10,000 permutations,
  permuting the validator label WITHIN translator (so each translation keeps its own difficulty),
  alpha = 0.05. Effect size reported as a rate difference in percentage points with a bootstrap CI.
- No other comparison is significance-tested; all other numbers are descriptive. This is declared
  now to stop a fishing expedition later.

## Amendments

**The body above is FROZEN and is not edited after the fact; where an amendment
below contradicts it, the amendment is authoritative.** That is the point of freezing it: a
reader can see both what was planned and what changed. In particular the integer boundary
catalogue quoted in the Strong Oracle section is the ORIGINAL one, superseded by Amendments 2
and 3.

Any change to anything above is appended here, numbered, dated, with the reason, and the
manuscript's stated amendment count must match the number of entries. Current count: **10**.

**Amendment 1 | Resource exhaustion is excluded from the domain on both sides, and
both children run under a kernel-enforced 2 GB memory cap with a 1 MB cap on any single
serialised result.**
Reason: the first reference run reached 41 GB of resident memory before it was killed. The cause
is structural rather than incidental: several fuzz inputs are integers that a function uses as an
allocation size, so an input of 10^18 asks for a list of 10^18 elements. A per-call timeout does
not catch it, because one huge allocation is fast.
What changed: (a) a Windows Job Object caps each child at 2 GB; (b) any single serialised result
above 1 MB is reported as `Oversize`; (c) `Oversize, MemoryError, OutOfMemoryError,
StackOverflowError, RecursionError, MemoryAbort, NegativeArraySizeException` join the existing
timeout names in the OUT-OF-DOMAIN set on BOTH sides.
Direction of the effect on the headline number: excluding these can only REMOVE candidate
divergences, never add one, so the escape rate reported under this amendment is a lower bound on
the escape rate that would be measured without it. That is the safe direction for the claim, and
it is the reason the amendment was accepted rather than the input catalogue being trimmed:
trimming the large integers would have removed the overflow probes that the study is about.
Nothing about model selection, prompts, K, N_fuzz, or the statistics changed, and no translation
or validation call had been made when this amendment was written.

**Amendment 2 | Amendment 1's memory cap DOES NOT WORK in this environment. The
integer boundary catalogue is split by argument position instead.**
Correction first: Amendment 1 asserted a kernel-enforced 2 GB cap. It was tested directly after
the second reference run also blew up (21 GB), and it does not hold: the job object is created
successfully but `AssignProcessToJobObject` does not take effect, almost certainly because the
parent process is already inside a job object in this sandbox. A child asked to allocate 2 GB
under a 512 MB cap allocated it and exited 0. The code is retained as best-effort defence in
depth and is explicitly NOT relied upon. Leaving Amendment 1 as written would have put a false
statement about the method in the frozen spec.
What replaces it: `INT_BOUNDARIES` becomes two catalogues selected by argument position.
`SCALAR_INT_BOUNDARIES` (top-level arguments) tops out at 10^6. `ELEM_INT_BOUNDARIES` (values
inside a list) additionally carries 2^31-1, -2^31, 2^31, 2^62, 2^63-1, -2^63 and 10^18.
Why this is not a loss of probing power: the blowups were all driven by a scalar argument used
as an allocation size or loop bound (`string_sequence(10**18)`). Java `long` wraps only near
2^63, and a scalar argument that large makes the source time out before it computes anything,
so it never yielded an overflow observation in the first place. Large values inside a bounded
list still reach the wrap point through the function's own arithmetic while both programs
terminate, which is where an overflow divergence is actually observable.
The out-of-domain exclusion set from Amendment 1 stands unchanged and is still correct.
No translation or validation call had been made when this amendment was written.

**Amendment 3 | Scalar integer magnitudes are further reduced: the scalar boundary
catalogue tops out at 10^4 (was 10^6) and random scalar draws top out at 10^4 (was 10^6). List
elements are unchanged and keep the full wide catalogue.**
Reason: runtime, measured rather than guessed. With 20% of scalar draws coming from +/-10^6, a
single problem (`string_sequence`) spent roughly 400 seconds building multi-megabyte strings that
were then discarded as oversize, and every problem with an integer argument paid a similar cost.
Projected over 147 problems this was hours of wall clock for zero additional discrimination.
Why it costs no power: a Java `long` cannot be made to overflow by the magnitude of a scalar
argument on its own; overflow becomes observable when the function's own arithmetic combines
values, which the list-element catalogue (unchanged, up to 2^63-1) still probes. Scalar magnitude
was buying allocation, not semantics.
Recorded rather than silently tuned because this is a fuzzer parameter that touches every
measurement, and because "I made the fuzzer weaker and the run got faster" is exactly the kind of
change that should be visible to a reviewer. No translation or validation call had been made when
this amendment was written.

**Amendment 4 | A wall-clock budget of 120 seconds bounds the execution of one
program over one input set. Inputs not reached are marked `Budget` and are out of domain.**
Reason: a per-call timeout cannot catch a program that is merely SLOW. HumanEval's
`is_multiply_prime` reference runs a triple loop with an inner primality test, roughly 10^6 inner
iterations per call, so every call completes just inside the 2 second per-call limit and 1000 of
them take about half an hour, twice over for the determinism control. The reference build sat on
that single problem for more than four minutes before it was killed, and several later problems
have the same shape.
What changed: `_run_with_restarts` now tracks elapsed wall clock and stops spawning once the
budget is spent, marking every unreached input `Budget`, which joins the out-of-domain set.
Effect on the numbers: a truncated problem simply has a smaller comparable input set, or falls
below the 100-input in-domain floor and is dropped from the corpus entirely. Both are visible in
the reported corpus counts. It cannot manufacture a divergence, because an input that was never
executed cannot disagree; it can only remove evidence, so escape rates remain lower bounds.
Recorded rather than silently applied because a budget that truncates evidence is exactly the
kind of engineering convenience that could quietly change a result. No translation or validation
call had been made when this amendment was written.

**Amendment 5 | An input on which the SOURCE returns a value outside its declared
return type is OUT OF DOMAIN.**
Reason, found by inspecting the first real divergences rather than by theory: the Java signature
is derived from types OBSERVED on the benchmark's own test inputs, so a fuzz input can take the
source somewhere those observations never went. `largest_divisor` returns an int on every
benchmark input and `None` on a negative one; `find_closest_elements` returns a list of floats
normally and `None` on a list shorter than two. No Java method returning a primitive `long` can
reproduce `None`, so on those inputs the translation is forced to differ whatever the model
writes.
Scale: 633 of 128,488 in-domain source outputs, confined to exactly 2 of 136 problems. Small in
total but concentrated: those two problems would have shown near-total divergence for EVERY
model, so the artifact would have inflated the divergence rate systematically and identically
across arms, which is the hardest kind of bias to notice from the aggregate.
Direction: the rule can only REMOVE divergences, never add one, so escape rates remain lower
bounds. Applied identically in the oracle, in parity scoring, and in the C2 mutation control, so
one domain definition holds everywhere.
Status when written: the Gemini arm had been evaluated and was recomputed from scratch under the
new rule. No parity validation had been run, so no measurement of the paper's actual question was
in hand when this was decided.

**Amendment 6 | Scoring one parity suite gets a 15 second wall-clock budget per
side. Inputs not reached are out of domain.**
Reason: throughput, measured. The first local validator ran at 1.5 cells per minute against a
generation cost of only 4.6 seconds, so roughly 35 seconds per cell was EXECUTION. The cause is
that a parity suite deliberately picks adversarial inputs, and the weaker translations hang on
several of them; each hang costs a 2 second per-call timeout plus a worker restart. At 2,064
local cells that projected to about 23 hours.
Why it cannot bias RQ2: the bound depends only on how the TRANSLATION behaves, never on which
validator supplied the inputs. Every validator scoring the same translation is subject to the
same bound, so the paired self-versus-cross contrast is untouched.
Effect on RQ1: it can only lose evidence, never manufacture it, so escape rates stay lower
bounds. A cell that exhausts the budget has fewer in-domain inputs and, if it drops below three,
is recorded as degenerate and reported rather than counted as a pass.
Status when written: 155 parity suites had been scored (143 Gemini, 12 Qwen) and NO analysis had
been run on them. The change was made for cost, not after seeing an unwelcome number.

**Amendment 7 | The pre-registered RQ2 significance test is CONFOUNDED. It is still
reported, and the primary test becomes an exact column-permutation test on the validator x
translator matrix.**
The pre-registered test permuted the "self" label among the five validators scoring one
translation. Its null is that validators are exchangeable. They are not, and the inequality is
enormous: on partial data the Gemini validator had caught 96 of 157 cells and the Qwen validator
4 of 53. Because the self validator is fixed by the translator, "self" cells and "cross" cells
differ in WHICH validators populate them, so the contrast mixes validator skill and translator
difficulty into the effect of interest. This is a defect in my pre-registration, not in the data.
Replacement, which is exact rather than sampled: the matrix holds one miss rate per (validator,
translator) cell, and self-validation is its diagonal. For each permutation pi of the model set,
read the pseudo-diagonal {(V, pi(V))} and compute its pooled miss rate. Permuting columns
preserves every row and column marginal exactly, so validator skill and translator difficulty are
held fixed by construction and the only thing varying is which cells are called diagonal. With
five models all 120 permutations are enumerated, giving an exact p-value with a floor near 0.008.
Also added, as the most interpretable view: a per-validator breakdown of each validator's miss
rate on its OWN translations versus on other models' translations, which holds validator skill
fixed by construction.
BOTH tests are reported, and the paper states plainly that the pre-registered one is confounded
and why. Deleting it and presenting only the corrected test would hide a real methodological
error. Status when written: no RQ2 number of any kind had been computed.

**Amendment 8 | The parity-suite parser becomes TYPE-AWARE, and all parity data
generated before this change is discarded and regenerated.**
The JSON format is genuinely ambiguous for single-argument functions. For a function taking one
`list[int]`, a model answering `[[0], [1], [-1]]` may mean three calls whose single argument is
the scalar 0, or three calls whose single argument is the LIST [0]. The original parser took the
first reading. For a list-typed parameter that makes the source raise a TypeError on every input,
so the suite scored as DEGENERATE, and 52 of Qwen's first 73 suites landed there with a median of
one usable input while Gemini's median was ten.
Why this had to change: read that way, the headline contrast between validators is substantially
a contrast in JSON FORMAT COMPLIANCE, not in the power to choose revealing inputs. That is a
different paper, and it is precisely the confound this design is meant to avoid.
What the fix does: every plausible reading of the response is enumerated and scored against the
DECLARED parameter types, and the best-typechecking reading wins. It resolves formatting only. It
never changes which values the model chose and cannot turn a poorly chosen input into a revealing
one. Verified on the exact ambiguous case in both directions, plus multi-argument, flat-array,
correctly-nested and unparseable inputs.
Cost and honesty: all previously generated parity suites (160 Gemini, 80 Qwen) were produced
under the old parser and are DISCARDED rather than mixed with new data; they are kept in
`data/_discarded_oldparser/` so the change is auditable. Suites are now stored with their raw
response text, so any future parser change can be applied by re-parsing instead of re-generating.
Format compliance is still reported, separately, as its own result.
Status when written: no RQ1 or RQ2 number had been computed from any parity data.

**Amendment 9 | A malformed JSON array is salvaged element by element instead of
being discarded, and a `reparse` stage re-scores from stored raw responses.**
Reason, again found by reading raw responses rather than totals: small models routinely emit Java
where JSON was asked for, INSIDE an otherwise well-formed list. Observed verbatim from Qwen:
`[Math.PI]`, `[Double.MAX_VALUE]`, `[1L, Long.MIN_VALUE, Long.MAX_VALUE]`, and one response with
a `)` where a `]` belonged. A single such element makes `json.loads` reject the entire array,
after which the scanner falls back to the largest valid fragment it can find: a model that
proposed eight inputs was being scored on one, and its suite was then recorded as degenerate.
The fix parses the array element by element and keeps the elements that are valid JSON. It cannot
invent an input and cannot improve a badly chosen one; it only stops one malformed element from
destroying the seven good ones beside it. Verified on all four observed failure shapes plus clean
multi-argument input: recovery went from 1 usable input to 4 on two of them.
Also added: `reparse.py`, which re-extracts inputs from STORED RAW RESPONSES and re-scores only
the records whose parse actually changed. The first parser change forced a full regeneration
purely because only parsed results had been kept; that will not happen again.
Format compliance remains a reported result in its own right, separately from detection power,
because the paper's question is which inputs a model chooses, not whether it can emit JSON.
Status when written: no RQ1 or RQ2 number had been computed from any parity data.

**Amendment 10 | Type conformance means REPRESENTABLE IN THE DECLARED JAVA TYPE, not
merely "is a Python integer".**
Found during review of the draft; it is the one finding that changed a number.
Amendment 5 excluded inputs where the source returns a value outside its declared return TYPE,
which caught `None` returns. It did not catch magnitude. Python integers are unbounded and the
declared Java return type is `long`, so a source that legitimately computes a value beyond
2^63-1 forces the translation to differ whatever the model writes. We fixed the signature, so we
made that outcome unavoidable, and counting it as a semantic divergence measures our harness
rather than the migration.
Scale: 2,098 of 60,819 integer-returning in-domain source outputs, spread over 25 of 136
problems, including `sum_squares`, `order_by_points` and `f`.
What is NOT excluded, and should not be: a source whose RESULT is in range while the Java
computation overflows on the way there. `median([MAX, MIN])` returns -0.5, which a `long` pair
and a double can both represent; the translation reaches 0.0 only because it summed first. A
careful translation avoids that, so it is a real defect and stays counted.
Direction: strictly removes divergences, so every rate remains a lower bound. Requires re-running
the oracle for all five arms and force-rescoring every parity cell, since per-input outputs are
not stored and the verdicts cannot be recomputed without re-executing.
Status when written: RQ1 and RQ2 had been computed and are being recomputed under this rule. The
change was made because the artifact is real, not because of where it moves the number, and the
direction it moves it is against the paper's headline.
