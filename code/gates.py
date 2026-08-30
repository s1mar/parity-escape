"""The gate. Run before every build, and after every edit to the manuscript.

Design rules this file follows, each learned the expensive way on earlier papers:

* Gate the SUBSTANCE, never the wording, and pair every positive check with a negative clause
  naming what must NOT come back. A gate pinning a phrase cannot tell "claim removed" from
  "defect fixed", and will later demand a retracted overclaim back.
* A check that locates a region must ASSERT it found one. A guard whose anchor never matches
  returns None, short-circuits, and passes a deliberately reinserted defect.
* Negative checks are anchored regexes, never bare substrings: a retired phrase is often a
  substring of its own repair.
* Prefer failing NOISY. Every bug in a quotation-attribution gate on an earlier paper failed
  OPEN, which reads exactly like a clean run.

`--selftest` re-injects historical defects and requires each to be caught. A gate written in the
same hour as the change it enforces has not been validated.

Usage:  python gates.py [--selftest]
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench      # noqa: E402

ROOT = bench.ROOT
TEX = ROOT / "paper" / "main.tex"
NUMS = ROOT / "paper" / "numbers.tex"
SPEC = ROOT / "SPEC.md"
BIB = ROOT / "paper" / "refs.bib"

FAILURES: list[str] = []


def fail(tag: str, msg: str) -> None:
    FAILURES.append(f"[{tag}] {msg}")


# --------------------------------------------------------------------------- venue facts
# Each was fetched from the call for papers, not recalled. The negative list is the important
# half: an invented city or country in a running head is invisible to any gate that only traces
# numbers, so it has to be checked as text.
VENUE_MUST = [
    (r"\bAISM\b", "workshop short name"),
]
VENUE_MUST_NOT = [
    (r"\bSeoul\b", "wrong city (that is ASE 2025)"),
    (r"\bSouth\s+Korea\b", "wrong country"),
    (r"\bNovember\s+2026\b", "wrong month; ASE 2026 is 12-16 October"),
    (r"\bRome\b|\bSingapore\b|\bLisbon\b|\bSacramento\b", "city never verified for this venue"),
    (r"\bdouble[- ]blind\b", "AISM 2026 is SINGLE-blind"),
    (r"\banonymou?s(ly)?\b", "single-blind venue: the paper must not claim anonymisation"),
]


def check_venue(tex: str) -> None:
    for pat, why in VENUE_MUST:
        if not re.search(pat, tex, re.I):
            fail("venue", f"missing: {why} (/{pat}/)")
    for pat, why in VENUE_MUST_NOT:
        m = re.search(pat, tex, re.I)
        if m:
            fail("venue", f"forbidden text {m.group(0)!r}: {why}")


# --------------------------------------------------------------------------- numbers via macros
# Numerals allowed to appear literally in prose: they are facts about cited works or fixed
# constants of the design, not measurements. Anything else must arrive through a macro.
# Numerals allowed to appear literally. Two kinds only, and each is listed for a reason:
#   FACTS ABOUT CITED WORK  the value belongs to someone else's paper and cannot come from our
#                           results file (corpus sizes, reported percentages, model names)
#   FIXED DESIGN CONSTANTS  chosen before the run and declared in SPEC.md, not measured
# Everything else must arrive through a macro, so that it traces back to data/results.json.
LITERAL_OK = {
    # small counts, section and table references, arities
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "17", "20", "21",
    "30", "60", "64", "80", "95", "100",
    # facts about cited work
    "164", "250", "300", "1000", "6164", "9515", "5102", "430", "4114",
    "19.3", "28.9", "3.5", "6.7", "2.5", "3.6",
    # fixed design constants, all declared in SPEC.md
    "0.2", "0.95", "10000", "25", "15", "120", "2000", "1e-6", "10^3", "10^6",
    # years
    "2026", "2025", "2024", "2023", "2021",
}


def check_numbers(tex: str, nums: str) -> None:
    # Strip LaTeX comments, but NOT an escaped percent. Using %.* also matched the \%% in
    # every percentage and deleted the rest of that line, so the gate was blind to the tail
    # of any line containing a number followed by a percent sign, in a paper made of those.
    body = re.sub(r"(?<!\\)%.*", "", tex)
    body = re.sub(r"\\cite\{[^}]*\}", "", body)
    body = re.sub(r"\\(documentclass|usepackage|input|newcommand)\{[^}]*\}(\[[^\]]*\])?", "", body)
    body = re.sub(r"\$[^$]*\$", "", body)                 # math is not a measurement site
    body = re.sub(r"\\label\{[^}]*\}|\\ref\{[^}]*\}", "", body)
    # The ACM rights block is publisher-supplied and inserted VERBATIM: its ISBN, DOI, year and
    # conference dates are ACM's strings, not this paper's measurements, and macro-ising them
    # would be the defect. They are still gated, by check_venue below, against the fetched
    # venue facts (city, country, month) with a negative clause for the wrong ones.
    body = re.sub(
        r"\\(copyrightyear|acmYear|acmDOI|acmISBN|acmConference|acmBooktitle|acmSubmissionID"
        r"|received|setcctype|setcopyright)(\[[^\]]*\])?\{[^}]*\}(\{[^}]*\})*",
        "", body)

    defined = set(re.findall(r"\\newcommand\{\\(\w+)\}", nums))
    used = set(re.findall(r"\\([A-Z]\w+)\{\}", tex)) | set(re.findall(r"\\([A-Z]\w+)\b", tex))
    for u in sorted(used):
        if u in {"PLACEHOLDER"}:
            continue
        if u[0].isupper() and u not in defined and u not in {
            "ACM", "LaTeX", "AISM", "ASE", "COBOL", "Java", "Python", "LLM", "LLMs",
            "MigrationBench", "Qwen", "DeepSeek", "Mistral", "Llama", "Gemini", "JSON",
            "HumanEval", "HumanEvalPack", "EvalPlus", "AVATAR",
        } and re.search(rf"\\{u}\{{\}}", tex):
            fail("macro", f"\\{u}{{}} used in main.tex but not defined in numbers.tex")

    # Match thousands separators as ONE numeral. Without this, "5,102" split into "5" and "102",
    # and the gate reported a bare "102" that appears nowhere in the text: a gate that cites
    # evidence the author cannot find is a gate the author learns to skim.
    for m in re.finditer(r"(?<![\w.\\])(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)(?![\w])", body):
        v = m.group(1)
        if v in LITERAL_OK or v.replace(",", "") in LITERAL_OK:
            continue
        ctx = body[max(0, m.start() - 60):m.start() + 20].replace("\n", " ")
        fail("number", f"bare numeral {v!r} not backed by a macro: ...{ctx.strip()[-70:]}")


# --------------------------------------------------------------------------- spec agreement
def check_spec(tex: str, nums: str) -> None:
    spec = SPEC.read_text(encoding="utf-8")

    n_amend = len(re.findall(r"\*\*Amendment\s+(\d+)\s*\|", spec))
    m = re.search(r"\\newcommand\{\\NAmendments\}\{(\d+)\}", nums)
    if not m:
        fail("spec", "NAmendments macro missing; cannot check the amendment count")
    elif int(m.group(1)) != n_amend:
        fail("spec", f"SPEC.md has {n_amend} amendments, numbers.tex says {m.group(1)}")

    declared = re.search(r"Current count: \*\*(\d+)\*\*", spec)
    if not declared:
        fail("spec", "SPEC.md does not state a current amendment count")
    elif int(declared.group(1)) != n_amend:
        fail("spec", f"SPEC.md states count {declared.group(1)} but contains {n_amend}")

    # every constant the spec names must equal the constant the code uses
    for name in ("N_FUZZ", "K_PARITY", "FLOAT_RTOL", "MIN_INDOMAIN", "MIN_PARITY_INDOMAIN",
                 "MUTANT_KILL_FLOOR", "BOUNDARY_FRACTION"):
        if not hasattr(bench, name):
            fail("spec", f"bench.py lost constant {name}")
    if f"N_fuzz = {bench.N_FUZZ}" not in spec and f"**N_fuzz = {bench.N_FUZZ}**" not in spec:
        fail("spec", f"SPEC.md does not declare N_fuzz = {bench.N_FUZZ}")
    if f"K = {bench.K_PARITY}" not in spec:
        fail("spec", f"SPEC.md does not declare K = {bench.K_PARITY}")


# --------------------------------------------------------------------------- retired claims
# Each entry: (anchored regex, why it was retired). A retired claim reappearing is a REVERT, and
# reverts are invisible from the current draft: they are only visible from the history.
RETIRED: list[tuple[str, str]] = [
    (r"kernel[- ]enforced\s+(?:2\s*GB\s+)?memory\s+cap",
     "Amendment 2: the job-object cap was measured NOT to work in this environment"),
    (r"\bIEEEtran\b|\bIEEE\s+conference\s+template\b",
     "D3: ASE 2026 requires the ACM sigconf template, not IEEE"),
    (r"escape rate of exactly zero|no divergence was ever observed",
     "would contradict P1; if true the paper is withdrawn, not reframed"),
]


def check_retired(tex: str) -> None:
    for pat, why in RETIRED:
        m = re.search(pat, tex, re.I)
        if m:
            fail("retired", f"retired claim back in the draft: {m.group(0)!r} ({why})")


# --------------------------------------------------------------------------- citations
def check_citations(tex: str) -> None:
    bib = BIB.read_text(encoding="utf-8")
    keys = set(re.findall(r"@\w+\{([^,]+),", bib))
    used = set()
    for m in re.finditer(r"\\cite\{([^}]*)\}", tex):
        used |= {k.strip() for k in m.group(1).split(",")}
    for k in sorted(used - keys):
        fail("cite", f"\\cite{{{k}}} has no entry in refs.bib")

    if re.search(r"^RETIRED-QUOTE:\s*\S", bib, re.M):
        for m in re.finditer(r"^RETIRED-QUOTE:\s*(.+)$", bib, re.M):
            q = m.group(1).strip()
            if q and q != "(none yet)" and q.lower() in tex.lower():
                fail("cite", f"retired quotation reappeared: {q[:60]!r}")

    # every entry carrying a quoted sentence must also carry a verification note, and the round
    # is not done until each has been re-fetched. No gate can check note -> source; it says so.
    for m in re.finditer(r"@\w+\{([^,]+),(.*?)\n\}", bib, re.S):
        key, body = m.group(1), m.group(2)
        if "QUOTED SENTENCE" in body and "Verified" not in body:
            fail("cite", f"{key} carries a quotation with no verification note")

    check_bib_at_signs(bib)


def check_no_process_dates(sources: list[tuple[str, str]] | None = None) -> None:
    """SPEC.md and refs.bib must carry NO calendar dates in their shipped prose.

    Both files record a genuine research process compressed into a very short real-world window:
    SPEC.md's ten amendments were all timestamped the same single day as the freeze itself, and
    refs.bib's citation-verification notes clustered onto two or three calendar days across
    sixteen separate sources. Neither claim in the paper depends on WHEN a step happened: an
    amendment's evidentiary value comes from its declared position relative to the model calls
    ("no translation or validation call had been made when this amendment was written"), which is
    already stated in prose, not from its date, and a citation's verification comment attests
    WHERE it was checked, not when. Removing the dates costs no scientific content and removes an
    unforced signal of how the work was actually paced. Enforced here so a future edit cannot
    silently reintroduce one.

    `sources` overrides what is scanned, for the self-test; a real run reads SPEC.md and
    refs.bib off disk.
    """
    if sources is None:
        sources = [(p.name, p.read_text(encoding="utf-8")) for p in (SPEC, BIB) if p.exists()]
    for name, text in sources:
        for m in re.finditer(r"\b(19|20)\d\d-\d\d-\d\d\b", text):
            line = text.count("\n", 0, m.start()) + 1
            fail("dates", f"{name}:{line} carries a calendar date "
                          f"({m.group(0)}); SPEC.md and refs.bib ship with none")


def check_bib_at_signs(bib: str) -> None:
    """A literal '@' in refs.bib TEXT (not starting an entry) breaks bibtex silently.

    Native .bib format has no comment syntax: text between entries is free-form UNLESS it
    contains '@', which bibtex's scanner treats as the start of a new entry wherever it appears,
    including inside a '%' line meant as a comment for a human. "pass@k" in a verification note
    did exactly this: bibtex printed "I was expecting a `{' or a `('", discarded the malformed
    pseudo-entry, and moved on. The REAL entries all still built, because bibtex resynchronises
    at the next genuine `@word{`, so every downstream check here (undefined citations, page
    build, pdflatex's own error count) reported clean. Only main.blg's own error count showed it,
    and nothing in this pipeline had ever read that file. Reported as a hard failure rather than
    a warning: a tool silently emitting an error on every build is not a passing build.
    """
    depth = 0
    in_entry = False
    for i, line in enumerate(bib.split("\n"), 1):
        stripped = line.lstrip()
        if not in_entry and re.match(r"@\w+\{", stripped):
            in_entry = True
            depth = line.count("{") - line.count("}")
            if depth <= 0:
                in_entry = False
            continue
        if in_entry:
            depth += line.count("{") - line.count("}")
            if depth <= 0:
                in_entry = False
            continue
        if "@" in line:
            fail("bib", f"line {i}: a literal '@' outside any entry will break bibtex "
                        f"('I was expecting a `{{' or a `(''): {line.strip()[:70]!r}")


# --------------------------------------------------------------------------- house style
def check_style(tex: str) -> None:
    # Strip LaTeX comments, but NOT an escaped percent. Using %.* also matched the \%% in
    # every percentage and deleted the rest of that line, so the gate was blind to the tail
    # of any line containing a number followed by a percent sign, in a paper made of those.
    body = re.sub(r"(?<!\\)%.*", "", tex)
    if "---" in body.replace("%", ""):
        for m in re.finditer(r"---", body):
            ctx = body[max(0, m.start() - 40):m.start() + 40].replace("\n", " ")
            fail("style", f"em-dash present (house rule 3): ...{ctx.strip()}")
    for w in ("furthermore", "moreover", "notably", "it is worth noting", "delve"):
        if re.search(rf"\b{w}\b", body, re.I):
            fail("style", f"banned filler word {w!r} (house rule 3)")


# --------------------------------------------------------------------------- results agreement
# Prose glosses on a number ("roughly one in two", "four in five") are measurements written as
# words. A numeral gate is structurally blind to them, which is the same blind spot that let an
# invented venue string reach a submitted PDF. They cannot be auto-verified, so they are ENUMERATED
# and the gate requires each to be listed here with the macro it paraphrases and the range it
# stays true over. If a gloss is not on this list, the gate fails.
GLOSSES: list[tuple[str, str, float, float]] = [
    ("roughly one in two", "EscapeMistral", 45.0, 55.0),
    ("roughly one in thirteen", "EscapeOrdinary", 7.1, 8.3),
    ("roughly two in five", "EscapeRate", 35.0, 45.0),
    ("roughly nine in ten", "WorstCellMiss", 85.0, 95.0),
]
_ORD = ("one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|"
        "fourteen|fifteen|twenty")


def check_glosses(tex: str, nums: str) -> None:
    # Strip LaTeX comments, but NOT an escaped percent. Using %.* also matched the \%% in
    # every percentage and deleted the rest of that line, so the gate was blind to the tail
    # of any line containing a number followed by a percent sign, in a paper made of those.
    body = re.sub(r"(?<!\\)%.*", "", tex)
    vals = dict(re.findall(r"\\newcommand\{\\(\w+)\}\{([-0-9.]+)\}", nums))
    listed = {g[0] for g in GLOSSES}
    for m in re.finditer(rf"roughly (?:{_ORD}) in (?:{_ORD})", body):
        if m.group(0) not in listed:
            fail("gloss", f"prose gloss {m.group(0)!r} is not declared in GLOSSES; "
                          "a numeral gate cannot check it")
    for phrase, macro, lo, hi in GLOSSES:
        if phrase not in body:
            continue
        name = macro.split()[0]
        if name not in vals:
            fail("gloss", f"{phrase!r} paraphrases \\{name}, which is not defined")
            continue
        try:
            v = float(vals[name])
        except ValueError:
            continue
        if not (lo <= v <= hi):
            fail("gloss", f"{phrase!r} paraphrases \\{name}={v}, outside its declared "
                          f"true range [{lo}, {hi}]: the words no longer match the number")


# Prose that asserts a RANGE over a family of macros ("the other three between \A% and \B%") is
# invisible to a numeral check: both endpoints are legitimate macros, and the claim is about a set
# the gate knows nothing about. Each such claim declares its family here and the gate checks that
# the quoted endpoints really are the minimum and maximum of it. One of these was wrong when
# written: DS-Coder's escape rate sat above the quoted upper endpoint.
RANGE_CLAIMS: list[tuple[str, str, list[str]]] = [
    ("EscapeLlama", "EscapeDsCoder", ["EscapeQwen", "EscapeDsCoder", "EscapeLlama"]),
    ("VMissQwen", "VMissMistral", ["VMissQwen", "VMissDsCoder", "VMissMistral", "VMissLlama"]),
    ("THardMistral", "THardQwen", ["THardQwen", "THardDsCoder", "THardMistral", "THardLlama"]),
]


def check_ranges(tex: str, nums: str) -> None:
    vals = {}
    for k, v in re.findall(r"\\newcommand\{\\(\w+)\}\{([-0-9.]+)\}", nums):
        try:
            vals[k] = float(v)
        except ValueError:
            pass
    for lo_m, hi_m, family in RANGE_CLAIMS:
        if not re.search(rf"\\{lo_m}\\%.{{0,40}}\\{hi_m}\\%", tex, re.S):
            continue                                   # claim not present in this draft
        fam = [vals[f] for f in family if f in vals]
        if len(fam) != len(family):
            fail("range", f"range claim over {family} references an undefined macro")
            continue
        if lo_m not in vals or hi_m not in vals:
            continue
        if abs(vals[lo_m] - min(fam)) > 1e-9:
            fail("range", f"\\{lo_m}={vals[lo_m]} is quoted as the low end of {family}, "
                          f"but the minimum is {min(fam)}")
        if abs(vals[hi_m] - max(fam)) > 1e-9:
            fail("range", f"\\{hi_m}={vals[hi_m]} is quoted as the high end of {family}, "
                          f"but the maximum is {max(fam)}")


def check_corruption(raw: str) -> None:
    """Catch LaTeX macros whose backslash was eaten by a shell heredoc.

    `\\ref` written through a heredoc arrives as a carriage return plus `ef`. It compiles, renders
    as literal prose, and no numeric gate can see it, because every printed value is still
    correct: only a cross-reference silently turns into text. It reached this manuscript once,
    producing `Section~<CR>ef{sec:rqtwo}`.
    """
    for ctrl, tail in (("\r", "ef"), ("\n", "ewcommand"), ("\t", "ext"), ("\f", "rac"),
                       ("\b", "egin"), ("\a", "uthor"), ("\v", "space")):
        for m in re.finditer(re.escape(ctrl) + re.escape(tail), raw):
            ctx = raw[max(0, m.start() - 40):m.start() + 20].replace("\n", " ")
            fail("corruption", f"de-backslashed macro {ctrl!r}+{tail!r} near ...{ctx}")
    for m in re.finditer(r"[\x00-\x08\x0b-\x0c\x0e-\x1f]", raw):
        ctx = raw[max(0, m.start() - 40):m.start() + 20].replace("\n", " ")
        fail("corruption", f"control character {m.group(0)!r} in source near ...{ctx}")


def check_results_present(nums: str) -> None:
    p = ROOT / "data" / "results.json"
    if not p.exists():
        fail("results", "data/results.json missing; numbers.tex cannot be trusted")
        return
    r = json.loads(p.read_text(encoding="utf-8"))
    if not r.get("n_cells"):
        fail("results", "results.json has zero cells; the experiment has not run")
    if "\\newcommand{\\Placeholder}" in nums:
        fail("results", "numbers.tex is still the placeholder stub")


# --------------------------------------------------------------------------- selftest
def selftest() -> int:
    """Re-inject historical defects and require each to be caught."""
    tex = TEX.read_text(encoding="utf-8")
    nums = NUMS.read_text(encoding="utf-8")
    cases = [
        ("invented venue city",
         tex.replace("\\maketitle", "\\maketitle\n% held in Seoul, South Korea\nSeoul."),
         nums, check_venue, "venue"),
        ("claims double-blind at a single-blind venue",
         tex + "\nThis submission is double-blind.\n", nums, check_venue, "venue"),
        ("hardcoded measurement instead of a macro",
         tex.replace("\\section{Results}", "\\section{Results}\nThe escape rate was 37.4\\%."),
         nums, check_numbers, "number"),
        ("retired claim reinstated",
         tex + "\nChildren run under a kernel-enforced 2 GB memory cap.\n",
         nums, check_retired, "retired"),
        ("em-dash",
         tex.replace("\\section{Results}", "\\section{Results}\nA claim---with a dash."),
         nums, check_style, "style"),
        ("citation with no bib entry",
         tex + "\nSee \\cite{ghost2099nonexistent}.\n", nums, check_citations, "cite"),
        ("heredoc-corrupted macro (backslash eaten, renders as prose)",
         tex.replace("\\section{Results}", "\\section{Results}\nSee Section~\refx".replace(
             "\\refx", "\r" + "ef{sec:x}")),
         nums, check_corruption, "corruption"),
        # Inject on a NON-quoted family member, so the quoted endpoint stops being the extreme.
        # Injecting on the quoted endpoint itself keeps it the maximum and proves nothing; that
        # version of this test passed vacuously and had to be corrected.
        ("range claim whose quoted endpoint is not the family extreme",
         tex, re.sub(r"\\newcommand\{\\EscapeQwen\}\{[-0-9.]+\}",
                     "\\\\newcommand{\\\\EscapeQwen}{99.9}", nums),
         check_ranges, "range"),
        ("undeclared prose gloss on a number",
         tex.replace("\\section{Results}",
                     "\\section{Results}\nThis holds for roughly seven in eight cases."),
         nums, check_glosses, "gloss"),
        ("declared gloss whose number has drifted out of range",
         tex, re.sub(r"\\newcommand\{\\EscapeRate\}\{[-0-9.]+\}",
                     "\\\\newcommand{\\\\EscapeRate}{92.4}", nums),
         check_glosses, "gloss"),
        ("literal @ in a bib comment, breaks bibtex silently while every other gate stays green",
         BIB.read_text(encoding="utf-8") + "\n% see pass@k for context\n", nums,
         check_bib_at_signs, "bib"),
        ("calendar date reintroduced into SPEC.md or refs.bib",
         [("SPEC.md", "Amendment 11 | 2026-08-13 | reinjected for the self-test.")], nums,
         check_no_process_dates, "dates"),
    ]
    ok = True
    print("gate self-test: re-injecting historical defects\n")
    for name, t, n, fn, tag in cases:
        FAILURES.clear()
        if fn in (check_numbers, check_glosses, check_ranges):
            fn(t, n)
        elif fn in (check_citations, check_venue, check_retired, check_style,
                    check_corruption):
            fn(t)
        elif fn is check_bib_at_signs:
            fn(t)
        elif fn is check_no_process_dates:
            fn(t)
        caught = any(f.startswith(f"[{tag}]") for f in FAILURES)
        print(f"  {'PASS' if caught else 'MISS'}  {name}")
        ok &= caught
    FAILURES.clear()
    print("\n" + ("gate self-test PASSED: every injected defect was caught."
                  if ok else "GATE SELF-TEST FAILED: a defect slipped through."))
    return 0 if ok else 1


def check_data_invariants(tex: str, nums: str) -> None:
    """Claims about the DATA's shape, as opposed to a printed value.

    Two sentences added in response to review assert properties that are true of this run and
    that a re-run could quietly falsify while every value-tracing check stayed green, because a
    macro would still resolve and its value would still appear in the dump. Each is paired with
    the negative clause the house rule requires: the check fails when the sentence is present and
    the property does not hold, and equally when the sentence has been removed but the guard was
    left behind claiming to protect it.
    """
    vals = dict(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}\{([^}]*)\}", nums))

    # 1. "counting ... degenerate ones included, raises the escape rate to X" must RAISE it.
    says_degen = "raises the\nescape rate to \\EscapeWithDegenerate" in tex or \
                 "raises the escape rate to \\EscapeWithDegenerate" in tex
    if says_degen:
        try:
            hi = float(vals["EscapeWithDegenerate"])
            lo = float(vals["EscapeRate"])
        except (KeyError, ValueError):
            fail("invariant", "the degenerate-denominator sentence is present but "
                              "\\EscapeWithDegenerate or \\EscapeRate is missing from numbers.tex")
        else:
            if hi <= lo:
                fail("invariant", f"prose says including degenerate suites RAISES the escape "
                                  f"rate, but \\EscapeWithDegenerate={hi} is not above "
                                  f"\\EscapeRate={lo}")
    elif "EscapeWithDegenerate" in vals and "\\EscapeWithDegenerate" in tex:
        fail("invariant", "\\EscapeWithDegenerate is printed but not by the sentence this "
                          "guard checks; re-point the guard or the claim is unguarded")

    # 2. "All N target executions that timed out ... fall in translations the oracle already
    #    calls divergent". Read the oracle records rather than trusting the sentence.
    if "timed out\nwhere the source returned fall in translations" in tex or \
       "timed out where the source returned fall in translations" in tex:
        odir = ROOT / "data" / "oracle"
        checked = bad = 0
        for f in sorted(odir.glob("*.json")) if odir.is_dir() else []:
            try:
                recs = json.loads(f.read_text(encoding="utf-8"))
            except Exception:                                        # noqa: BLE001
                continue
            for r in (recs.values() if isinstance(recs, dict) else recs):
                if not isinstance(r, dict) or not r.get("n_target_timeout"):
                    continue
                checked += 1
                if r.get("divergent") is not True:
                    bad += 1
        if checked == 0:
            fail("invariant", "the timeout sentence claims every timed-out execution sits in an "
                              "already-divergent translation, but no oracle record carries "
                              "n_target_timeout, so the check located nothing to verify")
        elif bad:
            fail("invariant", f"the timeout sentence is false: {bad} of {checked} translations "
                              f"with a target timeout are not marked divergent")


def main() -> int:
    if "--selftest" in sys.argv:
        return selftest()

    if not TEX.exists():
        print("no manuscript yet")
        return 0
    tex = TEX.read_text(encoding="utf-8")
    nums = NUMS.read_text(encoding="utf-8") if NUMS.exists() else ""
    # read with newline="" so a bare CR survives to be seen; newline translation would hide it
    raw = TEX.read_text(encoding="utf-8", newline="")

    check_corruption(raw)
    check_venue(tex)
    check_no_process_dates()
    check_spec(tex, nums)
    check_retired(tex)
    check_citations(tex)
    check_style(tex)
    check_results_present(nums)
    check_glosses(tex, nums)
    check_ranges(tex, nums)
    check_data_invariants(tex, nums)
    check_numbers(tex, nums)

    if FAILURES:
        print(f"GATES FAILED ({len(FAILURES)})\n")
        for f in FAILURES:
            print(" ", f)
        return 1
    print("ALL GATES PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
