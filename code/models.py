"""Model routes and prompts.

Two routes, one interface. Ollama serves the local checkpoints; the hosted model is reached
through a vendor CLI client. Every call is logged to data/calls.jsonl BEFORE the caller sees the result, so a run
that dies halfway still leaves an auditable trail of what was actually asked and answered.

Note on the CLI route: the prompt MUST be the value of --print and --model MUST come first
(`agy --model M --print "..."`). With the flags the other way round Go's flag parser makes
--print swallow the literal string "--model" as its prompt and the client answers a different question
with exit status 0, so the flag order matters for reproducing the frontier arm.
"""
from __future__ import annotations

import json
import re
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
LOG = ROOT / "data" / "calls.jsonl"

OLLAMA = "http://localhost:11434/api/generate"

TEMPERATURE = 0.2
TOP_P = 0.95
SEED = 0
MAX_TOKENS = 1024


@dataclass(frozen=True)
class Model:
    mid: str          # short id used in filenames and tables
    route: str        # "ollama" | "agy"
    name: str         # provider-side model name
    family: str


MODELS = [
    Model("qwen25", "ollama", "qwen2.5:7b-instruct", "Qwen"),
    Model("dscoder", "ollama", "deepseek-coder:6.7b-instruct", "DeepSeek"),
    Model("mistral", "ollama", "mistral:7b-instruct-v0.2-q4_0", "Mistral"),
    Model("llama3", "ollama", "llama3:latest", "Llama"),
    Model("gemini", "agy", "gemini-3.6-flash-high", "Gemini"),
]
BY_ID = {m.mid: m for m in MODELS}


def _log(rec: dict) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")


def call(model: Model, prompt: str, tag: str, max_tokens: int = MAX_TOKENS,
         retries: int = 5) -> str:
    """One completion. Returns raw text; never raises for a provider error, returns "" instead.

    Retries generously with growing backoff. The hosted route went through a sustained outage
    that returned empty responses for roughly 160 consecutive calls, and with only two quick
    retries every one of them was recorded as though the model had produced nothing parseable.
    That is a transport failure wearing a result's clothes, and it wiped out an entire
    translator column for one validator: precisely the validator-correlated hole that the paired
    analysis cannot absorb. An empty return is now reported to the caller, which records it as a
    failed CALL rather than as an unparseable ANSWER.
    """
    t0 = time.time()
    text, err = "", None
    for attempt in range(retries + 1):
        try:
            if model.route == "ollama":
                body = json.dumps({
                    "model": model.name, "prompt": prompt, "stream": False,
                    "options": {"temperature": TEMPERATURE, "top_p": TOP_P,
                                "seed": SEED, "num_predict": max_tokens},
                }).encode("utf-8")
                req = urllib.request.Request(OLLAMA, body,
                                             {"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=900) as r:
                    text = json.load(r).get("response", "")
            else:
                r = subprocess.run(
                    ["agy", "--model", model.name, "--print", prompt],
                    capture_output=True, text=True, timeout=900,
                    encoding="utf-8", errors="replace")
                text = r.stdout or ""
            if text.strip():
                err = None
                break
            err = "empty response"
        except Exception as e:                      # noqa: BLE001
            err = f"{type(e).__name__}: {e}"[:300]
        if attempt < retries:
            time.sleep(min(60, 3 * (2 ** attempt)))

    _log({"t": time.time(), "model": model.mid, "route": model.route, "tag": tag,
          "dt": round(time.time() - t0, 2), "err": err,
          "prompt_chars": len(prompt), "resp_chars": len(text)})
    return text


# ------------------------------------------------------------------------------------- prompts

TRANSLATE = """You are migrating a legacy Python function to Java as part of a system modernization.

Legacy Python source:
```python
{source}
```

Write the Java replacement. Requirements, all mandatory:
- Output a single public class named exactly `Solution`.
- It must contain exactly this method, with this signature, unchanged:
  {decl}
- The method must be static and must reproduce the behaviour of the Python source.
- You may add private helper methods and `import` statements inside the file.
- Do not write a `main` method. Do not write comments explaining your reasoning.
- Output ONLY Java source code. No markdown fences, no prose before or after.
"""

VALIDATE = """You are validating a legacy-to-Java migration before it goes to production.

The legacy Python source (the system of record):
```python
{source}
```

The candidate Java replacement:
```java
{java}
```

Your job is to choose the test INPUTS for a parity test. The inputs you choose will be run
through BOTH programs and their outputs compared; the expected values come from executing the
legacy source, so you do not need to predict any output. Choose inputs that are most likely to
expose a behavioural difference between the two programs, if one exists.

Return exactly {k} test inputs as a JSON array of arrays. Each inner array holds the
{nargs} argument(s) in order, matching these Java parameter types: {ptypes}

Rules:
- Output ONLY the JSON array. No markdown fences, no prose, no comments, no expected values.
- Every input must be a legal call to the legacy function.
- Use JSON types: numbers for numeric parameters, strings in double quotes, true/false for
  booleans, and JSON arrays for list parameters.

Example of the required output shape for a function taking (long, String):
[[1, "a"], [0, ""], [-5, "abc"]]
"""

VALIDATE_TARGETED = VALIDATE.replace(
    "Choose inputs that are most likely to expose a behavioural difference between the two\nprograms, if one exists.",
    """Choose inputs that are most likely to expose a behavioural difference between the two
programs, if one exists. Python and Java differ systematically in ways that survive a casual
review, so deliberately probe: integer width and overflow (Java `long` wraps, Python does not);
integer division and rounding of negative values; floating-point formatting and comparison;
empty and single-element collections; string indexing, slicing beyond bounds, case, and
non-ASCII characters; sort stability and the ordering of equal elements; and the boundary
between an empty result and an error.""")


# ------------------------------------------------------------------------------- output parsing

def extract_java(text: str) -> str:
    """Pull a compilable Java file out of a model response."""
    if not text:
        return ""
    fences = re.findall(r"```(?:java)?\s*\n(.*?)```", text, re.S)
    if fences:
        cand = max(fences, key=len)
    else:
        cand = text
    # drop any prose before the first import/class line
    m = re.search(r"^\s*(import\s|public\s+class\s|class\s|final\s+class\s)", cand, re.M)
    if m:
        cand = cand[m.start():]
    # drop trailing prose after the last closing brace
    last = cand.rfind("}")
    if last >= 0:
        cand = cand[:last + 1]
    return cand.strip()


def _conforms(v, tag: str) -> bool:
    """Does a Python value fit an abstract parameter type tag? (mirrors bench.conforms)"""
    if tag == "int":
        return isinstance(v, int) and not isinstance(v, bool)
    if tag == "float":
        return isinstance(v, (int, float)) and not isinstance(v, bool)
    if tag == "bool":
        return isinstance(v, bool)
    if tag == "str":
        return isinstance(v, str)
    if tag.startswith("list["):
        inner = tag[5:-1]
        return isinstance(v, list) and all(_conforms(x, inner) for x in v)
    return False


def _score(cands: list[list], ptypes: list[str] | None) -> float:
    """Share of candidate inputs whose arguments all type-check."""
    if not cands:
        return -1.0
    if not ptypes:
        return 0.0
    ok = 0
    for row in cands:
        if len(row) == len(ptypes) and all(_conforms(a, t) for a, t in zip(row, ptypes)):
            ok += 1
    return ok / len(cands)


def _split_top_level(inner: str) -> list[str]:
    """Split a JSON array body at depth-zero commas, respecting strings and escapes."""
    parts, depth, cur, instr, esc = [], 0, "", False, False
    for ch in inner:
        if instr:
            cur += ch
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                instr = False
            continue
        if ch == '"':
            instr = True
            cur += ch
            continue
        if ch in "[{":
            depth += 1
        elif ch in "]}":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append(cur)
            cur = ""
        else:
            cur += ch
    if cur.strip():
        parts.append(cur)
    return parts


def _salvage(block: str) -> list | None:
    """Parse a malformed JSON array element by element, keeping the elements that are valid.

    Small models routinely emit Java rather than JSON inside an otherwise well-formed list:
    `[Math.PI]`, `[Double.MAX_VALUE]`, `[1L, Long.MIN_VALUE, Long.MAX_VALUE]`, and occasionally a
    `)` where a `]` belongs. One such element makes `json.loads` reject the WHOLE array, after
    which the scanner falls back to whatever tiny valid fragment it can find: a model that
    proposed eight inputs was being scored on one.

    Salvaging keeps the inputs the model actually expressed and drops only the ones it failed to
    express in the requested format. It cannot invent an input and it cannot improve a badly
    chosen one. Elements that are dropped are counted, and format compliance is reported
    separately as its own result.
    """
    b = block.strip()
    if not (b.startswith("[") and b.endswith("]")):
        return None
    out = []
    for part in _split_top_level(b[1:-1]):
        try:
            out.append(json.loads(part.strip()))
        except Exception:                                    # noqa: BLE001
            continue
    return out or None


def extract_inputs(text: str, nargs: int,
                   ptypes: list[str] | None = None) -> list[list] | None:
    """Pull a JSON array of argument tuples out of a model response.

    Format is genuinely ambiguous for single-argument functions, and resolving it wrongly changes
    what the study measures. For a function taking one `list[int]`, a model that answers
    `[[0], [1], [-1]]` may mean three calls whose single argument is the scalar 0, 1, -1, or three
    calls whose single argument is the LIST [0], [1], [-1]. Reading it the first way made the
    source raise a TypeError on every input and the suite score as degenerate, so the measurement
    became JSON format compliance rather than input-selection power, which is a different paper.

    Every plausible reading is therefore enumerated and scored against the DECLARED parameter
    types, and the best-typechecking one wins. This resolves formatting only: it never changes
    which values the model chose, and it cannot make a badly chosen input into a good one.

    Returns None when nothing parseable is present, which is recorded as a degenerate suite
    rather than silently treated as an empty (and therefore vacuously passing) suite.
    """
    if not text:
        return None
    fences = re.findall(r"```(?:json)?\s*\n(.*?)```", text, re.S)
    best: list[list] | None = None
    best_score = -1.0

    for c in fences + [text]:
        for s in (i for i, ch in enumerate(c) if ch == "["):
            depth = 0
            for e in range(s, len(c)):
                if c[e] == "[":
                    depth += 1
                elif c[e] == "]":
                    depth -= 1
                    if depth == 0:
                        try:
                            v = json.loads(c[s:e + 1])
                        except Exception:                    # noqa: BLE001
                            v = _salvage(c[s:e + 1])
                            if v is None:
                                break
                        if not (isinstance(v, list) and v):
                            break
                        readings: list[list[list]] = []
                        # (a) each element is an argument tuple of the right arity
                        if all(isinstance(x, list) for x in v):
                            rows = [x for x in v if len(x) == nargs]
                            if rows:
                                readings.append(rows)
                        # (b) each element is the SINGLE argument itself
                        if nargs == 1:
                            readings.append([[x] for x in v])
                        # (c) the whole array is one argument tuple
                        if len(v) == nargs and not all(isinstance(x, list) for x in v):
                            readings.append([v])
                        for r in readings:
                            sc = _score(r, ptypes)
                            if sc > best_score:
                                best, best_score = r, sc
                        break
        if best is not None and best_score >= 1.0:
            break
    return best
