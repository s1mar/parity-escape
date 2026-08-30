"""Regenerate the paste-ready abstract from main.tex SOURCE, with macros resolved.

Never copy an abstract out of a rendered PDF. Doing that once carried a line-break hyphen,
"pub- lication churn", verbatim into a submission form and would have gone into the proceedings
metadata that way.

Usage:  python abstract_plain.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench  # noqa: E402

TEX = bench.ROOT / "paper" / "main.tex"
NUMS = bench.ROOT / "paper" / "numbers.tex"
OUT = bench.ROOT / "paper" / "abstract_plain.txt"


def main() -> None:
    tex = TEX.read_text(encoding="utf-8")
    vals = dict(re.findall(r"\\newcommand\{\\(\w+)\}\{([^}]*)\}",
                           NUMS.read_text(encoding="utf-8")))

    m = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", tex, re.S)
    if not m:
        raise SystemExit("no abstract found")
    a = m.group(1)

    # Strip LaTeX comments but NOT an escaped percent. Using %.* here deleted everything after
    # the first "\%" in the abstract, which is the identical fail-open bug that gates.py had:
    # in a paper made of percentages it silently truncates most of the text.
    a = re.sub(r"(?<!\\)%.*", "", a)
    # resolve macros, longest name first so \NModels is not eaten by a shorter prefix
    for k in sorted(vals, key=len, reverse=True):
        a = a.replace("\\" + k + "{}", vals[k]).replace("\\" + k + " ", vals[k] + " ")
        a = re.sub(r"\\" + k + r"\b", vals[k], a)
    a = re.sub(r"\\emph\{([^}]*)\}", r"\1", a)
    a = re.sub(r"\\textbf\{([^}]*)\}", r"\1", a)
    a = re.sub(r"\\texttt\{([^}]*)\}", r"\1", a)
    a = a.replace("\\%", "%").replace("\\&", "&").replace("~", " ")
    a = a.replace("$p{=}$", "p=").replace("${=}$", "=")
    a = re.sub(r"\$p\{=\}([^$]*)\$", r"p=\1", a)
    a = re.sub(r"\$([^$]*)\$", r"\1", a)
    a = a.replace("--", "-")
    a = re.sub(r"\\[a-zA-Z]+", "", a)
    a = re.sub(r"[ \t]+", " ", a)
    paras = [" ".join(p.split()) for p in a.split("\n\n") if p.strip()]
    text = "\n\n".join(paras).strip()

    OUT.write_text(text + "\n", encoding="utf-8")
    print(text)
    print()
    print("-" * 70)
    print(f"{len(text)} characters, {len(text.split())} words -> {OUT}")
    leftover = re.findall(r"[\\{}]", text)
    print("residual LaTeX characters:", "NONE" if not leftover else set(leftover))


if __name__ == "__main__":
    main()
