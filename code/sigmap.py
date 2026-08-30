"""Signature parsing and Python->Java type mapping.

The whole marshalling design rests on one idea: we do NOT let the translator choose the Java
signature. We derive it mechanically from the Python type hints and hand it to the model as a
requirement. That removes signature ambiguity as a confound and makes a single generic JSON
harness sufficient for every problem.

Type-mapping decision that matters and is recorded in SPEC.md: Python `int` maps to Java `long`,
not `int`. Python integers are arbitrary precision, so no fixed-width Java type is faithful.
`long` is the conservative choice: it makes overflow divergence HARDER to produce than `int`
would, so every escape rate measured under it is a lower bound.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# Supported abstract types. Anything outside this set makes a problem ineligible.
SCALARS = {"int", "float", "bool", "str"}


@dataclass(frozen=True)
class JType:
    """A supported type, as both its Java spelling and a tag the marshaller understands."""

    java: str
    tag: str  # one of: long,double,boolean,String,long[],double[],String[],boolean[],long[][],
    # double[][],String[][],mapSL,void


_PY_TO_J = {
    "int": JType("long", "long"),
    "float": JType("double", "double"),
    "bool": JType("boolean", "boolean"),
    "str": JType("String", "String"),
}

_LIST_TO_J = {
    "int": JType("long[]", "long[]"),
    "float": JType("double[]", "double[]"),
    "bool": JType("boolean[]", "boolean[]"),
    "str": JType("String[]", "String[]"),
}

_LIST_LIST_TO_J = {
    "int": JType("long[][]", "long[][]"),
    "float": JType("double[][]", "double[][]"),
    "str": JType("String[][]", "String[][]"),
}


def _norm(t: str) -> str:
    """Normalise a Python annotation string: strip spaces, unify List/list spellings."""
    t = t.strip()
    t = re.sub(r"\btyping\.", "", t)
    t = re.sub(r"\blist\b", "List", t)
    t = re.sub(r"\btuple\b", "Tuple", t)
    t = re.sub(r"\bdict\b", "Dict", t)
    t = re.sub(r"\s+", "", t)
    return t


def map_type(ann: str) -> JType | None:
    """Map one Python annotation to a JType, or None if unsupported."""
    t = _norm(ann)
    if t in _PY_TO_J:
        return _PY_TO_J[t]
    m = re.fullmatch(r"List\[(\w+)\]", t)
    if m and m.group(1) in _LIST_TO_J:
        return _LIST_TO_J[m.group(1)]
    m = re.fullmatch(r"List\[List\[(\w+)\]\]", t)
    if m and m.group(1) in _LIST_LIST_TO_J:
        return _LIST_LIST_TO_J[m.group(1)]
    return None


def split_top_level(s: str, sep: str = ",") -> list[str]:
    """Split on `sep` at bracket depth zero."""
    out, depth, cur = [], 0, ""
    for ch in s:
        if ch in "[(":
            depth += 1
        elif ch in ")]":
            depth -= 1
        if ch == sep and depth == 0:
            out.append(cur)
            cur = ""
        else:
            cur += ch
    if cur.strip():
        out.append(cur)
    return [x.strip() for x in out if x.strip()]


@dataclass
class Sig:
    name: str
    params: list[tuple[str, JType]]
    ret: JType

    @property
    def java_decl(self) -> str:
        args = ", ".join(f"{t.java} {n}" for n, t in self.params)
        return f"public static {self.ret.java} {self.name}({args})"


class Unsupported(Exception):
    pass


def parse_signature(sig: str) -> Sig:
    """Parse a HumanEvalPack `signature` string into a Sig, or raise Unsupported.

    Example input: `has_close_elements(numbers: List[float], threshold: float) -> bool`
    """
    m = re.fullmatch(r"\s*(\w+)\s*\((.*)\)\s*->\s*(.+?)\s*", sig, re.S)
    if not m:
        raise Unsupported(f"unparseable signature: {sig!r}")
    name, params_s, ret_s = m.group(1), m.group(2), m.group(3)

    ret = map_type(ret_s)
    if ret is None:
        raise Unsupported(f"unsupported return type: {ret_s!r}")

    params: list[tuple[str, JType]] = []
    for p in split_top_level(params_s):
        if "=" in p.split(":")[-1] and ":" in p:
            # a defaulted parameter; the default is dropped and the parameter stays required
            p = p.split("=")[0].strip()
        elif "=" in p:
            raise Unsupported(f"defaulted parameter without annotation: {p!r}")
        if ":" not in p:
            raise Unsupported(f"unannotated parameter: {p!r}")
        pname, ann = p.split(":", 1)
        jt = map_type(ann)
        if jt is None:
            raise Unsupported(f"unsupported parameter type: {ann!r}")
        params.append((pname.strip(), jt))

    if not params:
        raise Unsupported("zero-parameter function: nothing to fuzz")
    return Sig(name=name, params=params, ret=ret)
