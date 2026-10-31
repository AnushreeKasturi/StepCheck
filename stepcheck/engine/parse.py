"""LaTeX line -> MathObject.

SymPy's `parse_latex` is only ever given a single expression. Line structure
(`=`, "or", commas, `\\pm`) is split here first, because the raw parser silently
mangles it: `x = 2, x = -3` parses as just `x = 2`, `x=2 \\text{ or } x=-3`
evaluates to `False`, and `\\pm` becomes a variable named `pm`.
"""

import re
import warnings

import sympy as sp
from sympy.core.function import AppliedUndef
from sympy.parsing.latex import parse_latex
from sympy.parsing.latex.errors import LaTeXParsingError

from stepcheck.engine.model import Disjunction, Equation, Expression, MathObject, Step


class ParseError(ValueError):
    """A line could not be turned into a supported math object."""


_OR_SEPARATOR = "\x00"
_OR_PATTERNS = [
    r"\\(?:text|mathrm|textrm)\{\s*or\s*\}",
    r"\\lor\b",
    r"\\vee\b",
    r"\bor\b",
]
_SPACING = [r"\\quad", r"\\qquad", r"\\,", r"\\;", r"\\:", r"\\!", "~"]
_UNSUPPORTED_RELATIONS = ["<", ">", r"\le", r"\ge", r"\lt", r"\gt", r"\neq", r"\ne"]


def normalize_latex(latex: str) -> str:
    s = latex.strip().strip("$").strip()
    s = s.replace("\u2212", "-")  # unicode minus
    s = re.sub(r"\\[dt]frac", r"\\frac", s)
    s = re.sub(r"\\left\s*\.|\\right\s*\.", "", s)  # invisible delimiters
    s = re.sub(r"\\left|\\right", "", s)
    for pattern in _SPACING:
        s = re.sub(pattern, " ", s)
    for pattern in _OR_PATTERNS:
        s = re.sub(pattern, _OR_SEPARATOR, s)
    return s.strip()


def split_top_level(s: str, separators: str) -> list[str]:
    """Split on any char in `separators` that is not inside (), [] or {}."""
    parts: list[str] = []
    current: list[str] = []
    depth = 0
    for ch in s:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if depth == 0 and ch in separators:
            parts.append("".join(current))
            current = []
        else:
            current.append(ch)
    parts.append("".join(current))
    return [p.strip() for p in parts]


def expand_plus_minus(s: str) -> list[str]:
    """`x = 1 \\pm 2` -> [`x = 1 + 2`, `x = 1 - 2`].

    All \\pm in a line share one sign (the usual convention); \\mp takes the opposite sign.
    """
    if r"\pm" not in s and r"\mp" not in s:
        return [s]
    return [
        s.replace(r"\pm", "+").replace(r"\mp", "-"),
        s.replace(r"\pm", "-").replace(r"\mp", "+"),
    ]


def _repair_implicit_products(expr: sp.Expr) -> sp.Expr:
    """`x(x+1)` parses as a call of a function named x; turn it back into a product."""

    def is_call(e: sp.Basic) -> bool:
        return isinstance(e, AppliedUndef)

    def to_product(e: sp.Basic) -> sp.Expr:
        if len(e.args) != 1:
            raise ParseError(f"cannot interpret {e} as a product")
        return sp.Symbol(e.func.__name__) * e.args[0]

    return expr.replace(is_call, to_product)


def _floats_to_rationals(expr: sp.Expr) -> sp.Expr:
    """0.5 -> 1/2, so every later check can use exact arithmetic."""
    return expr.xreplace({f: sp.Rational(str(f)) for f in expr.atoms(sp.Float)})


def _make_symbols_real(expr: sp.Expr) -> sp.Expr:
    """We work over the reals: declaring x real (never positive) keeps sqrt(x^2) = |x|."""
    for s in expr.free_symbols:
        if len(s.name) != 1 or not s.name.isalpha():
            raise ParseError(f"unrecognized symbol '{s.name}'")
    return expr.xreplace({s: sp.Symbol(s.name, real=True) for s in expr.free_symbols})


def parse_expression(latex: str) -> sp.Expr:
    if not latex:
        raise ParseError("empty expression")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = parse_latex(latex)
    except (LaTeXParsingError, TypeError, ValueError) as exc:
        raise ParseError(f"could not parse '{latex}'") from exc
    if not isinstance(result, sp.Expr):
        raise ParseError(f"'{latex}' is not an expression")
    # parse_latex returns an *unevaluated* tree (1/0 stays as-is); evaluate it.
    expr = _repair_implicit_products(result.doit())
    expr = _floats_to_rationals(expr)
    expr = _make_symbols_real(expr)
    if expr.has(sp.zoo, sp.nan, sp.oo, -sp.oo):
        raise ParseError(f"'{latex}' is undefined (division by zero?)")
    return expr


def _parse_equation_or_expression(latex: str) -> Equation | Expression:
    sides = split_top_level(latex, "=")
    if len(sides) == 1:
        return Expression(parse_expression(sides[0]))
    if len(sides) == 2:
        return Equation(parse_expression(sides[0]), parse_expression(sides[1]))
    raise ParseError("chained equalities (a = b = c) are not supported; put each on its own line")


def parse_math(latex: str) -> MathObject:
    s = normalize_latex(latex)
    if any(rel in s for rel in _UNSUPPORTED_RELATIONS):
        raise ParseError("inequalities are not supported yet")
    if s.startswith("="):
        # Continuation line in a simplification chain: "= 2x + 2".
        return Expression(parse_expression(s[1:].strip()))

    disjuncts = [
        variant
        for part in split_top_level(s, _OR_SEPARATOR + ",;")
        if part
        for variant in expand_plus_minus(part)
    ]
    if not disjuncts:
        raise ParseError("empty line")
    parsed = [_parse_equation_or_expression(d) for d in disjuncts]
    if len(parsed) == 1:
        return parsed[0]
    equations = tuple(p for p in parsed if isinstance(p, Equation))
    if len(equations) != len(parsed):
        raise ParseError("every part of an 'or' line must be an equation")
    return Disjunction(equations)


def free_symbols(obj: MathObject) -> set[sp.Symbol]:
    match obj:
        case Expression(expr):
            return set(expr.free_symbols)
        case Equation(lhs, rhs):
            return set(lhs.free_symbols | rhs.free_symbols)
        case Disjunction(equations):
            return set().union(*(free_symbols(e) for e in equations))


def parse_line(latex: str, line_no: int) -> Step:
    try:
        obj = parse_math(latex)
    except ParseError as exc:
        return Step(line_no, latex, None, str(exc))
    return Step(line_no, latex, obj)
