"""Backward search: small edits to the student's line that would make it correct.

Catches slips in the *result* of a step (a wrong coefficient, one flipped sign), which forward
mutations of the previous line cannot reproduce reliably. The repaired line is, by
construction, what the student should have written.
"""

from collections.abc import Callable, Iterator
from dataclasses import dataclass

import sympy as sp

from stepcheck.engine.model import Disjunction, Equation, ErrorType, Expression, MathObject
from stepcheck.engine.rewrite import Path, evaluate, negate, nodes, ordered, replace_at


@dataclass(frozen=True)
class Repair:
    error_type: ErrorType
    repaired: MathObject  # the student's line with one small fix
    term: sp.Expr  # the part of the student's line that was wrong
    fixed: sp.Expr  # what that part should have been


def number_variants(n: sp.Rational) -> list[tuple[ErrorType, sp.Rational]]:
    """Plausible slips: off by one or two, or the wrong sign."""
    if n.is_Integer:
        slips = [n + d for d in (1, -1, 2, -2)]
    else:
        slips = [sp.Rational(n.p + d, n.q) for d in (1, -1)]
    found = [(ErrorType.ARITHMETIC_SLIP, v) for v in slips if v != 0 or n.is_Integer]
    if n != 0:
        found.append((ErrorType.SIGN_ERROR, -n))
    return found


def _is_exponent(path: Path, parents: dict[Path, sp.Basic]) -> bool:
    return bool(path) and isinstance(parents.get(path[:-1]), sp.Pow) and path[-1] == 1


RepairEdit = tuple[ErrorType, sp.Expr, sp.Expr, sp.Expr]  # (type, repaired, wrong, fixed)


def _is_sign_factor(path: Path, node: sp.Basic, parents: dict[Path, sp.Basic]) -> bool:
    """The -1 in Mul(-1, ...) is how "x - 3" is stored; flipping it is a sign repair."""
    parent = parents.get(path[:-1]) if path else None
    return node == -1 and isinstance(parent, sp.Mul) and len(parent.args) > 1


def number_repairs(expr: sp.Expr) -> Iterator[RepairEdit]:
    found = dict(nodes(expr))
    for path, node in found.items():
        if (
            node.is_Rational
            and not _is_sign_factor(path, node, found)
            and not _is_exponent(path, found)
        ):
            for kind, value in number_variants(node):
                yield kind, replace_at(expr, path, value), node, value


def _term_slots(expr: sp.Expr) -> Iterator[tuple[Path, sp.Expr]]:
    """Each top-level term of every sum, and the whole expression if it is a single term."""
    found = dict(nodes(expr))
    if not isinstance(expr, sp.Add):
        yield (), expr
    for path, node in found.items():
        if isinstance(node, sp.Add):
            for i, t in enumerate(node.args):
                yield (*path, i), t


def coefficient_repairs(expr: sp.Expr) -> Iterator[RepairEdit]:
    """-x should have been -3x: slips in a coefficient that is not written as a number."""
    for path, t in _term_slots(expr):
        coeff, body = evaluate(t).as_coeff_Mul()
        if abs(coeff) != 1 or not body.free_symbols:
            continue
        for d in (1, -1, 2, -2):
            if coeff + d != 0:
                fixed = (coeff + d) * body
                yield ErrorType.ARITHMETIC_SLIP, replace_at(expr, path, fixed), t, fixed


def term_sign_repairs(expr: sp.Expr) -> Iterator[RepairEdit]:
    for path, node in nodes(expr):
        if isinstance(node, sp.Add):
            for i, t in enumerate(node.args):
                flipped = negate(t)
                yield ErrorType.SIGN_ERROR, replace_at(expr, (*path, i), flipped), t, flipped


def side_sign_repairs(expr: sp.Expr) -> Iterator[RepairEdit]:
    if not isinstance(expr, sp.Add):
        yield ErrorType.SIGN_ERROR, negate(expr), expr, negate(expr)


# Most natural edits first, across the whole line: x = 5 -> x = -5 reads better than -x = 5.
PASSES = [number_repairs, coefficient_repairs, term_sign_repairs, side_sign_repairs]


def _slots(obj: MathObject) -> Iterator[tuple[sp.Expr, Callable[[sp.Expr], MathObject]]]:
    """Every expression in the line, with a function that puts a new one in its place."""
    match obj:
        case Expression(expr):
            yield expr, Expression
        case Equation(lhs, rhs):
            yield lhs, lambda e: Equation(e, rhs)
            yield rhs, lambda e: Equation(lhs, e)
        case Disjunction(equations):
            for i, eq in enumerate(equations):
                for expr, rebuild in _slots(eq):

                    def put(
                        e: sp.Expr, i: int = i, rebuild: Callable[..., MathObject] = rebuild
                    ) -> MathObject:
                        new = rebuild(e)
                        assert isinstance(new, Equation)
                        return Disjunction((*equations[:i], new, *equations[i + 1 :]))

                    yield expr, put


def repairs(obj: MathObject) -> Iterator[Repair]:
    slots = list(_slots(obj))
    for edits in PASSES:
        for expr, rebuild in slots:
            for kind, repaired, term, fixed in edits(expr):
                yield Repair(kind, rebuild(repaired), term, ordered(evaluate(fixed)))
