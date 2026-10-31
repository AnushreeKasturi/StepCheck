"""Forward search: apply known student mistakes ("mal-rules") to the previous line.

If a mutated version of line k-1 is equivalent to what the student wrote on line k, that
mutation names the misconception. Each rule also produces the *correct* version of the same
rewrite, which tells the student what they should have written.

All rules work on the unevaluated tree, so they see the student's literal structure.
"""

from collections.abc import Callable, Iterator
from dataclasses import dataclass

import sympy as sp

from stepcheck.engine.model import Disjunction, Equation, ErrorType, Expression, MathObject
from stepcheck.engine.rewrite import (
    evaluate,
    negate,
    nodes,
    ordered,
    replace_at,
    terms,
    unevaluated,
)


@dataclass(frozen=True)
class Mutation:
    error_type: ErrorType
    candidate: MathObject  # line k-1 with the mistake applied
    expected: MathObject  # line k-1 with the same rewrite done correctly
    term: sp.Expr  # the part of line k-1 that was rewritten
    wrong: sp.Expr  # what the mistake turned `term` into
    right: sp.Expr  # what `term` should have become


# (error type, wrong rewrite of the node, correct rewrite of the node)
NodeRewrite = tuple[ErrorType, sp.Expr, sp.Expr]


def _add(*args: sp.Expr) -> sp.Expr:
    return args[0] if len(args) == 1 else unevaluated(sp.Add, *args)


def _mul(*args: sp.Expr) -> sp.Expr:
    return args[0] if len(args) == 1 else unevaluated(sp.Mul, *args)


def _is_division(factor: sp.Expr) -> bool:
    return isinstance(factor, sp.Pow) and bool(factor.exp.is_negative)


def _cancelled_or_split(e: sp.Expr) -> sp.Expr:
    """The correct simplification of a fraction: cancel common factors, or if there are none,
    split it into terms ((x + 2)/x = 1 + 2/x)."""
    cancelled = sp.cancel(e)
    if cancelled != e or len(cancelled.free_symbols) != 1:
        return cancelled
    return sp.apart(cancelled)


def distribution_mistakes(node: sp.Expr) -> list[NodeRewrite]:
    """a(b + c) -> ab + c. With a = -1 that is a sign error; with a = 1/d it is dividing
    only one term of a numerator, i.e. an invalid cancellation."""
    if not isinstance(node, sp.Mul):
        return []
    sums = [a for a in node.args if isinstance(a, sp.Add)]
    if len(sums) != 1:
        return []
    (inner,) = sums
    factors = [a for a in node.args if a is not inner]
    coef = _mul(*factors)
    coef_value = evaluate(coef)
    right = ordered(sp.expand(evaluate(node)))
    if any(_is_division(f) for f in factors):
        kind = ErrorType.INVALID_CANCELLATION
    elif coef_value == -1:
        kind = ErrorType.SIGN_ERROR
    else:
        kind = ErrorType.PARTIAL_DISTRIBUTION

    ts = inner.args
    found: list[NodeRewrite] = []
    for i in range(len(ts)):
        multiplied = [_mul(coef, t) if j == i else t for j, t in enumerate(ts)]
        found.append((kind, _add(*multiplied), right))
    if coef_value.is_number and coef_value.is_negative and coef_value != -1:
        # -2(x - 1) -> -2x - 2: multiplied by 2 everywhere, but the minus only hit one term.
        size = -coef_value
        found.append(
            (ErrorType.SIGN_ERROR, _add(_mul(coef, ts[0]), *(_mul(size, t) for t in ts[1:])), right)
        )
    return found


def power_mistakes(node: sp.Expr) -> list[NodeRewrite]:
    """(a + b)^n -> a^n + b^n, and (a - b)^2 -> a^2 - b^2."""
    if not (isinstance(node, sp.Pow) and isinstance(node.base, sp.Add)):
        return []
    n = node.exp
    if not (n.is_Integer and n >= 2):
        return []
    ts = node.base.args
    right = ordered(sp.expand(evaluate(node)))
    found: list[NodeRewrite] = [
        (ErrorType.POWER_OF_SUM, _add(*(unevaluated(sp.Pow, t, n) for t in ts)), right)
    ]
    if any(evaluate(t).could_extract_minus_sign() for t in ts):
        signed = [
            negate(unevaluated(sp.Pow, negate(t), n))
            if evaluate(t).could_extract_minus_sign()
            else unevaluated(sp.Pow, t, n)
            for t in ts
        ]
        found.append((ErrorType.POWER_OF_SUM, _add(*signed), right))
    return found


def cancellation_mistakes(node: sp.Expr) -> list[NodeRewrite]:
    """(x + 2)/x -> 2 (crossing out a term), and (x + 2)/(x + 3) -> 2/3."""
    if not isinstance(node, sp.Mul):
        return []
    divisions = [f for f in node.args if _is_division(f) and f.exp == -1]
    numerators = [f for f in node.args if not _is_division(f)]
    if len(divisions) != 1 or len(numerators) != 1 or not isinstance(numerators[0], sp.Add):
        return []
    denominator, numerator = divisions[0].base, numerators[0]
    right = ordered(_cancelled_or_split(evaluate(node)))
    found: list[NodeRewrite] = []
    den_terms = terms(denominator)
    for i, t in enumerate(numerator.args):
        rest = [u for j, u in enumerate(numerator.args) if j != i]
        if evaluate(t) == evaluate(denominator):
            found.append((ErrorType.INVALID_CANCELLATION, _add(*rest), right))
            found.append((ErrorType.INVALID_CANCELLATION, _add(sp.Integer(1), *rest), right))
        shared = [u for u in den_terms if evaluate(u) == evaluate(t)]
        if shared and len(den_terms) > 1:
            den_rest = [u for u in den_terms if u is not shared[0]]
            wrong = _mul(_add(*rest), unevaluated(sp.Pow, _add(*den_rest), sp.Integer(-1)))
            found.append((ErrorType.INVALID_CANCELLATION, wrong, right))
    return found


NODE_RULES: list[Callable[[sp.Expr], list[NodeRewrite]]] = [
    distribution_mistakes,
    power_mistakes,
    cancellation_mistakes,
]


def _node_mutations(expr: sp.Expr, rebuild: Callable[[sp.Expr], MathObject]) -> Iterator[Mutation]:
    for path, node in nodes(expr):
        for rule in NODE_RULES:
            for kind, wrong, right in rule(node):
                yield Mutation(
                    kind,
                    rebuild(replace_at(expr, path, wrong)),
                    rebuild(replace_at(expr, path, right)),
                    node,
                    wrong,
                    right,
                )


def _orientations(
    eq: Equation,
) -> Iterator[tuple[sp.Expr, sp.Expr, Callable[[sp.Expr, sp.Expr], Equation]]]:
    """(side, other side, rebuild) for both sides, so each rule is written once."""
    yield eq.lhs, eq.rhs, lambda side, other: Equation(side, other)
    yield eq.rhs, eq.lhs, lambda side, other: Equation(other, side)


def _term_groups(ts: tuple[sp.Expr, ...]) -> Iterator[list[int]]:
    """Which terms a student moves together: each single term, all constants at once, all
    x-terms at once, or the whole side (x^2 + 6x = 7 -> x^2 + 6x + 7 = 0)."""
    seen: set[tuple[int, ...]] = set()
    constants = [i for i, t in enumerate(ts) if not evaluate(t).free_symbols]
    variables = [i for i in range(len(ts)) if i not in constants]
    for group in [*([i] for i in range(len(ts))), constants, variables, list(range(len(ts)))]:
        if group and tuple(group) not in seen:
            seen.add(tuple(group))
            yield group


def moved_term_mistakes(eq: Equation) -> Iterator[Mutation]:
    """3x + 5 = 11 -> 3x = 11 + 5: moved across '=' without changing sign."""
    for side, other, build in _orientations(eq):
        ts = terms(side)
        if side == 0:
            continue
        for group in _term_groups(ts):
            moved = _add(*(ts[i] for i in group))
            rest = [t for i, t in enumerate(ts) if i not in group]
            remaining = _add(*rest) if rest else sp.Integer(0)
            yield Mutation(
                ErrorType.MOVED_TERM_NO_SIGN_CHANGE,
                build(remaining, _add(other, moved)),
                build(remaining, _add(other, negate(moved))),
                moved,
                moved,
                negate(moved),
            )


def inverse_operation_mistakes(eq: Equation) -> Iterator[Mutation]:
    """2x = 6 -> x = 6 - 2 or x = 6 * 2 (wrong inverse), and -2x = 6 -> x = 3 (lost sign)."""
    for side, other, build in _orientations(eq):
        if not isinstance(side, sp.Mul):
            continue
        numeric = [f for f in side.args if evaluate(f).is_number]
        rest = [f for f in side.args if not evaluate(f).is_number]
        if not numeric or not rest:
            continue
        c = evaluate(_mul(*numeric))
        if c in (1, -1):
            continue
        unknown = _mul(*rest)
        expected = build(unknown, ordered(evaluate(other) / c))
        for wrong in (_add(other, negate(c)), _mul(other, c)):
            yield Mutation(
                ErrorType.INVERSE_OPERATION, build(unknown, wrong), expected, side, wrong, c
            )
        if c.is_negative:
            # -2x = 6 -> x = 3: divided by 2 instead of -2.
            wrong = _mul(other, unevaluated(sp.Pow, -c, sp.Integer(-1)))
            yield Mutation(ErrorType.SIGN_ERROR, build(unknown, wrong), expected, c, -c, c)


def plus_minus_mistakes(eq: Equation) -> Iterator[Mutation]:
    """x^2 = 4 -> x = 2: took only the positive square root."""
    for side, other, _ in _orientations(eq):
        if not (isinstance(side, sp.Pow) and side.exp == 2):
            continue
        root = unevaluated(sp.Pow, other, sp.Rational(1, 2))
        value = sp.sqrt(evaluate(other))
        both = Disjunction((Equation(side.base, value), Equation(side.base, -value)))
        yield Mutation(
            ErrorType.MISSING_PLUS_MINUS, Equation(side.base, root), both, side, root, root
        )


def mutations(obj: MathObject) -> Iterator[Mutation]:
    match obj:
        case Expression(expr):
            yield from _node_mutations(expr, Expression)
        case Equation(lhs, rhs):
            yield from _node_mutations(lhs, lambda e: Equation(e, rhs))
            yield from _node_mutations(rhs, lambda e: Equation(lhs, e))
            yield from moved_term_mistakes(obj)
            yield from inverse_operation_mistakes(obj)
            yield from plus_minus_mistakes(obj)
        case Disjunction():
            return
