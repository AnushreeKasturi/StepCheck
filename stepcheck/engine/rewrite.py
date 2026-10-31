"""Helpers for editing unevaluated expression trees (the student's literal structure)."""

from collections.abc import Iterator

import sympy as sp

from stepcheck.engine.model import Disjunction, Equation, Expression, MathObject

Path = tuple[int, ...]


def nodes(e: sp.Basic, path: Path = ()) -> Iterator[tuple[Path, sp.Basic]]:
    """Every subtree with its path (child indices from the root), parents before children."""
    yield path, e
    for i, arg in enumerate(e.args):
        yield from nodes(arg, (*path, i))


def replace_at(e: sp.Basic, path: Path, new: sp.Basic) -> sp.Basic:
    """Copy of `e` with the subtree at `path` replaced. Nothing else is evaluated."""
    if not path:
        return new
    i, rest = path[0], path[1:]
    args = list(e.args)
    args[i] = replace_at(args[i], rest, new)
    with sp.evaluate(False):
        return e.func(*args)


def unevaluated(func: type[sp.Basic], *args: sp.Basic) -> sp.Expr:
    with sp.evaluate(False):
        return func(*args)


def evaluate(e: sp.Basic) -> sp.Expr:
    return e.doit()


def flatten(e: sp.Basic) -> sp.Basic:
    """(a + b) + c -> a + b + c (same for products) without evaluating anything else.

    The LaTeX parser nests sums left to right; flat sums make "one term" mean what a student
    means by it. `-(x - 3)` is a product, so the bracket a student wrote is kept.
    """
    if not e.args:
        return e
    args: list[sp.Basic] = []
    for a in (flatten(a) for a in e.args):
        if isinstance(e, sp.Add | sp.Mul) and type(a) is type(e):
            args.extend(a.args)
        else:
            args.append(a)
    with sp.evaluate(False):
        return e.func(*args)


def _fold_numbers(e: sp.Basic) -> sp.Basic:
    """3*(-1) -> -3 and similar, keeping all other structure."""
    if not e.args:
        return e
    args = [_fold_numbers(a) for a in e.args]
    if isinstance(e, sp.Mul) and all(a.is_Number for a in args):
        return sp.Mul(*args)  # evaluates: plain number
    with sp.evaluate(False):
        return e.func(*args)


def to_latex(e: sp.Basic) -> str:
    """LaTeX of a (possibly unevaluated) expression, in the order the student wrote it."""
    return str(sp.latex(flatten(_fold_numbers(e)), order="none"))


def _degree(term: sp.Expr) -> int:
    if not term.free_symbols:
        return 0
    poly = term.as_poly()
    return int(poly.total_degree()) if poly is not None else 1


def ordered(e: sp.Basic) -> sp.Basic:
    """Fix the display order of a *computed* result: x^2 + 6x + 9, not 9 + x^2 + 6x.

    (to_latex keeps the order it is given, so the student's own terms stay in place.)
    """
    if not e.args:
        return e
    if isinstance(e, sp.Add):
        # Highest power first, like a textbook. SymPy alone would print -x + 3 as 3 - x.
        args = sorted(e.as_ordered_terms(), key=lambda t: -_degree(t))
    elif isinstance(e, sp.Mul):
        coeff, rest = e.as_coeff_Mul()  # keep -2 whole; as_ordered_factors gives -1, 2
        factors = rest.as_ordered_factors()
        args = factors if coeff == 1 else [coeff, *factors]
    else:
        args = e.args
    with sp.evaluate(False):
        return e.func(*(ordered(a) for a in args))


def negate(t: sp.Expr) -> sp.Expr:
    """-t without double negatives: -(-(x-3)) is (x-3), -(5) is -5."""
    if isinstance(t, sp.Mul) and t.args[0] == -1:
        return t.args[1] if len(t.args) == 2 else unevaluated(sp.Mul, *t.args[1:])
    if isinstance(t, sp.Mul) and t.args[0].is_Number:
        return unevaluated(sp.Mul, -t.args[0], *t.args[1:])  # -(-3x) is 3x
    if t.is_Number:
        return -t
    return unevaluated(sp.Mul, sp.Integer(-1), t)


def math_to_latex(obj: MathObject) -> str:
    match obj:
        case Expression(expr):
            return to_latex(expr)
        case Equation(lhs, rhs):
            return f"{to_latex(lhs)} = {to_latex(rhs)}"
        case Disjunction(equations):
            return r" \text{ or } ".join(math_to_latex(eq) for eq in equations)


def evaluate_math(obj: MathObject) -> MathObject:
    match obj:
        case Expression(expr):
            return Expression(evaluate(expr))
        case Equation(lhs, rhs):
            return Equation(evaluate(lhs), evaluate(rhs))
        case Disjunction(equations):
            return Disjunction(tuple(Equation(evaluate(e.lhs), evaluate(e.rhs)) for e in equations))


def terms(e: sp.Expr) -> tuple[sp.Expr, ...]:
    """Top-level terms of a sum (a single term if `e` is not a sum)."""
    return tuple(e.args) if isinstance(e, sp.Add) else (e,)


def is_negative(e: sp.Expr) -> bool:
    """Does the term carry a leading minus sign, as written?"""
    return bool(evaluate(e).could_extract_minus_sign())
