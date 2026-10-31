"""Name the mistake in a wrong step: forward mutations, backward repairs, special cases.

A candidate matches when it is *equivalent* to the student's line, not identical to it, so a
mistake combined with correct simplification in the same line is still recognized.
"""

from collections.abc import Callable, Iterator

import sympy as sp

from stepcheck.engine.equivalence import expressions_equivalent
from stepcheck.engine.explain import Finding, explain, format_values, unclassified_finding
from stepcheck.engine.model import (
    Diagnosis,
    Disjunction,
    Equation,
    ErrorType,
    Expression,
    MathObject,
    Step,
    StepCheck,
    Verdict,
)
from stepcheck.engine.mutations import mutations
from stepcheck.engine.repairs import repairs
from stepcheck.engine.rewrite import evaluate, evaluate_math, math_to_latex, ordered, to_latex
from stepcheck.engine.solutions import compare_equations, real_solution_set

# When several explanations fit, prefer the most specific misconception.
PRIORITY = [
    ErrorType.MISSING_PLUS_MINUS,
    ErrorType.DIVIDED_BY_ZERO_EXPRESSION,
    ErrorType.POWER_OF_SUM,
    ErrorType.INVALID_CANCELLATION,
    ErrorType.PARTIAL_DISTRIBUTION,
    ErrorType.MOVED_TERM_NO_SIGN_CHANGE,
    ErrorType.INVERSE_OPERATION,
    ErrorType.SIGN_ERROR,
    ErrorType.ARITHMETIC_SLIP,
    ErrorType.EXTRANEOUS_NOT_REJECTED,  # only if no mistake explains the extra values better
    ErrorType.UNCLASSIFIED,
]
# Ranking of how well an explanation fits, best first:
STRONG = 0  # forward mutation that reproduces the student's line side by side
REPAIR = 1  # a one-symbol edit to the student's line makes it correct
WEAK = 2  # forward mutation that only matches after rearranging (can be a coincidence)


def _safe_evaluate(obj: MathObject) -> MathObject | None:
    try:
        value = evaluate_math(obj)
    except (ValueError, TypeError, ZeroDivisionError):
        return None
    exprs = {
        Expression: lambda o: [o.expr],
        Equation: lambda o: [o.lhs, o.rhs],
        Disjunction: lambda o: [e for eq in o.equations for e in (eq.lhs, eq.rhs)],
    }[type(value)](value)
    if any(e.has(sp.zoo, sp.nan, sp.oo, -sp.oo) for e in exprs):
        return None
    return value


def same_equation(a: Equation, b: Equation, x: sp.Symbol) -> bool:
    """Same equation up to a non-zero constant factor (and moving terms across '=').

    Stricter than equal solution sets: two different wrong linear equations often share
    their single solution by coincidence.
    """
    d1, d2 = a.lhs - a.rhs, b.lhs - b.rhs
    if d1 == 0 or d2 == 0:
        return bool(sp.simplify(d1) == 0 and sp.simplify(d2) == 0)
    ratio = sp.cancel(sp.together(d1 / d2))
    return ratio != 0 and not ratio.has(x) and ratio.is_finite is not False


def match_strength(candidate: MathObject, target: MathObject, x: sp.Symbol) -> int | None:
    """Does a mutated previous line reproduce what the student wrote, and how closely?

    STRONG: equivalent side by side (left to left, right to right), so the mistake also explains
    the *shape* of the student's line. WEAK: only the same equation after rearranging.
    """
    value = _safe_evaluate(candidate)
    if isinstance(value, Expression) and isinstance(target, Expression):
        return STRONG if _equivalent(value.expr, target.expr, x) else None
    if isinstance(value, Equation) and isinstance(target, Equation):
        if not same_equation(value, target, x):
            return None
        sides = _equivalent(value.lhs, target.lhs, x) and _equivalent(value.rhs, target.rhs, x)
        return STRONG if sides else WEAK
    return None


def _equivalent(a: sp.Expr, b: sp.Expr, x: sp.Symbol) -> bool:
    return expressions_equivalent(a, b, x).verdict == Verdict.EQUIVALENT


def _satisfies(obj: MathObject, value: sp.Expr, x: sp.Symbol) -> bool:
    """Cheap numeric test: does x = value solve the equation (or one of the "or" parts)?"""
    equations = obj.equations if isinstance(obj, Disjunction) else (obj,)
    for eq in equations:
        assert isinstance(eq, Equation)
        residual = sp.N((eq.lhs - eq.rhs).subs(x, value), 30)
        if residual.is_number and not residual.has(sp.zoo, sp.nan) and abs(residual) < 1e-20:
            return True
    return False


def step_validator(before: MathObject, x: sp.Symbol) -> Callable[[MathObject], bool]:
    """Returns a test for "would this repaired line have been a correct step?".

    For equations, the known solutions of the previous line are substituted first. That
    rejects almost every candidate without running solveset, the expensive part.
    """
    if isinstance(before, Expression):

        def valid_expression(after: MathObject) -> bool:
            value = _safe_evaluate(after)
            return isinstance(value, Expression) and _equivalent(before.expr, value.expr, x)

        return valid_expression

    solutions = real_solution_set(before, x) if isinstance(before, Equation | Disjunction) else None

    def valid_equation(after: MathObject) -> bool:
        value = _safe_evaluate(after)
        if not isinstance(value, Equation | Disjunction) or isinstance(before, Expression):
            return False
        if isinstance(solutions, sp.FiniteSet) and not all(
            _satisfies(value, s, x) for s in solutions
        ):
            return False
        return compare_equations(before, value, x).verdict == Verdict.EQUIVALENT

    return valid_equation


def divided_by_zero(
    before: Equation, after: Equation, lost: sp.Set, x: sp.Symbol
) -> Finding | None:
    """2x^2 = 6x -> 2x = 6: both sides were divided by a factor that vanishes at the lost values."""
    d1, d2 = before.lhs - before.rhs, after.lhs - after.rhs
    if d2 == 0 or not isinstance(lost, sp.FiniteSet):
        return None
    divisor = sp.factor(sp.cancel(sp.together(d1 / d2)))
    if not divisor.is_polynomial(x) or not divisor.has(x):
        return None  # dividing both sides by an expression leaves a polynomial ratio
    if any(divisor.subs(x, r) != 0 for r in lost):
        return None
    for side in (before.lhs, before.rhs):
        if side != 0 and sp.fraction(sp.cancel(side / divisor))[1].has(x):
            return None  # the student did not divide *both sides* by it
    return Finding(
        ErrorType.DIVIDED_BY_ZERO_EXPRESSION,
        term=to_latex(divisor),
        expected=f"{sp.latex(sp.factor(d1))} = 0",
        values=format_values(lost, x),
    )


def _findings(
    before: Step, after: Step, check: StepCheck, x: sp.Symbol
) -> Iterator[tuple[int, Finding]]:
    """Every explanation that fits, with its match strength."""
    a, b = before.parsed, after.parsed
    if a is None or b is None or before.raw is None or after.raw is None:
        return
    for m in mutations(before.raw):
        strength = match_strength(m.candidate, b, x)
        if strength is not None:
            yield (
                strength,
                Finding(
                    m.error_type,
                    term=to_latex(m.term),
                    wrong=to_latex(ordered(evaluate(m.wrong))),  # what the student ended up with
                    right=to_latex(m.right),
                    expected=math_to_latex(m.expected),
                ),
            )
    # Dividing by an expression only ever loses solutions; it cannot add any.
    if (
        check.verdict == Verdict.LOST_SOLUTIONS
        and isinstance(a, Equation)
        and isinstance(b, Equation)
    ):
        found = divided_by_zero(a, b, check.lost, x)
        if found:
            yield STRONG, found
    valid = step_validator(a, x)
    for r in repairs(after.raw):
        if valid(r.repaired):
            yield (
                REPAIR,
                Finding(
                    r.error_type,
                    term=to_latex(r.term),
                    right=to_latex(r.fixed),
                    expected=math_to_latex(r.repaired),
                ),
            )
    if check.verdict == Verdict.EXTRA_SOLUTIONS:
        # e.g. squared both sides (a valid step) and never rejected the extra root.
        yield WEAK, Finding(ErrorType.EXTRANEOUS_NOT_REJECTED, values=format_values(check.extra, x))


def _values_at(before: Step, after: Step, check: StepCheck, x: sp.Symbol) -> str | None:
    p = check.counterexample
    a, b = before.parsed, after.parsed
    if p is None or not isinstance(a, Expression) or not isinstance(b, Expression):
        return None
    va, vb = a.expr.subs(x, p), b.expr.subs(x, p)
    return (
        f"For example, at ${sp.latex(x)} = {sp.latex(p)}$ line {before.line_no} equals "
        f"${sp.latex(va)}$ but line {after.line_no} equals ${sp.latex(vb)}$."
    )


def diagnose(before: Step, after: Step, check: StepCheck, x: sp.Symbol) -> Diagnosis:
    ranked = sorted(
        _findings(before, after, check, x),
        key=lambda found: (found[0], PRIORITY.index(found[1].error_type)),
    )
    findings = [f for _, f in ranked]
    if findings:
        best = findings[0]
    else:
        best = unclassified_finding(check, x, _values_at(before, after, check, x))
    alternatives = tuple(
        dict.fromkeys(f.error_type for f in findings if f.error_type != best.error_type)
    )
    return Diagnosis(
        best.error_type,
        after.line_no,
        explain(best, after.line_no),
        term=best.term,
        expected=best.expected,
        alternatives=alternatives,
    )
