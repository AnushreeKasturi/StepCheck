"""Check a whole solution: every consecutive pair, then localize the first error."""

from collections.abc import Sequence
from dataclasses import replace

import sympy as sp

from stepcheck.engine.diagnose import diagnose
from stepcheck.engine.equivalence import expressions_equivalent
from stepcheck.engine.model import (
    ERROR_VERDICTS,
    CheckResult,
    Disjunction,
    Equation,
    Expression,
    MathObject,
    Step,
    StepCheck,
    Verdict,
)
from stepcheck.engine.parse import free_symbols, parse_line
from stepcheck.engine.solutions import compare_equations, outside, real_solution_set

_UNVERIFIED = frozenset({Verdict.UNKNOWN, Verdict.KIND_MISMATCH, Verdict.UNPARSEABLE})


class UnsupportedInput(ValueError):
    """The solution as a whole is outside v1 scope (e.g. more than one variable)."""


def solution_variable(steps: Sequence[Step]) -> sp.Symbol:
    symbols = set().union(*(free_symbols(s.parsed) for s in steps if s.parsed is not None))
    if len(symbols) > 1:
        names = ", ".join(sorted(s.name for s in symbols))
        raise UnsupportedInput(f"only one variable is supported (found {names})")
    return symbols.pop() if symbols else sp.Symbol("x", real=True)


def check_pair(before: Step, after: Step, x: sp.Symbol) -> StepCheck:
    a, b = before.parsed, after.parsed
    lines = (before.line_no, after.line_no)
    if isinstance(a, Expression) and isinstance(b, Expression):
        r = expressions_equivalent(a.expr, b.expr, x)
        return StepCheck(
            *lines,
            r.verdict,
            r.method,
            counterexample=r.counterexample,
            is_error=r.verdict in ERROR_VERDICTS,
        )
    if isinstance(a, Equation | Disjunction) and isinstance(b, Equation | Disjunction):
        c = compare_equations(a, b, x)
        return StepCheck(
            *lines,
            c.verdict,
            "solveset",
            lost=c.lost,
            extra=c.extra,
            is_error=c.verdict in ERROR_VERDICTS,
        )
    return StepCheck(*lines, Verdict.KIND_MISMATCH, "none", note="expression vs. equation")


def check_all_pairs(steps: Sequence[Step], x: sp.Symbol) -> list[StepCheck]:
    """Check each line against the previous *readable* line (unreadable lines are bridged)."""
    checks: list[StepCheck] = []
    previous: Step | None = None
    for step in steps:
        if step.parsed is None:
            if previous is not None:
                checks.append(
                    StepCheck(
                        previous.line_no,
                        step.line_no,
                        Verdict.UNPARSEABLE,
                        "none",
                        note=step.parse_error,
                    )
                )
            continue
        if previous is not None:
            checks.append(check_pair(previous, step, x))
        previous = step
    return checks


def is_solved_form(obj: MathObject, x: sp.Symbol) -> bool:
    """`x = c`, `c = x`, an "or" of those, or a constant equation like `0 = 0` / `1 = 0`."""
    match obj:
        case Disjunction(equations):
            return all(is_solved_form(e, x) for e in equations)
        case Equation(lhs, rhs):
            if not (lhs.has(x) or rhs.has(x)):
                return True
            return (lhs == x and not rhs.has(x)) or (rhs == x and not lhs.has(x))
        case _:
            return False


def _solutions_of(step: Step, x: sp.Symbol) -> sp.Set | None:
    if isinstance(step.parsed, Equation | Disjunction):
        return real_solution_set(step.parsed, x)
    return None


def allow_rejected_extraneous(check: StepCheck, problem: sp.Set, x: sp.Symbol) -> StepCheck:
    """Dropping values that never solved the original problem is correct, not a lost solution."""
    if check.verdict != Verdict.LOST_SOLUTIONS or not isinstance(check.lost, sp.FiniteSet):
        return check
    if outside(check.lost, problem, x) != check.lost:
        return check  # at least one dropped value is a real solution
    return replace(check, verdict=Verdict.REJECTED_EXTRANEOUS, is_error=False)


def flag_surviving_extraneous(
    checks: list[StepCheck], final: Step, problem: sp.Set, x: sp.Symbol
) -> list[StepCheck]:
    """Extra values from a one-way step (e.g. squaring) are wrong only if they are never rejected.

    If some reach the final answer and nothing else is wrong, blame the step that introduced
    them. If another step is wrong (e.g. rejecting the real root instead of the fake one),
    that step is the actual mistake.
    """
    if any(c.is_error for c in checks):
        return checks
    final_solutions = _solutions_of(final, x)
    if final_solutions is None:
        return checks
    surviving = outside(final_solutions, problem, x)
    if surviving is None or surviving.is_empty:
        return checks
    for i, check in enumerate(checks):
        if check.verdict == Verdict.EXTRA_SOLUTIONS and outside(surviving, check.extra, x) != (
            surviving
        ):
            note = f"introduces {surviving}, which is never rejected"
            return [*checks[:i], replace(check, is_error=True, note=note), *checks[i + 1 :]]
    return checks


def check_solution(lines: Sequence[str]) -> CheckResult:
    steps = tuple(parse_line(latex, i) for i, latex in enumerate(lines, start=1))
    x = solution_variable(steps)
    pair_checks = check_all_pairs(steps, x)
    readable = [s for s in steps if s.parsed is not None]

    problem = _solutions_of(readable[0], x) if readable else None
    if problem is not None:
        pair_checks = [allow_rejected_extraneous(c, problem, x) for c in pair_checks]
        pair_checks = flag_surviving_extraneous(pair_checks, readable[-1], problem, x)
    checks = tuple(pair_checks)

    first_error = next((c for c in checks if c.is_error), None)
    if first_error is not None:
        before, after = steps[first_error.from_line - 1], steps[first_error.to_line - 1]
        diagnosis = diagnose(before, after, first_error, x)
        return CheckResult(steps, checks, "error", first_error.to_line, diagnosis=diagnosis)
    if len(readable) < 2:
        return CheckResult(steps, checks, "unverified", None)

    unverified = [c for c in checks if c.verdict in _UNVERIFIED]
    if unverified or len(readable) < len(steps):
        # Some step could not be checked, so check the end result against the problem directly.
        final = check_pair(readable[0], readable[-1], x)
        # Unlike a single step, extra values in the *final answer* are always wrong.
        if final.is_error or final.verdict == Verdict.EXTRA_SOLUTIONS:
            final = replace(final, is_error=True)
            # Every verified step was fine, so the mistake hides in an unverified one.
            suspect = unverified[0].to_line if len(unverified) == 1 else None
            return CheckResult(steps, checks, "error", suspect, final)
        return CheckResult(steps, checks, "unverified", None, final)

    first, last = readable[0].parsed, readable[-1].parsed
    assert first is not None and last is not None
    if not isinstance(first, Expression) and not is_solved_form(last, x):
        return CheckResult(steps, checks, "incomplete", None)
    return CheckResult(steps, checks, "correct", None)
