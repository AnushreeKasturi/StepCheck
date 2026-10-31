"""Template explanations, one per error type. LaTeX is wrapped in $...$ for KaTeX."""

from dataclasses import dataclass

import sympy as sp

from stepcheck.engine.model import ErrorType, StepCheck, Verdict


@dataclass(frozen=True)
class Finding:
    """One explanation of a wrong step, before it is turned into a Diagnosis."""

    error_type: ErrorType
    term: str | None = None  # LaTeX: where the mistake happened
    wrong: str | None = None  # LaTeX: what the student turned it into
    right: str | None = None  # LaTeX: what it should have become
    expected: str | None = None  # LaTeX: the corrected line
    values: str | None = None  # e.g. "x = 0", for solution-related errors


def format_values(s: sp.Set, x: sp.Symbol) -> str:
    if isinstance(s, sp.FiniteSet):
        return " and ".join(f"${sp.latex(x)} = {sp.latex(v)}$" for v in s)
    return f"${sp.latex(x)} \\in {sp.latex(s)}$"


def _template(f: Finding) -> str:
    match f.error_type:
        case ErrorType.SIGN_ERROR if f.wrong and f.term == f.right:
            return f"Sign error: you used ${f.wrong}$ instead of ${f.right}$."
        case ErrorType.SIGN_ERROR if f.wrong and f.term != f.wrong:
            return f"Sign error: ${f.term}$ should become ${f.right}$, but you wrote ${f.wrong}$."
        case ErrorType.SIGN_ERROR:
            return f"Sign error: check the sign of ${f.term}$; it should be ${f.right}$."
        case ErrorType.PARTIAL_DISTRIBUTION:
            return (
                f"When expanding ${f.term}$, multiply every term inside the brackets: "
                f"${f.term} = {f.right}$, not ${f.wrong}$."
            )
        case ErrorType.POWER_OF_SUM:
            return (
                "A power of a sum is not the sum of the powers (you also need the middle "
                f"terms): ${f.term} = {f.right}$, not ${f.wrong}$."
            )
        case ErrorType.INVALID_CANCELLATION:
            return (
                "You can only cancel factors that multiply the whole numerator and "
                f"denominator, not single terms: ${f.term} = {f.right}$, not ${f.wrong}$."
            )
        case ErrorType.MOVED_TERM_NO_SIGN_CHANGE:
            return (
                f"When ${f.term}$ moves to the other side of the equals sign, its sign "
                f"changes: it becomes ${f.right}$."
            )
        case ErrorType.INVERSE_OPERATION:
            return f"To undo multiplying by ${f.right}$, divide both sides by ${f.right}$."
        case ErrorType.ARITHMETIC_SLIP:
            return f"Arithmetic slip: ${f.term}$ should be ${f.right}$."
        case ErrorType.DIVIDED_BY_ZERO_EXPRESSION:
            return (
                f"You divided both sides by ${f.term}$, which is zero when {f.values}, so that "
                "solution was lost. Move everything to one side and factor instead."
            )
        case ErrorType.MISSING_PLUS_MINUS:
            return "Taking a square root gives two answers, a positive and a negative one."
        case ErrorType.EXTRANEOUS_NOT_REJECTED:
            return (
                f"This step introduces {f.values}, which does not satisfy the original "
                "equation and is never rejected. Check every answer in the original equation."
            )
        case ErrorType.UNCLASSIFIED:
            detail = f" {f.values}" if f.values else ""
            return f"This line does not follow from the previous one.{detail}"


def explain(f: Finding, line: int) -> str:
    text = f"Line {line}: {_template(f)}"
    if f.expected:
        text += f" The line should be ${f.expected}$."
    return text


def unclassified_finding(check: StepCheck, x: sp.Symbol, values_at: str | None) -> Finding:
    """Fall back to what the equivalence check itself found."""
    detail: str | None
    match check.verdict:
        case Verdict.NOT_EQUIVALENT if values_at:
            detail = values_at
        case Verdict.LOST_SOLUTIONS:
            detail = f"It loses the solution {format_values(check.lost, x)}."
        case Verdict.LOST_AND_EXTRA:
            detail = (
                f"It changes the solutions: {format_values(check.lost, x)} no longer works, "
                f"and {format_values(check.extra, x)} is wrong."
            )
        case _:
            detail = None
    return Finding(ErrorType.UNCLASSIFIED, values=detail)
