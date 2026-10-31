"""Core data types for the engine. Pure data, no logic."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Literal

import sympy as sp


@dataclass(frozen=True)
class Expression:
    expr: sp.Expr


@dataclass(frozen=True)
class Equation:
    lhs: sp.Expr
    rhs: sp.Expr


@dataclass(frozen=True)
class Disjunction:
    """Several equations joined by "or", e.g. `x = 2 or x = -3`, or `x = \\pm 3`."""

    equations: tuple[Equation, ...]


MathObject = Expression | Equation | Disjunction


@dataclass(frozen=True)
class Step:
    line_no: int  # 1-based, as the student sees it
    latex: str
    parsed: MathObject | None  # evaluated: used for checking
    parse_error: str | None = None
    raw: MathObject | None = None  # unevaluated, as the student wrote it: used for diagnosis


class Verdict(Enum):
    EQUIVALENT = "equivalent"
    NOT_EQUIVALENT = "not_equivalent"  # expressions with different values
    LOST_SOLUTIONS = "lost_solutions"  # e.g. divided both sides by x
    # e.g. squared both sides. A valid one-way step (=>) unless the extra values survive
    # to the final answer, so the checker decides whether it is an error.
    EXTRA_SOLUTIONS = "extra_solutions"
    REJECTED_EXTRANEOUS = "rejected_extraneous"  # dropped values that never solved the problem
    LOST_AND_EXTRA = "lost_and_extra"  # solution set simply changed
    UNKNOWN = "unknown"  # engine could not decide; never reported as an error
    KIND_MISMATCH = "kind_mismatch"  # expression <-> equation; not blamed on the student
    UNPARSEABLE = "unparseable"


# Verdicts that are wrong regardless of what the rest of the solution does.
ERROR_VERDICTS = frozenset(
    {
        Verdict.NOT_EQUIVALENT,
        Verdict.LOST_SOLUTIONS,
        Verdict.LOST_AND_EXTRA,
    }
)

Method = Literal["exact", "numeric", "solveset", "none"]


@dataclass(frozen=True)
class StepCheck:
    """Result of checking that line `to_line` follows from line `from_line`."""

    from_line: int
    to_line: int
    verdict: Verdict
    method: Method
    counterexample: sp.Expr | None = None  # value of the variable where the expressions differ
    lost: sp.Set = field(default=sp.S.EmptySet)  # solutions of from_line missing in to_line
    extra: sp.Set = field(default=sp.S.EmptySet)  # solutions of to_line not in from_line
    note: str | None = None
    is_error: bool = False  # set by the checker: an EXTRA step's fate depends on later lines


class ErrorType(Enum):
    SIGN_ERROR = "sign_error"
    PARTIAL_DISTRIBUTION = "partial_distribution"  # a(b + c) -> ab + c
    POWER_OF_SUM = "power_of_sum"  # (a + b)^2 -> a^2 + b^2
    INVALID_CANCELLATION = "invalid_cancellation"  # (x + 2)/x -> 2
    MOVED_TERM_NO_SIGN_CHANGE = "moved_term_no_sign_change"
    INVERSE_OPERATION = "inverse_operation"  # 2x = 6 -> x = 6 - 2
    ARITHMETIC_SLIP = "arithmetic_slip"
    DIVIDED_BY_ZERO_EXPRESSION = "divided_by_zero_expression"  # divided both sides by x
    MISSING_PLUS_MINUS = "missing_plus_minus"  # x^2 = 4 -> x = 2
    EXTRANEOUS_NOT_REJECTED = "extraneous_not_rejected"
    UNCLASSIFIED = "unclassified"


@dataclass(frozen=True)
class Diagnosis:
    error_type: ErrorType
    line: int
    explanation: str
    term: str | None = None  # LaTeX of the part of the work where the mistake happened
    expected: str | None = None  # LaTeX of what the line should have been
    alternatives: tuple[ErrorType, ...] = ()  # other explanations that also fit


Status = Literal["correct", "error", "incomplete", "unverified"]


@dataclass(frozen=True)
class CheckResult:
    steps: tuple[Step, ...]
    checks: tuple[StepCheck, ...]
    status: Status
    first_error: int | None  # line number of the first wrong line, if it can be pinned down
    # First line vs last line; only run when some step could not be verified.
    final_check: StepCheck | None = None
    diagnosis: Diagnosis | None = None
