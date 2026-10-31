import pytest
import sympy as sp

from stepcheck.engine.checker import check_solution
from stepcheck.engine.diagnose import STRONG, WEAK, match_strength, same_equation
from stepcheck.engine.model import Equation, ErrorType, Expression
from stepcheck.engine.parse import parse_math

x = sp.Symbol("x", real=True)
E = ErrorType

# (name, solution lines, expected error type, text the explanation must contain)
CASES = [
    (
        "minus before bracket",
        ["5 - (x - 3) = 2", "5 - x - 3 = 2", "x = 0"],
        E.SIGN_ERROR,
        "- x + 3",
    ),
    (
        "partial distribution",
        ["3(x + 4) = 21", "3x + 4 = 21", "3x = 17"],
        E.PARTIAL_DISTRIBUTION,
        "3 x + 12",
    ),
    (
        "negative coefficient distribution",
        ["-2(x - 1) = 8", "-2x - 2 = 8"],
        E.SIGN_ERROR,
        "- 2 x + 2",
    ),
    ("power of sum", ["(x + 3)^2", "x^2 + 9"], E.POWER_OF_SUM, "x^{2} + 6 x + 9"),
    ("power of sum plus other work", ["(x + 3)^2 - 2x", "x^2 - 2x + 9"], E.POWER_OF_SUM, "6 x"),
    ("square of difference", ["(x - 4)^2 = 0", "x^2 - 16 = 0"], E.POWER_OF_SUM, "8 x"),
    ("crossed out x", [r"\frac{x + 2}{x}", "2"], E.INVALID_CANCELLATION, r"\frac{2}{x}"),
    ("divided one term", [r"\frac{2x + 6}{2}", "x + 6"], E.INVALID_CANCELLATION, "x + 3"),
    (
        "moved without sign change",
        ["3x + 5 = 11", "3x = 11 + 5", "3x = 16"],
        E.MOVED_TERM_NO_SIGN_CHANGE,
        "-5",
    ),
    ("subtracted instead of divided", ["4x = 12", "x = 12 - 4"], E.INVERSE_OPERATION, "x = 3"),
    ("divided by positive", ["-2x = 6", "x = 3"], E.SIGN_ERROR, "x = -3"),
    (
        "arithmetic slip in equation",
        ["2x + 3 = 11", "2x = 9", "x = 4.5"],
        E.ARITHMETIC_SLIP,
        "2 x = 8",
    ),
    ("arithmetic slip in expression", ["4x + 3x - 2", "8x - 2"], E.ARITHMETIC_SLIP, "7 x - 2"),
    ("divided by x", ["2x^2 = 6x", "2x = 6", "x = 3"], E.DIVIDED_BY_ZERO_EXPRESSION, "x = 0"),
    ("missing plus minus", ["x^2 = 9", "x = 3"], E.MISSING_PLUS_MINUS, "x = -3"),
    (
        "missing plus minus, shifted",
        ["(x - 1)^2 = 16", "x - 1 = 4", "x = 5"],
        E.MISSING_PLUS_MINUS,
        "x - 1 = -4",
    ),
    (
        "wrong sign of a root",
        ["(x - 2)(x + 5) = 0", r"x = 2 \text{ or } x = 5"],
        E.SIGN_ERROR,
        "x = -5",
    ),
    (
        "extraneous root kept",
        [r"\sqrt{x+3} = x - 3", "x + 3 = (x-3)^2", "x^2 - 7x + 6 = 0", r"x = 1 \text{ or } x = 6"],
        E.EXTRANEOUS_NOT_REJECTED,
        "x = 1",
    ),
    ("no known pattern", ["x^2 + 1", "x^3"], E.UNCLASSIFIED, "at $x = 1$"),
]


@pytest.mark.parametrize(("name", "lines", "error_type", "text"), CASES, ids=[c[0] for c in CASES])
def test_diagnosis(name: str, lines: list[str], error_type: ErrorType, text: str) -> None:
    result = check_solution(lines)
    assert result.first_error == 2
    assert result.diagnosis is not None
    assert result.diagnosis.error_type == error_type
    assert result.diagnosis.line == 2
    assert text in result.diagnosis.explanation


@pytest.mark.parametrize(("name", "lines", "error_type", "text"), CASES, ids=[c[0] for c in CASES])
def test_suggested_correction_is_itself_correct(
    name: str, lines: list[str], error_type: ErrorType, text: str
) -> None:
    # Self-consistency: if we tell a student "the line should be X", X must pass the checker.
    diagnosis = check_solution(lines).diagnosis
    assert diagnosis is not None
    if diagnosis.expected is None:
        pytest.skip("no corrected line for this error type")
    corrected = check_solution([lines[0], diagnosis.expected])
    assert corrected.status != "error", diagnosis.expected


def test_minus_bracket_prefers_sign_error_over_moved_term() -> None:
    # Both explain the algebra, but only the sign error explains the *shape* of the line.
    diagnosis = check_solution(["5 - (x - 3) = 2", "5 - x - 3 = 2"]).diagnosis
    assert diagnosis is not None
    assert diagnosis.error_type == E.SIGN_ERROR
    assert E.MOVED_TERM_NO_SIGN_CHANGE in diagnosis.alternatives


def test_squaring_is_not_misdiagnosed_when_rejected_correctly() -> None:
    lines = [r"\sqrt{x+3} = x - 3", "x + 3 = (x-3)^2", r"x = 1 \text{ or } x = 6", "x = 6"]
    result = check_solution(lines)
    assert result.status == "correct"
    assert result.diagnosis is None


def test_no_diagnosis_for_correct_solutions() -> None:
    assert check_solution(["2x + 4 = 10", "2x = 6", "x = 3"]).diagnosis is None


def test_same_equation_up_to_scaling_and_moving_terms() -> None:
    def eq(latex: str) -> Equation:
        obj = parse_math(latex)
        assert isinstance(obj, Equation)
        return obj

    assert same_equation(eq("2x + 4 = 10"), eq("x + 2 = 5"), x)
    assert same_equation(eq("2x + 4 = 10"), eq("2x = 6"), x)
    assert same_equation(eq("2x + 4 = 10"), eq("x = 3"), x)  # 2x - 6 = 2(x - 3)
    assert not same_equation(eq("x = 3"), eq("x^2 = 9"), x)


def test_match_strength_side_by_side_beats_rearranged() -> None:
    target = parse_math("5 - x - 3 = 2")
    side_by_side = parse_math("5 + (-x) + (-3) = 2", evaluate=False)
    rearranged = parse_math("5 = 2 + x + 3", evaluate=False)
    assert match_strength(side_by_side, target, x) == STRONG
    assert match_strength(rearranged, target, x) == WEAK
    assert match_strength(Expression(x), target, x) is None
