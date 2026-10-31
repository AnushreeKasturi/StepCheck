import pytest
import sympy as sp

from stepcheck.engine.checker import UnsupportedInput, check_solution, is_solved_form
from stepcheck.engine.model import Verdict
from stepcheck.engine.parse import parse_math

x = sp.Symbol("x", real=True)


def test_correct_linear_solution() -> None:
    result = check_solution(["3(x - 2) = 2x + 1", "3x - 6 = 2x + 1", "x - 6 = 1", "x = 7"])
    assert result.status == "correct"
    assert result.first_error is None
    assert all(c.verdict == Verdict.EQUIVALENT for c in result.checks)


def test_correct_quadratic_solution() -> None:
    result = check_solution(
        ["x^2 - 5x = -6", "x^2 - 5x + 6 = 0", "(x-2)(x-3) = 0", r"x = 2 \text{ or } x = 3"]
    )
    assert result.status == "correct"


def test_correct_simplification_with_continuation_lines() -> None:
    result = check_solution([r"\frac{x^2 - 9}{x + 3} + 2(x+1)", "= x - 3 + 2x + 2", "= 3x - 1"])
    assert result.status == "correct"


def test_sign_error_is_localized() -> None:
    result = check_solution(["5 - (x - 3) = 2", "5 - x - 3 = 2", "2 - x = 2", "x = 0"])
    assert result.status == "error"
    assert result.first_error == 2
    assert result.checks[0].verdict == Verdict.LOST_AND_EXTRA


def test_steps_after_the_error_are_still_checked() -> None:
    # Line 2 is wrong, but lines 3-4 follow correctly from it.
    result = check_solution(["5 - (x - 3) = 2", "5 - x - 3 = 2", "2 - x = 2", "x = 0"])
    assert [c.verdict for c in result.checks[1:]] == [Verdict.EQUIVALENT, Verdict.EQUIVALENT]


def test_expression_error_has_counterexample() -> None:
    result = check_solution(["(x+3)^2", "x^2 + 9"])
    assert result.first_error == 2
    assert result.checks[0].counterexample is not None


def test_lost_solution_by_dividing_by_x() -> None:
    result = check_solution(["x^2 = 5x", "x = 5"])
    assert result.first_error == 2
    assert result.checks[0].verdict == Verdict.LOST_SOLUTIONS
    assert result.checks[0].lost == sp.FiniteSet(0)


SQUARING_METHOD = [r"\sqrt{x+3} = x - 3", "x + 3 = (x-3)^2", "x^2 - 7x + 6 = 0"]


def test_regression_squaring_then_rejecting_extraneous_root_is_correct() -> None:
    # Bug: the squaring step (extra candidates) and the rejection step (lost x = 1)
    # were both flagged, although this is the textbook method.
    result = check_solution([*SQUARING_METHOD, r"x = 1 \text{ or } x = 6", "x = 6"])
    assert result.status == "correct"
    assert result.checks[0].verdict == Verdict.EXTRA_SOLUTIONS
    assert result.checks[-1].verdict == Verdict.REJECTED_EXTRANEOUS
    assert not any(c.is_error for c in result.checks)


def test_extraneous_root_never_rejected_is_blamed_on_the_squaring_step() -> None:
    result = check_solution([*SQUARING_METHOD, r"x = 1 \text{ or } x = 6"])
    assert result.status == "error"
    assert result.first_error == 2
    assert result.checks[0].extra == sp.FiniteSet(1)


def test_rejecting_a_real_solution_is_an_error() -> None:
    result = check_solution([*SQUARING_METHOD, r"x = 1 \text{ or } x = 6", "x = 1"])
    assert result.status == "error"
    assert result.first_error == 5  # x = 6 was the real solution
    assert result.checks[-1].verdict == Verdict.LOST_SOLUTIONS
    assert result.checks[-1].lost == sp.FiniteSet(6)
    assert not result.checks[0].is_error  # squaring itself was fine


def test_extraneous_root_from_cancelled_denominator() -> None:
    # (x^2-1)/(x-1) = 2 has no solution; x = 1 is introduced by cancelling and kept.
    result = check_solution([r"\frac{x^2-1}{x-1} = 2", "x + 1 = 2", "x = 1"])
    assert result.status == "error"
    assert result.first_error == 2


def test_incomplete_solution() -> None:
    result = check_solution(["2x + 4 = 10", "2x = 6"])
    assert result.status == "incomplete"


def test_unparseable_line_is_bridged() -> None:
    result = check_solution(["2x + 4 = 10", r"2x = \frac{6}{", "x = 3"])
    assert result.checks[0].verdict == Verdict.UNPARSEABLE
    assert (result.checks[1].from_line, result.checks[1].to_line) == (1, 3)
    assert result.checks[1].verdict == Verdict.EQUIVALENT
    assert result.status == "unverified"  # never "correct" if we could not read every line


def test_error_after_unparseable_line_is_still_found() -> None:
    result = check_solution(["2x + 4 = 10", r"2x = \frac{6}{", "x = 4"])
    assert result.status == "error"
    assert result.first_error == 3


def test_kind_mismatch_is_not_blamed_on_student() -> None:
    # Student writes the factored left side alone, a common shorthand.
    result = check_solution(["x^2 - 4 = 0", "(x-2)(x+2)", r"x = \pm 2"])
    assert result.status == "unverified"
    assert result.final_check is not None
    assert result.final_check.verdict == Verdict.EQUIVALENT


def test_error_hidden_behind_single_unverified_step_is_localized_by_final_check() -> None:
    result = check_solution(["x^2 - 4 = 0", "(x-2)(x+2)", "x = 2"])
    assert result.status == "error"
    assert result.first_error is None  # two unverified transitions: cannot pin it down
    assert result.final_check is not None
    assert result.final_check.verdict == Verdict.LOST_SOLUTIONS


def test_unknown_step_gets_resolved_by_final_check() -> None:
    result = check_solution(["2^x = 8", "2^x = 2^3", "x = 3"])
    assert result.status == "correct"  # solveset handles these directly


def test_single_line_is_unverified() -> None:
    assert check_solution(["x + 1 = 2"]).status == "unverified"


def test_multiple_variables_rejected() -> None:
    with pytest.raises(UnsupportedInput):
        check_solution(["x + 1 = 2", "y = 1"])


def test_pure_arithmetic() -> None:
    assert check_solution([r"\frac{1}{2} + \frac{1}{3}", r"\frac{5}{6}"]).status == "correct"
    assert check_solution([r"\frac{1}{2} + \frac{1}{3}", r"\frac{2}{5}"]).first_error == 2


@pytest.mark.parametrize(
    ("latex", "solved"),
    [
        ("x = 3", True),
        ("3 = x", True),
        (r"x = \pm 2", True),
        ("0 = 0", True),
        ("2x = 6", False),
        ("x = x", False),  # identity still contains x; the student should conclude "all reals"
        (r"x = 2 \text{ or } 2x = 6", False),
    ],
)
def test_is_solved_form(latex: str, solved: bool) -> None:
    assert is_solved_form(parse_math(latex), x) is solved
