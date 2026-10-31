import pytest
import sympy as sp

from stepcheck.engine.model import Disjunction, Equation, Verdict
from stepcheck.engine.parse import parse_math
from stepcheck.engine.solutions import compare_equations, real_solution_set

x = sp.Symbol("x", real=True)


def eq(latex: str) -> Equation | Disjunction:
    obj = parse_math(latex)
    assert isinstance(obj, Equation | Disjunction)
    return obj


def compare(a: str, b: str) -> tuple[Verdict, sp.Set, sp.Set]:
    r = compare_equations(eq(a), eq(b), x)
    return r.verdict, r.lost, r.extra


@pytest.mark.parametrize(
    ("a", "b"),
    [
        ("2x + 3 = 7", "2x = 4"),
        ("2x = 4", "x = 2"),
        ("3(x-1) = 6", "x - 1 = 2"),
        ("x^2 - 5x + 6 = 0", "(x-2)(x-3) = 0"),
        ("(x-2)(x-3) = 0", r"x = 2 \text{ or } x = 3"),
        ("x^2 = 2", r"x = \pm\sqrt{2}"),
        ("x^2 - x - 1 = 0", r"x = \frac{1 \pm \sqrt{5}}{2}"),
        (r"x^2 = 8", r"x = \pm 2\sqrt{2}"),  # same numbers, different form
        ("x + 1 = x + 1", "0 = 0"),  # identity: all reals
        ("x + 1 = x", "1 = 0"),  # contradiction: no solution
        (r"\frac{x}{2} = 3", "x = 6"),
        ("0.5x = 3", "x = 6"),
    ],
)
def test_equivalent_steps(a: str, b: str) -> None:
    assert compare(a, b)[0] == Verdict.EQUIVALENT


def test_lost_solution_by_dividing_by_x() -> None:
    verdict, lost, extra = compare("x^2 = 3x", "x = 3")
    assert verdict == Verdict.LOST_SOLUTIONS
    assert lost == sp.FiniteSet(0)
    assert extra.is_empty


def test_lost_negative_root() -> None:
    verdict, lost, _ = compare("x^2 = 4", "x = 2")
    assert verdict == Verdict.LOST_SOLUTIONS
    assert lost == sp.FiniteSet(-2)


def test_lost_abs_branch() -> None:
    assert compare("|x| = 3", "x = 3")[:2] == (Verdict.LOST_SOLUTIONS, sp.FiniteSet(-3))


def test_extra_solution_by_squaring() -> None:
    # sqrt(x) = x - 2 has only x = 4; squaring introduces x = 1.
    verdict, lost, extra = compare(r"\sqrt{x} = x - 2", "x = (x-2)^2")
    assert verdict == Verdict.EXTRA_SOLUTIONS
    assert extra == sp.FiniteSet(1)
    assert lost.is_empty


def test_extraneous_root_from_cancelling_denominator() -> None:
    # (x^2-1)/(x-1) = 2 has NO solution: x = 1 makes the denominator zero.
    verdict, _, extra = compare(r"\frac{x^2-1}{x-1} = 2", "x + 1 = 2")
    assert verdict == Verdict.EXTRA_SOLUTIONS
    assert extra == sp.FiniteSet(1)


def test_plain_wrong_answer() -> None:
    verdict, lost, extra = compare("2x = 4", "x = 3")
    assert verdict == Verdict.LOST_AND_EXTRA
    assert (lost, extra) == (sp.FiniteSet(2), sp.FiniteSet(3))


def test_infinite_sets_use_set_difference() -> None:
    # |x| = x holds exactly for x >= 0; x = x holds everywhere.
    verdict, _, extra = compare("|x| = x", "x = x")
    assert verdict == Verdict.EXTRA_SOLUTIONS
    assert extra == sp.Interval.open(-sp.oo, 0)


def test_rounded_answer_is_flagged() -> None:
    # Known v1 limitation: decimal approximations are treated as exact values.
    assert compare("x^2 = 2", "x = 1.414")[0] == Verdict.LOST_AND_EXTRA


def test_unsolvable_is_unknown_not_error() -> None:
    assert real_solution_set(eq("2^x = x + 3"), x) is None
    assert compare("2^x = x + 3", "x = 1")[0] == Verdict.UNKNOWN


def test_known_limitation_x_over_x_is_simplified_at_parse_time() -> None:
    # SymPy turns x/x into 1 while parsing, so the x != 0 restriction is invisible.
    # Documented rather than fixed in v1; this test pins the behaviour.
    assert real_solution_set(eq(r"\frac{x}{x} = 1"), x) == sp.S.Reals
