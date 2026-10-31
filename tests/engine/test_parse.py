import pytest
import sympy as sp

from stepcheck.engine.model import Disjunction, Equation, Expression
from stepcheck.engine.parse import ParseError, parse_line, parse_math, split_top_level

x = sp.Symbol("x", real=True)


def expr_of(latex: str) -> sp.Expr:
    obj = parse_math(latex)
    assert isinstance(obj, Expression)
    return obj.expr


@pytest.mark.parametrize(
    ("latex", "expected"),
    [
        ("2x", 2 * x),
        ("2(x+1)", 2 * (x + 1)),
        ("x(x+1)", x * (x + 1)),  # SymPy alone parses this as a function call
        ("3x(x-2)", 3 * x * (x - 2)),
        (r"2 \cdot x", 2 * x),
        (r"2 \times x", 2 * x),
        (r"\frac{x^2-1}{x-1}", (x**2 - 1) / (x - 1)),
        (r"\dfrac{1}{2}x", x / 2),
        (r"\left(x+1\right)^2", (x + 1) ** 2),
        (r"\left|x\right|", sp.Abs(x)),
        ("|x-3|", sp.Abs(x - 3)),
        (r"\sqrt{x^2}", sp.Abs(x)),  # x is real, so SymPy rewrites this itself
        ("0.5x", x / 2),  # floats become exact rationals
        ("$x+1$", x + 1),
        ("x \N{MINUS SIGN} 1", x - 1),  # unicode minus, as pasted from OCR/Word
        ("2^{x+1}", 2 ** (x + 1)),
    ],
)
def test_expressions(latex: str, expected: sp.Expr) -> None:
    assert sp.simplify(expr_of(latex) - expected) == 0


def test_symbols_are_real() -> None:
    (sym,) = expr_of("x+1").free_symbols
    assert sym.is_real


def test_equation() -> None:
    assert parse_math("3x - 2 = 7") == Equation(3 * x - 2, sp.Integer(7))


@pytest.mark.parametrize(
    "latex",
    [
        "x = 2, x = -3",
        "x = 2; x = -3",
        r"x = 2 \text{ or } x = -3",
        r"x=2 \quad\text{or}\quad x=-3",
        r"x = 2 \lor x = -3",
        "x = 2 or x = -3",
    ],
)
def test_disjunction_separators(latex: str) -> None:
    assert parse_math(latex) == Disjunction((Equation(x, 2), Equation(x, -3)))


def test_plus_minus() -> None:
    assert parse_math(r"x = \pm 3") == Disjunction((Equation(x, 3), Equation(x, -3)))


def test_plus_minus_signs_are_correlated() -> None:
    obj = parse_math(r"x = 1 \pm 2 \mp 0")
    assert isinstance(obj, Disjunction)
    assert [e.rhs for e in obj.equations] == [3, -1]


def test_continuation_line_is_expression() -> None:
    assert parse_math("= 2x + 2") == Expression(2 * x + 2)


def test_comma_inside_braces_is_not_a_separator() -> None:
    assert split_top_level("a,{b,c},d", ",") == ["a", "{b,c}", "d"]


@pytest.mark.parametrize(
    ("latex", "message"),
    [
        ("x < 3", "inequalities"),
        (r"x \le 3", "inequalities"),
        ("x = 1 = 2", "chained"),
        ("x = 2, 3", "must be an equation"),
        (r"\frac{1}{0}", "undefined"),
        ("", "empty"),
        (r"\frac{x}{", "could not parse"),
    ],
)
def test_parse_errors(latex: str, message: str) -> None:
    with pytest.raises(ParseError, match=message):
        parse_math(latex)


def test_parse_line_reports_error_instead_of_raising() -> None:
    step = parse_line("x <", 4)
    assert step.parsed is None
    assert step.line_no == 4
    assert step.parse_error


def test_multiple_variables_parse_fine() -> None:
    # The single-variable rule is enforced for the whole solution, in the checker.
    step = parse_line("x + y", 1)
    assert step.parsed is not None


def test_regression_raw_parse_works_as_first_call_in_fresh_process() -> None:
    # Bug: parse_latex lazily imports sympy.physics, which crashed when the first call was
    # made inside sp.evaluate(False). Needs a fresh interpreter to reproduce.
    import subprocess
    import sys

    code = "from stepcheck.engine.parse import parse_math; parse_math('5-(x-3)', evaluate=False)"
    subprocess.run([sys.executable, "-c", code], check=True)


def test_raw_parse_keeps_student_structure() -> None:
    raw = parse_math("5 - (x - 3)", evaluate=False)
    assert isinstance(raw, Expression)
    assert raw.expr != 8 - x  # not distributed
    assert raw.expr.doit() == 8 - x
