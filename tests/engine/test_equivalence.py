import pytest
import sympy as sp
from hypothesis import given, settings
from hypothesis import strategies as st

from stepcheck.engine.equivalence import expressions_equivalent, real_value
from stepcheck.engine.model import Verdict
from stepcheck.engine.parse import parse_math

x = sp.Symbol("x", real=True)


def verdict(a: str, b: str) -> Verdict:
    return expressions_equivalent(_expr(a), _expr(b), x).verdict


def _expr(latex: str) -> sp.Expr:
    obj = parse_math(latex)
    assert hasattr(obj, "expr")
    return obj.expr  # type: ignore[no-any-return]


@pytest.mark.parametrize(
    ("a", "b"),
    [
        ("(x+1)^2", "x^2+2x+1"),
        (r"\frac{x^2-1}{x-1}", "x+1"),  # equal wherever both are defined
        (r"\frac{1}{2}x", r"\frac{x}{2}"),
        (r"\frac{1}{x} + \frac{1}{x+1}", r"\frac{2x+1}{x(x+1)}"),
        ("2^{x+1}", r"2 \cdot 2^x"),
        (r"\sqrt{8}", r"2\sqrt{2}"),
        (r"\sqrt{x^2}", "|x|"),
        ("|x|^2", "x^2"),
        (r"\sqrt{4x^2}", "2|x|"),
        ("x^{-2}", r"\frac{1}{x^2}"),
        (r"(x^{3})^{2}", "x^6"),
        ("2+3", "5"),
        ("-(x-3)", "-x+3"),
        (r"\frac{6}{4}", "1.5"),
    ],
)
def test_equivalent(a: str, b: str) -> None:
    assert verdict(a, b) == Verdict.EQUIVALENT


@pytest.mark.parametrize(
    ("a", "b"),
    [
        ("(x+1)^2", "x^2+1"),  # classic "freshman's dream"
        ("-(x-3)", "-x-3"),  # sign error
        ("2(x+3)", "2x+3"),  # partial distribution
        (r"\sqrt{x^2}", "x"),  # wrong for negative x
        (r"\frac{x+2}{2}", "x+1"),  # wrong "cancellation"
        (r"\frac{x+2}{x}", "2"),
        ("x^2 x^3", "x^6"),
        (r"\sqrt{x^2+9}", "x+3"),
        ("2^{x+1}", "2^x + 2"),
        ("2+3", "6"),
    ],
)
def test_not_equivalent(a: str, b: str) -> None:
    assert verdict(a, b) == Verdict.NOT_EQUIVALENT


def test_regression_denested_radical_is_equivalent() -> None:
    # Bug: cancel() treated sqrt(2) as a plain coefficient and called this NOT_EQUIVALENT.
    a = sp.sqrt(2) + sp.sqrt(3)
    b = sp.sqrt(5 + 2 * sp.sqrt(6))
    assert expressions_equivalent(a, b, x).verdict == Verdict.EQUIVALENT
    assert expressions_equivalent(x * a, x * b, x).verdict == Verdict.EQUIVALENT


def test_not_equivalent_always_has_counterexample() -> None:
    pairs = [("(x+1)^2", "x^2+1"), (r"\sqrt{x^2}", "x"), ("2+3", "6"), (r"\sqrt{8}", "3")]
    for a, b in pairs:
        result = expressions_equivalent(_expr(a), _expr(b), x)
        assert result.verdict == Verdict.NOT_EQUIVALENT
        assert result.counterexample is not None


def test_counterexample_is_a_real_disagreement() -> None:
    a, b = _expr(r"\sqrt{x^2}"), x
    result = expressions_equivalent(a, b, x)
    p = result.counterexample
    assert p is not None and p < 0
    assert a.subs(x, p) != b.subs(x, p)


def test_rational_counterexample_avoids_poles() -> None:
    a, b = _expr(r"\frac{1}{x}"), _expr(r"\frac{2}{x}")
    p = expressions_equivalent(a, b, x).counterexample
    assert p is not None and p != 0


def test_unknown_when_no_common_domain() -> None:
    # sqrt(x - 100) and sqrt(-x - 100) are never both real: nothing to compare.
    result = expressions_equivalent(sp.sqrt(x - 100), sp.sqrt(-x - 100), x)
    assert result.verdict == Verdict.UNKNOWN


def test_sqrt_squared_counts_as_equivalent_on_shared_domain() -> None:
    # (sqrt x)^2 = x for x >= 0, which is where the left side is defined.
    assert expressions_equivalent(sp.sqrt(x) ** 2, x, x).verdict == Verdict.EQUIVALENT


def test_real_value_rejects_complex_and_undefined() -> None:
    assert real_value(sp.sqrt(x), x, sp.Integer(-4)) is None
    assert real_value(1 / x, x, sp.Integer(0)) is None
    assert real_value(sp.sqrt(x), x, sp.Integer(4)) == 2


# --- Property tests: correct rewrites must NEVER be flagged (false-positive guard) ----------

_leaves = st.one_of(st.just(x), st.integers(-9, 9).map(sp.Integer))


def _combine(children: st.SearchStrategy[sp.Expr]) -> st.SearchStrategy[sp.Expr]:
    return st.one_of(
        st.tuples(children, children).map(lambda t: t[0] + t[1]),
        st.tuples(children, children).map(lambda t: t[0] - t[1]),
        st.tuples(children, children).map(lambda t: t[0] * t[1]),
        st.tuples(children, st.integers(2, 3)).map(lambda t: t[0] ** t[1]),
    )


polynomials = st.recursive(_leaves, _combine, max_leaves=8)
rational_functions = (
    st.tuples(polynomials, polynomials).filter(lambda t: t[1] != 0).map(lambda t: t[0] / t[1])
)


@settings(max_examples=150, deadline=None)
@given(polynomials)
def test_polynomial_rewrites_are_equivalent(e: sp.Expr) -> None:
    for rewritten in (sp.expand(e), sp.factor(e)):
        assert expressions_equivalent(e, rewritten, x).verdict == Verdict.EQUIVALENT


@settings(max_examples=100, deadline=None)
@given(rational_functions)
def test_rational_rewrites_are_equivalent(e: sp.Expr) -> None:
    for rewritten in (sp.together(e), sp.cancel(e), sp.apart(e, x)):
        assert expressions_equivalent(e, rewritten, x).verdict == Verdict.EQUIVALENT


@settings(max_examples=100, deadline=None)
@given(polynomials)
def test_abs_rewrites_are_equivalent_numerically(e: sp.Expr) -> None:
    # Abs is not a rational function, so this exercises the numeric path.
    result = expressions_equivalent(sp.Abs(e), sp.Abs(sp.expand(e)), x)
    assert result.verdict == Verdict.EQUIVALENT


@settings(max_examples=100, deadline=None)
@given(polynomials)
def test_off_by_one_is_always_caught(e: sp.Expr) -> None:
    assert expressions_equivalent(e, sp.expand(e) + 1, x).verdict == Verdict.NOT_EQUIVALENT
    assert expressions_equivalent(sp.Abs(e), sp.Abs(e) + 1, x).verdict == Verdict.NOT_EQUIVALENT


def test_counterexample_prefers_small_integers() -> None:
    result = expressions_equivalent(_expr("(x+3)^2"), _expr("x^2+9"), x)
    assert result.counterexample == 1


def test_friendly_point_agreement_is_not_trusted() -> None:
    # x^3 - x and 0 agree at 0, 1 and -1 but are clearly different.
    assert expressions_equivalent(x**3 - x, sp.Integer(0), x).verdict == Verdict.NOT_EQUIVALENT
