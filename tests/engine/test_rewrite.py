import pytest
import sympy as sp

from stepcheck.engine.model import Expression
from stepcheck.engine.parse import parse_math
from stepcheck.engine.rewrite import (
    evaluate,
    flatten,
    negate,
    nodes,
    ordered,
    replace_at,
    terms,
    to_latex,
)

x = sp.Symbol("x", real=True)


def raw(latex: str) -> sp.Expr:
    obj = parse_math(latex, evaluate=False)
    assert isinstance(obj, Expression)
    return obj.expr


@pytest.mark.parametrize(
    ("latex", "shown"),
    [
        ("5 - (x - 3)", r"5 - \left(x - 3\right)"),  # the bracket the student wrote stays
        ("3x - 2 + 5", "3 x - 2 + 5"),  # not collected into 3x + 3
        (r"\frac{x + 2}{2}", r"\frac{x + 2}{2}"),
    ],
)
def test_to_latex_keeps_what_the_student_wrote(latex: str, shown: str) -> None:
    assert to_latex(raw(latex)) == shown


@pytest.mark.parametrize(
    ("value", "shown"),
    [
        (-x + 3, "- x + 3"),  # SymPy alone prints 3 - x
        (x**2 + 6 * x + 9, "x^{2} + 6 x + 9"),
        (-2 * x + 2, "- 2 x + 2"),  # not (-1) 2 x
    ],
)
def test_ordered_results_read_like_a_textbook(value: sp.Expr, shown: str) -> None:
    assert to_latex(ordered(value)) == shown


@pytest.mark.parametrize(
    ("term", "expected"),
    [(raw("-(x - 3)"), x - 3), (sp.Integer(5), -5), (raw("-3x"), 3 * x), (x, -x)],
)
def test_negate_avoids_double_negatives(term: sp.Expr, expected: sp.Expr) -> None:
    assert evaluate(negate(term)) == expected
    assert "- -" not in to_latex(negate(term))


def test_flatten_and_terms() -> None:
    nested = raw("3x - 2 + 5")
    assert len(terms(flatten(nested))) == 3


def test_replace_at_changes_only_that_node_and_does_not_evaluate() -> None:
    e = raw("2(x + 3) + 1")
    path = next(p for p, n in nodes(e) if n == 3)
    changed = replace_at(e, path, sp.Integer(4))
    assert evaluate(changed) == 2 * x + 9
    assert isinstance(changed, sp.Add) and len(changed.args) == 2  # bracket kept
