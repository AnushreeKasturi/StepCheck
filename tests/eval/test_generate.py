"""The generator's own correctness. The engine's *parser* is used here only to read the
generated LaTeX back; ground truth is computed with plain SymPy, never with the checker."""

import ast
import inspect
import random

import pytest
import sympy as sp

from stepcheck.engine.model import Disjunction, Equation, Expression, MathObject
from stepcheck.engine.parse import parse_math
from stepcheck.eval import generate

x = sp.Symbol("x", real=True)
SAMPLES = [generate.generate_one(random.Random(seed), error_rate=0.6) for seed in range(300)]


def meaning(obj: MathObject) -> sp.Basic:
    """Ground truth: real solution set for equations, expanded value for expressions."""
    match obj:
        case Expression(expr):
            return sp.expand(expr)
        case Equation(lhs, rhs):
            return sp.solveset(sp.Eq(lhs, rhs), x, sp.S.Reals)
        case Disjunction(equations):
            return sp.Union(*(meaning(e) for e in equations))


def test_generator_does_not_import_the_engine() -> None:
    # Otherwise the benchmark would measure the engine against its own mutation code.
    tree = ast.parse(inspect.getsource(generate))
    imported = [n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
    imported += [a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names]
    assert not [m for m in imported if m.startswith("stepcheck")]


def test_every_generated_line_parses() -> None:
    for sample in SAMPLES:
        for line in sample["lines"]:
            parse_math(line)


def test_lines_before_the_error_are_correct() -> None:
    for sample in SAMPLES:
        lines = sample["lines"]
        last_good = (sample["error_line"] or len(lines) + 1) - 1
        truth = meaning(parse_math(lines[0]))
        for line in lines[1:last_good]:
            assert meaning(parse_math(line)) == truth, (sample, line)


def test_the_error_line_really_changes_the_math() -> None:
    errors = [s for s in SAMPLES if s["error_line"]]
    assert len(errors) > 100
    for sample in errors:
        lines = sample["lines"]
        before = meaning(parse_math(lines[sample["error_line"] - 2]))
        assert meaning(parse_math(lines[sample["error_line"] - 1])) != before, sample


def test_correct_solutions_end_at_the_true_answer() -> None:
    for sample in SAMPLES:
        if sample["error_line"] is None:
            first, last = (parse_math(sample["lines"][i]) for i in (0, -1))
            assert meaning(first) == meaning(last), sample


def test_errors_are_never_on_the_first_line() -> None:
    assert all(s["error_line"] != 1 for s in SAMPLES)


def test_all_error_types_and_families_occur() -> None:
    assert {s["family"] for s in SAMPLES} == set(generate.FAMILIES)
    expected = {t for _, types in generate.FAMILIES.values() for t in types}
    assert {s["error_type"] for s in SAMPLES if s["error_type"]} == expected


def test_deterministic() -> None:
    assert generate.generate_one(random.Random(7), 0.7) == generate.generate_one(
        random.Random(7), 0.7
    )


@pytest.mark.parametrize(("value", "text"), [(3, "3"), (sp.Rational(-1, 2), r"-\frac{1}{2}")])
def test_num(value: sp.Rational, text: str) -> None:
    assert generate.num(value) == text


def test_join_writes_like_a_student() -> None:
    assert generate.join([(1, 2), (-1, 1), (0, 0)]) == "x^{2} - x"
    assert generate.join([(-3, 1), (5, 0)]) == "-3x + 5"
    assert generate.join([]) == "0"
