"""Synthetic dataset: step-by-step solutions with at most one known, injected mistake.

INDEPENDENCE: this module must not import stepcheck.engine. It has its own simulated student
who performs operations (expand, move a term, divide, factor, ...) and its own faulty versions
of them. If it reused the engine's mutation code, the benchmark would only measure that the
engine agrees with itself. SymPy is used only as ground truth: to confirm that an injected
mistake really changes the math (otherwise it is not a mistake and the sample is dropped).

Usage: python -m stepcheck.eval.generate --n 5000 --seed 0 --out data/synthetic.jsonl
"""

import argparse
import json
import random
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

import sympy as sp

x = sp.Symbol("x", real=True)

# Error labels: the same vocabulary as the engine's ErrorType values (plain strings on purpose).
SIGN = "sign_error"
PARTIAL = "partial_distribution"
POWER = "power_of_sum"
CANCEL = "invalid_cancellation"
MOVED = "moved_term_no_sign_change"
INVERSE = "inverse_operation"
SLIP = "arithmetic_slip"
DIV_ZERO = "divided_by_zero_expression"
PLUS_MINUS = "missing_plus_minus"


class Skip(Exception):
    """These parameters cannot produce a valid sample; draw new ones."""


# --- Rendering: how a student writes things ---------------------------------------------


def num(r: sp.Rational) -> str:
    r = sp.Rational(r)
    if r.q == 1:
        return str(r.p)
    sign = "-" if r < 0 else ""
    return f"{sign}\\frac{{{abs(r.p)}}}{{{r.q}}}"


def _body(c: sp.Rational, power: int) -> str:
    mag = abs(sp.Rational(c))
    if power == 0:
        return num(mag)
    var = "x" if power == 1 else f"x^{{{power}}}"
    return var if mag == 1 else f"{num(mag)}{var}"


def join(terms: list[tuple[sp.Rational, int]]) -> str:
    """Terms (coefficient, power of x) in the given order, e.g. [(3, 1), (-2, 0)] -> 3x - 2."""
    out = ""
    for c, p in terms:
        if c == 0:
            continue
        body = _body(c, p)
        if not out:
            out = f"-{body}" if c < 0 else body
        else:
            out += f" - {body}" if c < 0 else f" + {body}"
    return out or "0"


def poly(e: sp.Expr) -> str:
    coeffs = sp.Poly(sp.expand(e), x).all_coeffs()
    n = len(coeffs) - 1
    return join([(sp.Rational(c), n - i) for i, c in enumerate(coeffs)])


def coef_prefix(a: int) -> str:
    return {1: "", -1: "-"}.get(a, str(a))


def bracket(a: int, b: int) -> str:
    """a(x + b) as written by hand."""
    return f"{coef_prefix(a)}({join([(1, 1), (b, 0)])})"


def linear_coeffs(e: sp.Expr) -> tuple[sp.Rational, sp.Rational]:
    e = sp.expand(e)
    return sp.Rational(e.coeff(x, 1)), sp.Rational(e.coeff(x, 0))


# --- Sample assembly -------------------------------------------------------------------


@dataclass
class Sample:
    family: str
    lines: list[str] = field(default_factory=list)
    error_line: int | None = None
    error_type: str | None = None

    def add(self, line: str) -> None:
        self.lines.append(line)

    def add_error(self, line: str, error_type: str) -> None:
        self.lines.append(line)
        self.error_line, self.error_type = len(self.lines), error_type


def solutions(lhs: sp.Expr, rhs: sp.Expr) -> sp.Set:
    return sp.solveset(sp.Eq(lhs, rhs), x, sp.S.Reals)


def require_different_equations(
    wrong: tuple[sp.Expr, sp.Expr], right: tuple[sp.Expr, sp.Expr]
) -> None:
    if solutions(*wrong) == solutions(*right):
        raise Skip("injected mistake does not change the solutions")


def require_different_expressions(wrong: sp.Expr, right: sp.Expr) -> None:
    if sp.expand(wrong - right) == 0:
        raise Skip("injected mistake does not change the value")


Pick = Callable[[str], bool]  # "inject this error type here?"


# --- Generic continuations (also used after a mistake, so errors carry forward) ----------


def finish_linear(s: Sample, lhs: sp.Expr, rhs: sp.Expr, rng: random.Random, pick: Pick) -> None:
    """px + q = rx + t  ->  (p - r)x + q = t  ->  (p - r)x = t - q  ->  x = ..."""
    p, q = linear_coeffs(lhs)
    r, t = linear_coeffs(rhs)
    merge = rng.random() < 0.35  # do both moves in one line, like many students

    k = p - r
    if r != 0 and not merge:
        if pick(MOVED):
            wrong_k = p + r
            require_different_equations((wrong_k * x + q, t), (k * x + q, t))
            s.add_error(f"{join([(wrong_k, 1), (q, 0)])} = {num(t)}", MOVED)
            k = wrong_k
        else:
            s.add(f"{join([(k, 1), (q, 0)])} = {num(t)}")
    if k == 0:
        raise Skip("no x left")

    rhs_value = t - q
    if q != 0 and pick(MOVED):
        wrong_value = t + q
        require_different_equations((k * x, wrong_value), (k * x, rhs_value))
        s.add_error(f"{join([(k, 1)])} = {num(wrong_value)}", MOVED)
        rhs_value = wrong_value
    elif pick(SLIP):
        # Avoid slips that coincide with another named mistake (that would be a different,
        # equally valid label): -rhs_value is a sign error, t + q is moving q without its sign.
        slips = [d for d in (1, -1, 2, -2) if rhs_value + d not in (-rhs_value, t + q)]
        wrong_value = rhs_value + rng.choice(slips)
        s.add_error(f"{join([(k, 1)])} = {num(wrong_value)}", SLIP)
        rhs_value = wrong_value
    else:
        s.add(f"{join([(k, 1)])} = {num(rhs_value)}")

    if k == 1:
        return
    answer = rhs_value / k
    if k != -1 and pick(INVERSE):
        wrong = rhs_value - k
        require_different_equations((x, wrong), (x, answer))
        s.add_error(f"x = {num(wrong)}", INVERSE)
    elif k < 0 and rhs_value != 0 and pick(SIGN):  # -0 = 0: no visible mistake
        s.add_error(f"x = {num(rhs_value / -k)}", SIGN)
    elif answer.q == 1 and pick(SLIP):
        slip = rng.choice([d for d in (1, -1) if d != -2 * answer])
        s.add_error(f"x = {num(answer + slip)}", SLIP)
    else:
        s.add(f"x = {num(answer)}")


def roots_line(roots: list[sp.Rational], rng: random.Random) -> str:
    sep = rng.choice([r" \text{ or } ", ", "])
    return sep.join(f"x = {num(r)}" for r in roots)


def finish_quadratic(s: Sample, e: sp.Expr, rng: random.Random, pick: Pick) -> None:
    """e = 0 with integer coefficients: factor if possible, else the quadratic formula."""
    a, b, c = (sp.Rational(v) for v in sp.Poly(sp.expand(e), x).all_coeffs())
    disc = b * b - 4 * a * c
    if disc < 0:
        raise Skip("no real roots")
    roots = sorted(set(sp.solve(e, x)), key=lambda v: float(v))
    if a == 1 and all(r.is_Integer for r in roots) and len(roots) == 2:
        r1, r2 = roots
        if r1 != 0 and -r1 != r2 and pick(SIGN):
            s.add_error(f"{factor_text(-r1)}{factor_text(r2)} = 0", SIGN)
            r1 = -r1
        elif r1 != 0 and pick(SLIP):
            # Keep the factor's shape: a slip from (x - 1) to plain x is a different mistake.
            r1 = r1 + rng.choice([d for d in (1, -1) if r1 + d != 0])
            s.add_error(f"{factor_text(r1)}{factor_text(r2)} = 0", SLIP)
        else:
            s.add(f"{factor_text(r1)}{factor_text(r2)} = 0")
        if r1 != 0 and -r1 != r2 and pick(SIGN):
            s.add_error(roots_line([-r1, r2], rng), SIGN)
        else:
            s.add(roots_line(sorted({r1, r2}), rng))
    elif all(r.is_Rational for r in roots):
        s.add(roots_line(roots, rng))
    else:
        s.add(f"x = \\frac{{{num(-b)} \\pm \\sqrt{{{num(disc)}}}}}{{{num(2 * a)}}}")


def factor_text(root: sp.Integer) -> str:
    return "x" if root == 0 else f"({join([(1, 1), (-root, 0)])})"


def finish_expression(s: Sample, e: sp.Expr, rng: random.Random, pick: Pick, prefix: str) -> None:
    """Final simplified polynomial; a slip can hit one coefficient."""
    coeffs = [sp.Rational(c) for c in sp.Poly(sp.expand(e), x).all_coeffs()]
    n = len(coeffs) - 1
    nonzero = [i for i, c in enumerate(coeffs) if c != 0]
    if nonzero and pick(SLIP):
        i = rng.choice(nonzero)
        coeffs[i] += rng.choice([d for d in (1, -1, 2) if d != -2 * coeffs[i]])
        if coeffs[i] == 0:
            raise Skip("slip removed the term")
        s.add_error(prefix + join([(c, n - j) for j, c in enumerate(coeffs)]), SLIP)
    elif len(nonzero) > 1 and pick(SIGN):
        i = rng.choice(nonzero[1:])
        coeffs[i] = -coeffs[i]
        s.add_error(prefix + join([(c, n - j) for j, c in enumerate(coeffs)]), SIGN)
    else:
        s.add(prefix + poly(e))


# --- Problem families -----------------------------------------------------------------


def nz(rng: random.Random, lo: int, hi: int) -> int:
    return rng.choice([v for v in range(lo, hi + 1) if v != 0])


def linear_bracket(rng: random.Random, pick: Pick) -> Sample:
    """a(x + b) + c = dx + e"""
    a, b, c = rng.choice([2, 3, 4, 5, -2, -3]), nz(rng, -9, 9), nz(rng, -9, 9)
    d, x0 = rng.choice([v for v in range(-3, 4) if v != a]), rng.randint(-8, 8)
    e = a * (x0 + b) + c - d * x0
    s = Sample("linear")
    rhs = d * x + e
    s.add(f"{bracket(a, b)} {'+' if c > 0 else '-'} {abs(c)} = {poly(rhs)}")
    right = a * x + a * b + c
    if pick(PARTIAL):
        lhs = a * x + b + c
        require_different_equations((lhs, rhs), (right, rhs))
        s.add_error(f"{join([(a, 1), (b, 0), (c, 0)])} = {poly(rhs)}", PARTIAL)
    elif a < 0 and pick(SIGN):
        lhs = a * x + abs(a) * b + c  # the minus reached only the first term
        require_different_equations((lhs, rhs), (right, rhs))
        s.add_error(f"{join([(a, 1), (abs(a) * b, 0), (c, 0)])} = {poly(rhs)}", SIGN)
    else:
        lhs = right
        s.add(f"{join([(a, 1), (a * b, 0), (c, 0)])} = {poly(rhs)}")
    finish_linear(s, lhs, rhs, rng, pick)
    return s


def linear_minus_bracket(rng: random.Random, pick: Pick) -> Sample:
    """c - (x - b) = d"""
    c, b, x0 = nz(rng, -9, 12), nz(rng, -9, 9), rng.randint(-8, 8)
    d = c - (x0 - b)
    s = Sample("linear")
    s.add(f"{c} - ({join([(1, 1), (-b, 0)])}) = {d}")
    right = c - x + b
    if pick(SIGN):
        lhs = c - x - b
        require_different_equations((lhs, d), (right, d))
        s.add_error(f"{join([(c, 0), (-1, 1), (-b, 0)])} = {d}", SIGN)
    else:
        lhs = right
        s.add(f"{join([(c, 0), (-1, 1), (b, 0)])} = {d}")
    finish_linear(s, lhs, sp.Integer(d), rng, pick)
    return s


def linear_simple(rng: random.Random, pick: Pick) -> Sample:
    """ax + b = c (optionally with x terms on both sides)"""
    a, b, x0 = rng.choice([2, 3, 4, 5, 6, -2, -3, -4]), nz(rng, -12, 12), rng.randint(-9, 9)
    s = Sample("linear")
    if rng.random() < 0.5:
        r = rng.choice([v for v in range(-3, 4) if v not in (0, a)])
        t = a * x0 + b - r * x0
        s.add(f"{join([(a, 1), (b, 0)])} = {join([(r, 1), (t, 0)])}")
        finish_linear(s, a * x + b, r * x + t, rng, pick)
    else:
        s.add(f"{join([(a, 1), (b, 0)])} = {a * x0 + b}")
        finish_linear(s, a * x + b, sp.Integer(a * x0 + b), rng, pick)
    return s


def linear_fraction(rng: random.Random, pick: Pick) -> Sample:
    r"""\frac{x}{a} + b = c"""
    a, b, k = rng.choice([2, 3, 4, 5]), nz(rng, -9, 9), rng.randint(-6, 6)
    c = k + b  # x = a*k
    s = Sample("linear")
    s.add(f"\\frac{{x}}{{{a}}} {'+' if b > 0 else '-'} {abs(b)} = {c}")
    rhs = c - b
    if pick(MOVED):
        require_different_equations((x / a, c + b), (x / a, rhs))
        s.add_error(f"\\frac{{x}}{{{a}}} = {c + b}", MOVED)
        rhs = c + b
    else:
        s.add(f"\\frac{{x}}{{{a}}} = {rhs}")
    if rhs != 0 and pick(INVERSE):  # 0/a = 0*a: no visible mistake
        s.add_error(f"x = {num(sp.Rational(rhs, a))}", INVERSE)  # divided instead of multiplied
    else:
        s.add(f"x = {a * rhs}")
    return s


def quadratic_standard(rng: random.Random, pick: Pick) -> Sample:
    """x^2 + px = q, moved to one side and factored."""
    r1, r2 = rng.sample([v for v in range(-7, 8)], 2)
    p, q = -(r1 + r2), r1 * r2
    s = Sample("quadratic")
    if q != 0 and rng.random() < 0.6:
        s.add(f"{join([(1, 2), (p, 1)])} = {num(-q)}")
        right = x**2 + p * x + q
        if pick(MOVED):
            e = x**2 + p * x - q
            require_different_equations((e, 0), (right, 0))
            s.add_error(f"{poly(e)} = 0", MOVED)
        else:
            e = right
            s.add(f"{poly(e)} = 0")
    else:
        e = x**2 + p * x + q
        s.add(f"{poly(e)} = 0")
    finish_quadratic(s, e, rng, pick)
    return s


def quadratic_square(rng: random.Random, pick: Pick) -> Sample:
    """(x + a)^2 = k^2: square root (needs ±) or, wrongly, expand as a^2 + b^2."""
    a, k = nz(rng, -6, 6), rng.randint(1, 7)
    s = Sample("quadratic")
    s.add(f"({join([(1, 1), (a, 0)])})^{{2}} = {k * k}")
    if pick(POWER):
        require_different_equations((x**2 + a * a, k * k), ((x + a) ** 2, k * k))
        s.add_error(f"{join([(1, 2), (a * a, 0)])} = {k * k}", POWER)
        if a * a > k * k:
            raise Skip("no real roots after the mistake")
        s.add(f"x^{{2}} = {k * k - a * a}")
        root = sp.sqrt(k * k - a * a)
        s.add(f"x = \\pm {sp.latex(root)}" if root != 0 else "x = 0")
        return s
    if pick(PLUS_MINUS):
        s.add_error(f"{join([(1, 1), (a, 0)])} = {k}", PLUS_MINUS)
        s.add(f"x = {k - a}")
        return s
    s.add(f"{join([(1, 1), (a, 0)])} = \\pm {k}")
    s.add(roots_line([k - a, -k - a], rng))
    return s


def quadratic_common_factor(rng: random.Random, pick: Pick) -> Sample:
    """ax^2 = bx: factor out x (dividing by x loses x = 0)."""
    a, r = rng.choice([1, 2, 3, 4]), nz(rng, -6, 6)
    b = a * r
    s = Sample("quadratic")
    s.add(f"{join([(a, 2)])} = {join([(b, 1)])}")
    if pick(DIV_ZERO):
        s.add_error(f"{join([(a, 1)])} = {b}", DIV_ZERO)
        if a != 1:
            s.add(f"x = {r}")
        return s
    s.add(f"{join([(a, 2), (-b, 1)])} = 0")
    s.add(f"{coef_prefix(a)}x({join([(1, 1), (-r, 0)])}) = 0")
    s.add(roots_line([0, r], rng))
    return s


def quadratic_root(rng: random.Random, pick: Pick) -> Sample:
    """x^2 = k^2 (sometimes as x^2 - k^2 = 0 first)."""
    k = rng.randint(1, 9)
    s = Sample("quadratic")
    if rng.random() < 0.5:
        s.add(f"x^{{2}} - {k * k} = 0")
    s.add(f"x^{{2}} = {k * k}")
    if pick(PLUS_MINUS):
        s.add_error(f"x = {k}", PLUS_MINUS)
    else:
        s.add(rng.choice([f"x = \\pm {k}", roots_line([k, -k], rng)]))
    return s


def expression_distribute(rng: random.Random, pick: Pick) -> Sample:
    """a(x + b) + c(x + d) -> expanded -> combined"""
    a, b, c, d = nz(rng, -5, 6), nz(rng, -9, 9), nz(rng, -5, 6), nz(rng, -9, 9)
    s = Sample("expression")
    s.add(f"{bracket(a, b)} {'+' if c > 0 else '-'} {bracket(abs(c), d)}")
    prefix = "= " if rng.random() < 0.5 else ""
    terms = [(a, 1), (a * b, 0), (c, 1), (c * d, 0)]
    kind = None
    if abs(a) != 1 and pick(PARTIAL):
        terms[1], kind = (b, 0), PARTIAL  # multiplied only the x by a
    elif c < 0 and pick(SIGN):
        terms[3], kind = (-c * d, 0), SIGN  # the minus did not reach the second term
    expanded = sum(cf * x**p for cf, p in terms)
    if kind:
        require_different_expressions(expanded, a * (x + b) + c * (x + d))
        s.add_error(prefix + join(terms), kind)
    else:
        s.add(prefix + join(terms))
    finish_expression(s, expanded, rng, pick, prefix)
    return s


def expression_square(rng: random.Random, pick: Pick) -> Sample:
    """(x + a)^2 + bx -> x^2 + 2ax + a^2 + bx -> combined"""
    a, b = nz(rng, -7, 7), nz(rng, -9, 9)
    s = Sample("expression")
    s.add(f"({join([(1, 1), (a, 0)])})^{{2}} {'+' if b > 0 else '-'} {_body(abs(b), 1)}")
    prefix = "= " if rng.random() < 0.5 else ""
    if pick(POWER):
        value = x**2 + a * a + b * x
        s.add_error(prefix + join([(1, 2), (a * a, 0), (b, 1)]), POWER)
    else:
        value = (x + a) ** 2 + b * x
        s.add(prefix + join([(1, 2), (2 * a, 1), (a * a, 0), (b, 1)]))
    finish_expression(s, value, rng, pick, prefix)
    return s


def expression_fraction(rng: random.Random, pick: Pick) -> Sample:
    r"""\frac{cx + cd}{c} + ex -> x + d + ex -> combined"""
    c, d, e = rng.choice([2, 3, 4, 5]), nz(rng, -9, 9), nz(rng, -6, 6)
    s = Sample("expression")
    s.add(
        f"\\frac{{{join([(c, 1), (c * d, 0)])}}}{{{c}}} {'+' if e > 0 else '-'} {_body(abs(e), 1)}"
    )
    prefix = "= " if rng.random() < 0.5 else ""
    if pick(CANCEL):
        value = x + c * d + e * x  # divided only the first term by c
        s.add_error(prefix + join([(1, 1), (c * d, 0), (e, 1)]), CANCEL)
    else:
        value = x + d + e * x
        s.add(prefix + join([(1, 1), (d, 0), (e, 1)]))
    finish_expression(s, value, rng, pick, prefix)
    return s


FAMILIES: dict[str, tuple[Callable[[random.Random, Pick], Sample], list[str]]] = {
    "linear_bracket": (linear_bracket, [PARTIAL, SIGN, MOVED, SLIP, INVERSE]),
    "linear_minus_bracket": (linear_minus_bracket, [SIGN, MOVED, SLIP, INVERSE]),
    "linear_simple": (linear_simple, [MOVED, SLIP, INVERSE, SIGN]),
    "linear_fraction": (linear_fraction, [MOVED, INVERSE]),
    "quadratic_standard": (quadratic_standard, [MOVED, SIGN, SLIP]),
    "quadratic_square": (quadratic_square, [POWER, PLUS_MINUS]),
    "quadratic_common_factor": (quadratic_common_factor, [DIV_ZERO]),
    "quadratic_root": (quadratic_root, [PLUS_MINUS]),
    "expression_distribute": (expression_distribute, [PARTIAL, SIGN, SLIP]),
    "expression_square": (expression_square, [POWER, SLIP, SIGN]),
    "expression_fraction": (expression_fraction, [CANCEL, SLIP, SIGN]),
}


def generate_one(rng: random.Random, error_rate: float) -> dict[str, object]:
    """One sample. Retries until the drawn parameters give a valid sample."""
    while True:
        name = rng.choice(list(FAMILIES))
        build, error_types = FAMILIES[name]
        target = rng.choice(error_types) if rng.random() < error_rate else None
        used = False

        def pick(error_type: str, target: str | None = target) -> bool:
            # Inject the target error at the first opportunity (each opportunity taken with
            # probability 0.6, so the error position varies), at most once.
            nonlocal used
            if used or error_type != target or rng.random() > 0.6:
                return False
            used = True
            return True

        try:
            sample = build(rng, pick)
        except Skip:
            continue
        if target is not None and sample.error_type is None:
            continue  # never got the chance to inject; draw again rather than mislabel
        record = asdict(sample)
        record["family"] = name
        return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--error-rate", type=float, default=0.7)
    parser.add_argument("--out", type=Path, default=Path("data/synthetic.jsonl"))
    args = parser.parse_args()

    rng = random.Random(args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for i in range(args.n):
            record = {"id": i, **generate_one(rng, args.error_rate)}
            f.write(json.dumps(record) + "\n")
    print(f"wrote {args.n} samples to {args.out}")


if __name__ == "__main__":
    main()
