"""Are two expressions equal as real functions of x?

Strategy (cheapest and most certain first):
1. Rational functions (polynomials and fractions): `cancel(a - b) == 0` is a complete
   decision procedure, so the answer is exact either way.
2. Otherwise, `expand(a - b) == 0` proves equality cheaply when it works.
3. Otherwise, evaluate at random rational points. A single mismatch *proves* non-equivalence
   (and gives a counterexample); agreement everywhere is strong evidence of equivalence.

Points where either side is undefined over the reals are skipped, so two expressions count
as equivalent if they agree wherever both are defined (x^2/x vs x is a valid simplification).
We deliberately avoid `simplify()`: it is heuristic and can run for a very long time.
"""

import random
from dataclasses import dataclass

import sympy as sp

from stepcheck.engine.model import Method, Verdict

N_POINTS = 20
MIN_POINTS = 5  # fewer comparable points than this -> UNKNOWN
MAX_ATTEMPTS = 200
_PRECISION = 50  # significant digits for numeric evaluation
_REL_TOL = sp.Float("1e-30")
FRIENDLY_POINTS = [sp.Integer(n) for n in (1, 2, -1, 3, -2, 0, 4, -3, 5, -4, 10, -10)]


@dataclass(frozen=True)
class EquivResult:
    verdict: Verdict  # EQUIVALENT, NOT_EQUIVALENT or UNKNOWN
    method: Method
    counterexample: sp.Rational | None = None


def sample_points(seed: int = 0) -> list[sp.Rational]:
    """Deterministic random rationals in [-40, 40]; non-integers avoid lucky coincidences."""
    rng = random.Random(seed)
    return [
        sp.Rational(rng.randint(-40 * q, 40 * q), q)
        for q in (rng.randint(1, 7) for _ in range(MAX_ATTEMPTS))
    ]


def real_value(e: sp.Expr, x: sp.Symbol, point: sp.Rational) -> sp.Expr | None:
    """Value of e at x = point, or None if it is undefined or not real there."""
    v = e.xreplace({x: point})
    if v.has(sp.zoo, sp.nan, sp.oo, -sp.oo):
        return None
    if v.is_Rational:
        return v
    n = sp.N(v, _PRECISION)
    if not n.is_Number or n.has(sp.zoo, sp.nan, sp.oo, -sp.oo):
        return None  # complex (e.g. sqrt of a negative) or undefined
    return n


def values_equal(u: sp.Expr, v: sp.Expr) -> bool:
    if u.is_Rational and v.is_Rational:
        return bool(u == v)
    scale = max(sp.Integer(1), abs(u), abs(v))
    return bool(abs(u - v) <= _REL_TOL * scale)


def find_counterexample(
    a: sp.Expr, b: sp.Expr, x: sp.Symbol, seed: int = 0
) -> tuple[sp.Rational | None, int]:
    """Search for a point where a and b are both defined but differ.

    Small "friendly" values are tried first so the witness is easy to show a student. Agreement
    there proves little (0 and 1 are special), so only random points count towards `compared`.

    Returns (counterexample or None, number of random points where both were defined).
    """
    for p in FRIENDLY_POINTS:
        va, vb = real_value(a, x, p), real_value(b, x, p)
        if va is not None and vb is not None and not values_equal(va, vb):
            return p, 0
    compared = 0
    for p in sample_points(seed):
        va, vb = real_value(a, x, p), real_value(b, x, p)
        if va is None or vb is None:
            continue
        if not values_equal(va, vb):
            return p, compared
        compared += 1
        if compared >= N_POINTS:
            break
    return None, compared


def is_rational_over_q(e: sp.Expr, x: sp.Symbol) -> bool:
    """Rational function of x with rational coefficients: no radicals, no functions.

    SymPy's own `is_rational_function` accepts sqrt(2) as a coefficient, but `cancel` is only
    a complete equality test over the rationals: it misses sqrt(2) + sqrt(3) = sqrt(5 + 2 sqrt(6)).
    """
    return bool(
        e.is_rational_function(x)
        and all(p.exp.is_Integer for p in e.atoms(sp.Pow))
        and not e.atoms(sp.Function)
    )


def expressions_equivalent(a: sp.Expr, b: sp.Expr, x: sp.Symbol) -> EquivResult:
    """Invariant: NOT_EQUIVALENT is only returned together with a concrete counterexample."""
    diff = a - b
    if is_rational_over_q(diff, x):
        if sp.cancel(sp.together(diff)) == 0:
            return EquivResult(Verdict.EQUIVALENT, "exact")
        point, _ = find_counterexample(a, b, x)
        if point is not None:
            return EquivResult(Verdict.NOT_EQUIVALENT, "exact", point)
        return EquivResult(Verdict.UNKNOWN, "exact")  # cannot happen in theory; never guess

    if sp.expand(diff) == 0:
        return EquivResult(Verdict.EQUIVALENT, "exact")

    point, compared = find_counterexample(a, b, x)
    if point is not None:
        return EquivResult(Verdict.NOT_EQUIVALENT, "numeric", point)
    if compared < MIN_POINTS:
        return EquivResult(Verdict.UNKNOWN, "numeric")
    return EquivResult(Verdict.EQUIVALENT, "numeric")
