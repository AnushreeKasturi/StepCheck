"""Equation steps: compare real solution sets.

An equation step is valid when it preserves the solution set. Comparing sets (rather than
the equations' shapes) accepts every legitimate move: adding to both sides, multiplying
by a non-zero constant, factoring. It also tells us *how* a step went wrong:
lost solutions (divided by x) versus extra solutions (squared both sides).
"""

from dataclasses import dataclass

import sympy as sp

from stepcheck.engine.equivalence import expressions_equivalent
from stepcheck.engine.model import Disjunction, Equation, Verdict


@dataclass(frozen=True)
class SetComparison:
    verdict: Verdict
    lost: sp.Set = sp.S.EmptySet
    extra: sp.Set = sp.S.EmptySet


def real_solution_set(obj: Equation | Disjunction, x: sp.Symbol) -> sp.Set | None:
    """Solutions over the reals, or None if SymPy cannot give them in closed form."""
    if isinstance(obj, Disjunction):
        sets = [real_solution_set(eq, x) for eq in obj.equations]
        if any(s is None for s in sets):
            return None
        return sp.Union(*sets)
    result = sp.solveset(sp.Eq(obj.lhs, obj.rhs), x, sp.S.Reals)
    if result.has(sp.ConditionSet, sp.ImageSet):
        return None
    return result


def _is_finite(s: sp.Set) -> bool:
    return isinstance(s, sp.FiniteSet) or s is sp.S.EmptySet


def _provably_different(u: sp.Expr, v: sp.Expr, x: sp.Symbol) -> bool:
    return expressions_equivalent(u, v, x).verdict == Verdict.NOT_EQUIVALENT


def outside(source: sp.Set, target: sp.Set, x: sp.Symbol) -> sp.Set | None:
    """The part of `source` that is not in `target`, or None if that cannot be decided."""
    if _is_finite(source) and _is_finite(target):
        # Match elements one by one so equal numbers in different forms (sqrt(8)/2 vs sqrt(2))
        # still match: a value only counts as outside if it is provably different.
        return sp.FiniteSet(
            *(u for u in source if all(_provably_different(u, v, x) for v in target))
        )
    difference = sp.Complement(source, target)
    return None if difference.is_empty is None else difference


def _verdict(lost_empty: bool, extra_empty: bool) -> Verdict:
    if lost_empty and extra_empty:
        return Verdict.EQUIVALENT
    if extra_empty:
        return Verdict.LOST_SOLUTIONS
    if lost_empty:
        return Verdict.EXTRA_SOLUTIONS
    return Verdict.LOST_AND_EXTRA


def compare_solution_sets(before: sp.Set, after: sp.Set, x: sp.Symbol) -> SetComparison:
    lost, extra = outside(before, after, x), outside(after, before, x)
    if lost is None or extra is None:
        return SetComparison(Verdict.UNKNOWN)
    return SetComparison(_verdict(bool(lost.is_empty), bool(extra.is_empty)), lost, extra)


def compare_equations(
    before: Equation | Disjunction, after: Equation | Disjunction, x: sp.Symbol
) -> SetComparison:
    a, b = real_solution_set(before, x), real_solution_set(after, x)
    if a is None or b is None:
        return SetComparison(Verdict.UNKNOWN)
    return compare_solution_sets(a, b, x)
