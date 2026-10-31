# Design decisions (Phase 1: engine)

The guiding rule: **telling a student that correct work is wrong is the worst failure.**
Most decisions below trade a little recall for a lower false-alarm rate.

## 1. Three-valued answers, and errors need a witness
Every check returns EQUIVALENT, an error verdict, or UNKNOWN. UNKNOWN is never shown as a
mistake ("could not verify" instead). An expression step is only NOT_EQUIVALENT if we have a
concrete value of x where both sides are defined and differ. If we cannot produce one, we do
not accuse the student.

## 2. Expression equivalence: exact when possible, numeric otherwise
1. **Rational functions with rational coefficients** (polynomials, fractions: most of v1):
   `cancel(a - b) == 0` is a complete decision procedure, so no guessing is involved.
   *Regression:* SymPy's `is_rational_function` also accepts coefficients like sqrt(2), where
   `cancel` is *not* complete. `sqrt(2)+sqrt(3)` vs `sqrt(5+2sqrt(6))` was wrongly flagged.
   We now require integer exponents everywhere.
2. `expand(a - b) == 0` is a cheap proof for things like `2^(x+1)` vs `2*2^x`.
3. Otherwise evaluate at random rational points with 50-digit precision. One mismatch proves
   inequality. Agreement at 20 points is strong evidence of equality. For polynomials,
   Schwartz-Zippel bounds the failure probability by degree/|sample space|.
   Friendly values (1, 2, -1, ...) are tried first so the counterexample shown to a student is
   readable, but agreement at them is not counted (0 and ±1 hide too many differences).

`simplify()` is deliberately not used: it is heuristic, can run for a very long time, and
Windows has no `signal.alarm` to time it out.

## 3. Domains
The variable is declared `real=True`, never `positive=True`, so `sqrt(x^2) -> x` is caught
(it is wrong for x < 0). Two expressions are equivalent if they agree wherever both are
defined, so cancelling `(x^2-1)/(x-1) -> x+1` is an accepted *simplification*. The domain only
matters in *equations*, and there the solution-set check catches it.
**Known limitation:** SymPy rewrites `x/x` to `1` while parsing, so that restriction is lost.

## 4. Equation steps compare real solution sets
Uses `solveset(..., Reals)`. This accepts every legitimate rearrangement and classifies
failures as lost solutions (divided by x), extra solutions (squared) or both (plain mistake).
Individual solutions are matched with rule 2, so `sqrt(8)/2` matches `sqrt(2)`.

## 5. Steps must be valid implications, not equivalences
Squaring both sides is the textbook method for radical equations, but it adds extraneous
roots. Flagging it would be a false alarm, so:
- EXTRA solutions in one step are fine (a valid "=>" step that creates candidates).
- Dropping values that never solved the original problem is REJECTED_EXTRANEOUS (correct).
- Extra values are an error only if they survive to the final answer. Then we blame the step
  that introduced them, unless another step is already wrong (e.g. the student rejected the
  real root instead of the fake one: that rejection is the real mistake).

## 6. Parsing: we split the line structure ourselves
`parse_latex` silently drops the second half of `x = 2, x = -3`, turns
`x=2 \text{ or } x=-3` into `False`, and reads `\pm` as a variable named `pm`. So we split on
`=`, "or", commas and `\pm` ourselves and only give it single expressions. Other repairs:
`x(x+1)` (parsed as a *function call*), floats to exact rationals, and `\left|...\right|`.
`parse_latex` returns unevaluated trees. We evaluate them for Phase 1; Phase 2's mutation
search will want the unevaluated form, since it is what the student literally wrote.

## 7. Localization
Every pair is checked, not just up to the first error, so we can also say "lines 5-7 follow
correctly from your mistake". Unreadable lines are bridged (line k-1 is compared with k+1).
If any step is unverified, the last line is also compared with the first. If that fails and
there was exactly one unverified step, it must contain the mistake.

## Other known v1 limitations
- Decimal approximations (`x = 1.414` for sqrt(2)) are treated as exact, so they are flagged.
- Expression <-> equation steps (writing `(x-2)(x+2)` under `x^2-4=0`) are not compared.
- One variable per solution; no inequalities or systems.
