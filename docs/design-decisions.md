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

---

# Design decisions (Phase 2: diagnosis and evaluation)

## 8. Two searches: forward mutations and backward repairs
- **Forward** (`mutations.py`): apply a known mistake ("mal-rule", as in the intelligent
  tutoring literature) to line k-1 and see if it reproduces line k. Names misconceptions:
  partial distribution, (a+b)^2 -> a^2+b^2, moving a term without changing its sign, ...
- **Backward** (`repairs.py`): make one small edit to the student's line k (a number off by
  1 or 2, one sign, one coefficient) and see if it becomes a valid step. Catches slips in the
  *result*, which forward search cannot reproduce, and gives the corrected line for free.

## 9. Mutations run on what the student literally wrote
SymPy normally evaluates `-(x - 3)` to `3 - x` at once, which erases the misconception. Each
line is therefore also parsed with `sp.evaluate(False)` (`Step.raw`). Nested sums are
flattened so "a term" means what a student means by it.
*Regression:* `parse_latex` lazily imports `sympy.physics` on first use, and that import
crashes inside `evaluate(False)`. The parser is now warmed up at import time.

## 10. Matching by equivalence, ranked by how well the shape fits
A candidate matches if it is *equivalent* to the student's line, not identical, so a mistake
combined with correct simplification in the same line is still recognized. For equations,
"equivalent" means the same equation up to a constant factor, which is stricter than equal
solution sets (two different wrong linear equations often share their one solution).
When several explanations fit, they are ranked:
1. **strong** forward match: equivalent side by side, so it also explains the line's shape;
2. **repair**: a one-symbol edit makes the line correct;
3. **weak** forward match: equal only after rearranging, which can be a coincidence;
then by a fixed specificity order (named misconception before generic sign error).
*Example:* for `5 - (x-3) = 2 -> 5 - x - 3 = 2`, "moved a term without changing its sign"
also fits algebraically, but only the sign error explains the shape: the student moved nothing.
*Measured:* ranking weak forward matches above repairs instead lowered dev-set type accuracy
from 99.7% to 98.4% (39 sign errors were explained by coincidental "moved term" matches).
*Known gap:* when a student does two moves in one line and only one is wrong, the right
explanation is only a weak match and a coincidental coefficient repair can win (7 of 2,876
dev errors).

## 11. Special cases outside the search
- **Divided by zero:** a LOST-solutions step where line k-1 equals line k times a polynomial
  that vanishes at the lost values. *Regression:* a rational ratio such as (x-2)/(x+2) used to
  pass and turned factoring sign errors into "divided by zero".
- **Extraneous root kept:** from Phase 1's implication rule; only used when no mistake in the
  catalog explains the extra solutions better.

## 12. The suggested correction is checked by the checker itself
A test runs every diagnosis's "the line should be ..." through `check_solution`. Telling a
student to write something wrong would be worse than saying nothing.

## 13. Benchmark design
- **Independent generator.** `eval/generate.py` has its own simulated student with its own
  faulty operations and must not import the engine (a test enforces this via the AST).
  Otherwise type accuracy would only show the engine agreeing with itself.
- **Every injected error is verified** with SymPy to change the math. *Regression:* dividing
  0 by 2 instead of -2 was labelled a sign error although -0 = 0.
- **Errors carry forward:** after a mistake, the simulated student keeps solving correctly
  from the wrong line, like a real student.
- **No mislabelled slips:** an "off by 2" slip that lands on -value is really a sign error, so
  the generator does not produce it.
- **Dev/test split:** the engine was tuned on failures from seed 0; reported numbers come from
  seed 1, generated once and not inspected for tuning.
- Rates are reported with 95% Wilson intervals: "0 false alarms in 1,500" still means
  "at most ~0.25%", not "exactly 0%".
