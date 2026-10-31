# StepCheck

**Find the exact step where a math solution goes wrong, and explain the mistake.**

Tools like Photomath solve the problem for you. StepCheck reads *your* step-by-step work,
checks that each line follows from the one before, and points to the first wrong line:

```
$ python -m stepcheck check solution.txt
      1  5 - (x - 3) = 2  (start)
  XX  2  5 - x - 3 = 2    changes the solutions of line 1: loses {6}, adds {0}
  ok  3  2 - x = 2        ok (solveset)
  ok  4  x = 0            ok (solveset)

Result: ERROR - the first mistake is on line 2.
[sign error] Line 2: Sign error: -(x - 3) should become -x + 3, but you wrote -x - 3.
The line should be 5 - x + 3 = 2.
```

> **Status:** in progress. The checking engine and error diagnosis work on typed LaTeX.
> Next: reading handwritten solutions from photos, then a mobile-friendly web app.

## How it works

1. **Parse** each line (LaTeX) into an expression, an equation, or "x = a or x = b".
2. **Check every step with a symbolic engine (SymPy), never an LLM.**
   - Expressions: exact test for polynomials and fractions; otherwise evaluation at random
     points. A step is only called wrong with a concrete counterexample.
   - Equations: compare the real solution sets, which shows *how* a step went wrong:
     lost solutions (divided by x) or extra ones (squared both sides). Squaring and then
     rejecting the extraneous root is accepted, because that is the textbook method.
3. **Diagnose** the first wrong step:
   - *forward:* apply known student mistakes to the previous line (sign errors,
     a(b + c) -> ab + c, (a + b)^2 -> a^2 + b^2, moving a term without changing its sign, ...)
     and see which one reproduces what the student wrote;
   - *backward:* try one-symbol repairs to the student's line (an arithmetic slip, one sign)
     and see which makes it correct.

   Each diagnosis comes with the corrected line, which is itself verified by the checker.

The reasoning behind each choice is in [docs/design-decisions.md](docs/design-decisions.md).

## Results

On 5,000 held-out synthetic solutions (single-variable algebra, one injected mistake in 56%
of them), full table in [docs/results.md](docs/results.md):

| Metric | Result |
|---|---|
| Finds the exact wrong line | 100% (2,802 / 2,802) |
| Names the type of mistake | 99.96% (2,801 / 2,802) |
| False alarms on correct solutions | 0 of 2,198 (95% upper bound: 0.2%) |
| Time per solution (median) | ~0.3 s |

These numbers are from synthetic data and mainly show that the engine covers its catalog of
mistakes. An evaluation on real handwritten student work is in progress.

## Scope

Single-variable algebra: simplifying expressions, linear and quadratic equations, fractions,
exponents, square roots. Not yet: inequalities, systems of equations, calculus, word problems.

## Running it

```bash
python -m venv .venv && .venv/Scripts/activate    # Linux/macOS: source .venv/bin/activate
pip install -e ".[dev]"

python -m stepcheck check tests/fixtures/sign_error.txt   # one LaTeX line per step
pytest                                                    # tests

python -m stepcheck.eval.generate --n 5000 --seed 1 --out data/test.jsonl
python -m stepcheck.eval.benchmark data/test.jsonl        # writes docs/results.md
```

## Related work

[CheckMySteps](https://justinmath.com/checkmysteps/) checks algebra steps by comparing solution
sets. StepCheck adds error diagnosis, handwritten input, and a measured false-alarm rate.
Recent studies (FERMAT, ACL 2025; "LLMs cannot spot math errors", EMNLP 2025) find that
large language models struggle to locate the first wrong step, which is why correctness here
is decided symbolically.
