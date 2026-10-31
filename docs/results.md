# Benchmark results

`python -m stepcheck.eval.benchmark data\test.jsonl`

5000 solutions: 2802 with exactly one injected mistake, 2198 fully correct. Rates show the 95% Wilson interval and counts.

| Metric | Result |
|---|---|
| **Error localization** (exact wrong line) | 100.0% (99.9-100.0%, 2802/2802) |
| **Error type accuracy** (given correct line) | 100.0% (99.8-100.0%, 2801/2802) |
| End-to-end (right line *and* right type) | 100.0% (99.8-100.0%, 2801/2802) |
| **False alarms on correct solutions** | 0.0% (0.0-0.2%, 0/2198) |
| Blamed a correct line before the real mistake | 0.0% (0.0-0.1%, 0/2802) |
| Mistake detected at all | 100.0% (99.9-100.0%, 2802/2802) |
| True type among alternatives | 100.0% (99.9-100.0%, 2802/2802) |
| Correct solutions left unverified/incomplete | 0.0% (0.0-0.2%, 0/2198) |
| Crashes | 0 |
| Latency p50 / p95 / max (15 parallel workers) | 288 / 679 / 1674 ms |

## By error type

| Injected error | Localization | Type accuracy (given line) |
|---|---|---|
| arithmetic_slip | 100.0% (99.4-100.0%, 599/599) | 99.8% (99.1-100.0%, 598/599) |
| divided_by_zero_expression | 100.0% (98.6-100.0%, 274/274) | 100.0% (98.6-100.0%, 274/274) |
| invalid_cancellation | 100.0% (95.5-100.0%, 81/81) | 100.0% (95.5-100.0%, 81/81) |
| inverse_operation | 100.0% (98.3-100.0%, 220/220) | 100.0% (98.3-100.0%, 220/220) |
| missing_plus_minus | 100.0% (99.1-100.0%, 403/403) | 100.0% (99.1-100.0%, 403/403) |
| moved_term_no_sign_change | 100.0% (99.0-100.0%, 376/376) | 100.0% (99.0-100.0%, 376/376) |
| partial_distribution | 100.0% (97.3-100.0%, 138/138) | 100.0% (97.3-100.0%, 138/138) |
| power_of_sum | 100.0% (97.7-100.0%, 163/163) | 100.0% (97.7-100.0%, 163/163) |
| sign_error | 100.0% (99.3-100.0%, 548/548) | 100.0% (99.3-100.0%, 548/548) |

## By problem family

| Family | Localization | False alarms |
|---|---|---|
| expression_distribute | 100.0% (98.7-100.0%, 282/282) | 0.0% (0.0-1.8%, 0/204) |
| expression_fraction | 100.0% (98.4-100.0%, 241/241) | 0.0% (0.0-1.9%, 0/200) |
| expression_square | 100.0% (98.5-100.0%, 256/256) | 0.0% (0.0-1.9%, 0/196) |
| linear_bracket | 100.0% (98.6-100.0%, 268/268) | 0.0% (0.0-2.0%, 0/186) |
| linear_fraction | 100.0% (98.6-100.0%, 268/268) | 0.0% (0.0-1.9%, 0/202) |
| linear_minus_bracket | 100.0% (98.4-100.0%, 235/235) | 0.0% (0.0-1.8%, 0/204) |
| linear_simple | 100.0% (98.7-100.0%, 282/282) | 0.0% (0.0-1.7%, 0/222) |
| quadratic_common_factor | 100.0% (98.6-100.0%, 274/274) | 0.0% (0.0-1.9%, 0/195) |
| quadratic_root | 100.0% (98.5-100.0%, 256/256) | 0.0% (0.0-2.0%, 0/190) |
| quadratic_square | 100.0% (98.4-100.0%, 229/229) | 0.0% (0.0-1.9%, 0/198) |
| quadratic_standard | 100.0% (98.2-100.0%, 211/211) | 0.0% (0.0-1.9%, 0/201) |

## Confusion matrix (localized errors; rows = injected, columns = predicted)

| injected \ predicted | arithmetic_slip | divided_by_zero_expression | invalid_cancellation | inverse_operation | missing_plus_minus | moved_term_no_sign_change | partial_distribution | power_of_sum | sign_error |
|---|---|---|---|---|---|---|---|---|---|
| arithmetic_slip | 598 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 |
| divided_by_zero_expression | 0 | 274 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| invalid_cancellation | 0 | 0 | 81 | 0 | 0 | 0 | 0 | 0 | 0 |
| inverse_operation | 0 | 0 | 0 | 220 | 0 | 0 | 0 | 0 | 0 |
| missing_plus_minus | 0 | 0 | 0 | 0 | 403 | 0 | 0 | 0 | 0 |
| moved_term_no_sign_change | 0 | 0 | 0 | 0 | 0 | 376 | 0 | 0 | 0 |
| partial_distribution | 0 | 0 | 0 | 0 | 0 | 0 | 138 | 0 | 0 |
| power_of_sum | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 163 | 0 |
| sign_error | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 548 |

## How to read these numbers

- **Synthetic data.** The generator and the engine both encode the same author's model of
  student mistakes, so they agree by design. These numbers show the engine does what it was
  built to do; accuracy on real handwritten work will be lower and is measured separately.
- **Held-out test set.** The engine was tuned while looking at failures on a *dev* set
  (seed 0). The reported numbers come from a different seed that was never used for tuning.
- **Single injected mistake.** Real solutions can contain several, or mistakes outside the
  catalog; those become "unclassified" (the line is still found, the type is not named).
- **Latency** is wall-clock per solution with several worker processes sharing the CPU; a
  single process is roughly 2-4x faster.
- **Saturated.** At ~100% this benchmark only shows that the engine covers the generator's
  mistake catalog; it can no longer separate a better engine from a worse one. Harder tiers
  (two mistakes in one line, mistakes outside the catalog) are needed for that.
- **Reproduce:** `python -m stepcheck.eval.generate --n 5000 --seed 1 --out data/test.jsonl`
  then `python -m stepcheck.eval.benchmark data/test.jsonl`.

