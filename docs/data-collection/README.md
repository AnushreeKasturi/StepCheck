# Collecting real student solutions

Goal: about 100 handwritten solutions (10-15 students × the 10 problems in
[problem-sheet.md](problem-sheet.md)) to measure two things separately:

1. **Engine on real mistakes:** run the checker on *your* typed transcription of each photo.
2. **End to end:** run the checker on what the OCR reads from the photo (Phase 3).

Keeping them separate shows whether a failure comes from reading the handwriting or from
the math.

## Each problem tests something

| # | Problem | Answer | Common mistakes it invites |
|---|---|---|---|
| 1 | 3(x − 4) − 2 = x + 6 | x = 10 | partial distribution, moved term without sign change |
| 2 | 7 − (2x − 5) = 3x − 8 | x = 4 | sign error with the bracket |
| 3 | x/3 + 5 = 2x − 5 | x = 6 | multiplying only some terms by 3 |
| 4 | x² − 5x = −6 | x = 2 or x = 3 | sign of the moved −6, signs of the roots |
| 5 | (x − 3)² = 16 | x = 7 or x = −1 | missing ±, (a − b)² = a² − b² |
| 6 | 3x² = 12x | x = 0 or x = 4 | dividing by x (loses x = 0) |
| 7 | (x + 4)² − 2(x − 3) | x² + 6x + 22 | (a + b)² = a² + b², sign error with −2 |
| 8 | (x² − 9)/(x − 3) + 2(x − 1) | 3x + 1 | invalid cancellation |
| 9 | √(x + 7) = x + 1 | x = 2 | keeping the extraneous root x = −3 |
| 10 | (2x + 1)/(x − 2) = 3 | x = 7 | clearing the fraction wrongly |

Don't give students the answers before they finish.

## Privacy

- Ask for consent (the sheet explains it). Students can withdraw and have their photos deleted.
- Store photos in `data/real/photos/`; `data/` is in `.gitignore`, so nothing is published.
- Rename photos to codes like `s07_p3.jpg` (student 7, problem 3). Keep any name-to-code
  list outside the repo, or don't keep one at all.

## Transcribing and labelling

Make one text file per photo in `data/real/solutions/`, named like the photo
(`s07_p2.txt` for `s07_p2.jpg`):

```
% problem: p2
% error_line: 2
% error_type: sign_error
7 - (2x - 5) = 3x - 8
7 - 2x - 5 = 3x - 8
2 - 2x = 3x - 8
10 = 5x
x = 2
```

- **Solution lines:** what the student **actually wrote**, one step per line in LaTeX
  (`\frac{x}{3}`, `x^2`, `\sqrt{x + 7}`, `x = 2 \text{ or } x = 3`), mistakes included.
  Do not fix anything. Leave out crossed-out lines and side calculations in the margin.
- **`error_line`:** the number of the **first** wrong line (the problem itself is line 1),
  or `none` if the whole solution is correct.
- **`error_type`:** one of `sign_error`, `partial_distribution`, `power_of_sum`,
  `invalid_cancellation`, `moved_term_no_sign_change`, `inverse_operation`, `arithmetic_slip`,
  `divided_by_zero_expression`, `missing_plus_minus`, `extraneous_not_rejected`, or `other`
  if none fits.
- Optional `% note: ...` for anything unclear.
- Decide the label **before** running StepCheck on the file, so the checker can't influence
  you. If possible, have a friend label ~20 of them independently: how often you agree tells
  you how reliable the labels are.

Convert the folder into a dataset (it reports any label mistakes by file name):

```bash
python -m stepcheck.eval.collect data/real/solutions --out data/real/labels.jsonl
```

Then run the benchmark on it exactly like on synthetic data:

```bash
python -m stepcheck.eval.benchmark data/real/labels.jsonl --report docs/results-real.md
```

Mistakes outside the catalog (`other`) are expected to come back "unclassified"; the line
should still be found.
