"""Run the engine over a labelled dataset and report how well it finds and names mistakes.

Usage: python -m stepcheck.eval.benchmark data/synthetic.jsonl [--workers 8] [--limit 500]
"""

import argparse
import json
import math
import os
import statistics
import time
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

Record = dict[str, Any]


@dataclass(frozen=True)
class Outcome:
    id: int
    status: str
    first_error: int | None
    predicted_type: str | None
    alternatives: tuple[str, ...]
    latency_ms: float
    crash: str | None = None


def run_one(record: Record) -> Outcome:
    from stepcheck.engine.checker import check_solution  # imported in the worker process

    start = time.perf_counter()
    try:
        result = check_solution(record["lines"])
    except Exception as exc:  # a crash is a result to report, not a reason to stop
        return Outcome(record["id"], "crash", None, None, (), 0.0, f"{type(exc).__name__}: {exc}")
    latency = (time.perf_counter() - start) * 1000
    d = result.diagnosis
    return Outcome(
        record["id"],
        result.status,
        result.first_error,
        d.error_type.value if d else None,
        tuple(a.value for a in d.alternatives) if d else (),
        latency,
    )


# --- Metrics (pure functions) -----------------------------------------------------------


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% confidence interval for a proportion; sensible even when k is 0 or n is small."""
    if n == 0:
        return 0.0, 1.0
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


@dataclass(frozen=True)
class Rate:
    k: int
    n: int

    @property
    def value(self) -> float:
        return self.k / self.n if self.n else 0.0

    def __str__(self) -> str:
        lo, hi = wilson(self.k, self.n)
        return f"{100 * self.value:.1f}% ({100 * lo:.1f}-{100 * hi:.1f}%, {self.k}/{self.n})"


def compute_metrics(records: Sequence[Record], outcomes: Sequence[Outcome]) -> dict[str, Any]:
    by_id = {o.id: o for o in outcomes}
    pairs = [(r, by_id[r["id"]]) for r in records]
    errors = [(r, o) for r, o in pairs if r["error_line"] is not None]
    correct = [(r, o) for r, o in pairs if r["error_line"] is None]

    localized = [(r, o) for r, o in errors if o.first_error == r["error_line"]]
    typed = [(r, o) for r, o in localized if o.predicted_type == r["error_type"]]
    in_top = [
        (r, o) for r, o in localized if r["error_type"] in (o.predicted_type, *o.alternatives)
    ]
    latencies = sorted(o.latency_ms for o in outcomes if o.crash is None)

    per_type: dict[str, dict[str, Rate]] = {}
    for t in sorted({r["error_type"] for r, _ in errors}):
        group = [(r, o) for r, o in errors if r["error_type"] == t]
        loc = [(r, o) for r, o in group if o.first_error == r["error_line"]]
        per_type[t] = {
            "localization": Rate(len(loc), len(group)),
            "type_given_localized": Rate(sum(o.predicted_type == t for _, o in loc), len(loc)),
        }

    confusion: dict[str, Counter[str]] = defaultdict(Counter)
    for r, o in localized:
        confusion[r["error_type"]][o.predicted_type or "none"] += 1

    per_family: dict[str, dict[str, Rate]] = {}
    for fam in sorted({r["family"] for r in records}):
        fe = [(r, o) for r, o in errors if r["family"] == fam]
        fc = [(r, o) for r, o in correct if r["family"] == fam]
        per_family[fam] = {
            "localization": Rate(sum(o.first_error == r["error_line"] for r, o in fe), len(fe)),
            "false_alarm": Rate(sum(o.status == "error" for _, o in fc), len(fc)),
        }

    return {
        "n": len(records),
        "n_errors": len(errors),
        "n_correct": len(correct),
        "detection": Rate(sum(o.status == "error" for _, o in errors), len(errors)),
        "localization": Rate(len(localized), len(errors)),
        "flagged_too_early": Rate(
            sum(o.first_error is not None and o.first_error < r["error_line"] for r, o in errors),
            len(errors),
        ),
        "type_given_localized": Rate(len(typed), len(localized)),
        "type_in_alternatives": Rate(len(in_top), len(localized)),
        "end_to_end": Rate(len(typed), len(errors)),
        "false_alarm": Rate(sum(o.status == "error" for _, o in correct), len(correct)),
        "unverified_on_correct": Rate(
            sum(o.status in ("unverified", "incomplete") for _, o in correct), len(correct)
        ),
        "crashes": sum(o.crash is not None for o in outcomes),
        "latency_ms": {
            "p50": statistics.median(latencies) if latencies else 0.0,
            "p95": latencies[int(0.95 * (len(latencies) - 1))] if latencies else 0.0,
            "max": latencies[-1] if latencies else 0.0,
        },
        "per_type": per_type,
        "confusion": {k: dict(v) for k, v in confusion.items()},
        "per_family": per_family,
    }


def render_markdown(m: dict[str, Any], command: str, workers: int) -> str:
    lat = m["latency_ms"]
    lines = [
        "# Benchmark results",
        "",
        f"`{command}`",
        "",
        f"{m['n']} solutions: {m['n_errors']} with exactly one injected mistake, "
        f"{m['n_correct']} fully correct. Rates show the 95% Wilson interval and counts.",
        "",
        "| Metric | Result |",
        "|---|---|",
        f"| **Error localization** (exact wrong line) | {m['localization']} |",
        f"| **Error type accuracy** (given correct line) | {m['type_given_localized']} |",
        f"| End-to-end (right line *and* right type) | {m['end_to_end']} |",
        f"| **False alarms on correct solutions** | {m['false_alarm']} |",
        f"| Blamed a correct line before the real mistake | {m['flagged_too_early']} |",
        f"| Mistake detected at all | {m['detection']} |",
        f"| True type among alternatives | {m['type_in_alternatives']} |",
        f"| Correct solutions left unverified/incomplete | {m['unverified_on_correct']} |",
        f"| Crashes | {m['crashes']} |",
        f"| Latency p50 / p95 / max ({workers} parallel workers) "
        f"| {lat['p50']:.0f} / {lat['p95']:.0f} / {lat['max']:.0f} ms |",
        "",
        "## By error type",
        "",
        "| Injected error | Localization | Type accuracy (given line) |",
        "|---|---|---|",
    ]
    for t, rates in m["per_type"].items():
        lines.append(f"| {t} | {rates['localization']} | {rates['type_given_localized']} |")
    lines += [
        "",
        "## By problem family",
        "",
        "| Family | Localization | False alarms |",
        "|---|---|---|",
    ]
    for fam, rates in m["per_family"].items():
        lines.append(f"| {fam} | {rates['localization']} | {rates['false_alarm']} |")

    predicted = sorted({p for row in m["confusion"].values() for p in row})
    lines += [
        "",
        "## Confusion matrix (localized errors; rows = injected, columns = predicted)",
        "",
        "| injected \\ predicted | " + " | ".join(predicted) + " |",
        "|---" * (len(predicted) + 1) + "|",
    ]
    for t, row in sorted(m["confusion"].items()):
        lines.append(f"| {t} | " + " | ".join(str(row.get(p, 0)) for p in predicted) + " |")
    lines += ["", CAVEATS]
    return "\n".join(lines) + "\n"


CAVEATS = """## How to read these numbers

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
"""


def load(path: Path, limit: int | None) -> list[Record]:
    with path.open(encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    return records[:limit] if limit else records


def run_all(records: Iterable[Record], workers: int) -> list[Outcome]:
    records = list(records)
    if workers <= 1:
        return [run_one(r) for r in records]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(run_one, records, chunksize=8))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--report", type=Path, default=Path("docs/results.md"))
    parser.add_argument("--outcomes", type=Path, default=Path("data/outcomes.jsonl"))
    args = parser.parse_args()

    records = load(args.dataset, args.limit)
    start = time.perf_counter()
    outcomes = run_all(records, args.workers)
    print(f"checked {len(records)} solutions in {time.perf_counter() - start:.0f} s")

    args.outcomes.parent.mkdir(parents=True, exist_ok=True)
    with args.outcomes.open("w", encoding="utf-8") as f:
        for o in outcomes:
            f.write(json.dumps(asdict(o)) + "\n")
    command = "python -m stepcheck.eval.benchmark " + " ".join(
        a for a in [str(args.dataset), f"--limit {args.limit}" if args.limit else ""] if a
    )
    report = render_markdown(compute_metrics(records, outcomes), command, args.workers)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
