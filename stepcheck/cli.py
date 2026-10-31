"""Command line: `python -m stepcheck check solution.txt`.

Input file: one LaTeX line per step. Blank lines and lines starting with % are ignored.
Exit codes: 0 correct, 1 error found, 2 incomplete or unverified, 3 unusable input.
"""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import sympy as sp

from stepcheck.engine.checker import UnsupportedInput, check_solution
from stepcheck.engine.model import CheckResult, StepCheck, Verdict

EXIT_CODES = {"correct": 0, "error": 1, "incomplete": 2, "unverified": 2}
EXIT_BAD_INPUT = 3


def read_solution(path: Path) -> list[str]:
    lines = (line.strip() for line in path.read_text(encoding="utf-8").splitlines())
    return [line for line in lines if line and not line.startswith("%")]


def _fmt_set(s: sp.Set) -> str:
    return str(s).replace("Interval.open", "").replace("Interval", "")


def describe(check: StepCheck) -> str:
    prev = f"line {check.from_line}"
    match check.verdict:
        case Verdict.EQUIVALENT:
            return f"ok ({check.method})"
        case Verdict.NOT_EQUIVALENT:
            where = f" (they differ at x = {check.counterexample})" if check.counterexample else ""
            return f"not equal to {prev}{where}"
        case Verdict.LOST_SOLUTIONS:
            return f"loses solution(s) {_fmt_set(check.lost)} of {prev}"
        case Verdict.EXTRA_SOLUTIONS if check.is_error:
            return (
                f"adds {_fmt_set(check.extra)}, which do not solve {prev} "
                "and are never rejected later"
            )
        case Verdict.EXTRA_SOLUTIONS:
            return (
                f"ok, but adds candidate(s) {_fmt_set(check.extra)}: "
                "check them against the original equation"
            )
        case Verdict.REJECTED_EXTRANEOUS:
            return f"ok, correctly rejects extraneous {_fmt_set(check.lost)}"
        case Verdict.LOST_AND_EXTRA:
            return (
                f"changes the solutions of {prev}: loses {_fmt_set(check.lost)}, "
                f"adds {_fmt_set(check.extra)}"
            )
        case Verdict.UNKNOWN:
            return "could not verify this step"
        case Verdict.KIND_MISMATCH:
            return f"cannot compare with {prev} (expression vs. equation)"
        case Verdict.UNPARSEABLE:
            return f"could not read this line: {check.note}"


def _marker(check: StepCheck) -> str:
    if check.is_error:
        return "XX"
    valid = {Verdict.EQUIVALENT, Verdict.EXTRA_SOLUTIONS, Verdict.REJECTED_EXTRANEOUS}
    return "ok" if check.verdict in valid else "??"


def format_result(result: CheckResult) -> str:
    by_line = {c.to_line: c for c in result.checks}
    width = max(len(s.latex) for s in result.steps)
    out = []
    for step in result.steps:
        check = by_line.get(step.line_no)
        if check is None:
            note = f"could not read: {step.parse_error}" if step.parse_error else "(start)"
            out.append(f"     {step.line_no:>2}  {step.latex:<{width}}  {note}")
        else:
            out.append(
                f"  {_marker(check)} {step.line_no:>2}  {step.latex:<{width}}  {describe(check)}"
            )
    out.append("")
    out.append(_summary(result))
    return "\n".join(out)


def _summary(result: CheckResult) -> str:
    match result.status:
        case "correct":
            return "Result: CORRECT - every step follows from the previous one."
        case "incomplete":
            return "Result: INCOMPLETE - every step is valid, but the final line is not solved."
        case "error" if result.first_error is not None:
            return f"Result: ERROR - the first mistake is on line {result.first_error}."
        case "error":
            assert result.final_check is not None
            return (
                "Result: ERROR - the final answer does not match the problem "
                f"({describe(result.final_check)}), but the wrong step could not be pinned down."
            )
        case _:
            return "Result: UNVERIFIED - no mistakes found, but some steps could not be checked."


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="stepcheck")
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("check", help="check a step-by-step solution")
    check.add_argument("file", type=Path, help="one LaTeX line per step")
    args = parser.parse_args(argv)

    try:
        lines = read_solution(args.file)
        if not lines:
            raise UnsupportedInput("the file has no lines to check")
        result = check_solution(lines)
    except (OSError, UnicodeDecodeError, UnsupportedInput) as exc:
        print(f"stepcheck: {exc}", file=sys.stderr)
        return EXIT_BAD_INPUT

    print(format_result(result))
    return EXIT_CODES[result.status]
