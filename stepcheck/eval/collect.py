"""Turn hand-labelled transcriptions into a benchmark dataset.

One text file per photographed solution, e.g. data/real/solutions/s07_p2.txt:

    % problem: p2
    % error_line: 2
    % error_type: sign_error
    7 - (2x - 5) = 3x - 8
    7 - 2x - 5 = 3x - 8
    ...

Plain LaTeX lines (no JSON escaping), and `python -m stepcheck check` reads the same file.
Use `error_line: none` for a fully correct solution.

Usage: python -m stepcheck.eval.collect data/real/solutions --out data/real/labels.jsonl
"""

import argparse
import json
from pathlib import Path
from typing import Any

ERROR_TYPES = {
    "sign_error",
    "partial_distribution",
    "power_of_sum",
    "invalid_cancellation",
    "moved_term_no_sign_change",
    "inverse_operation",
    "arithmetic_slip",
    "divided_by_zero_expression",
    "missing_plus_minus",
    "extraneous_not_rejected",
    "other",
}
REQUIRED = ("problem", "error_line")


class LabelError(ValueError):
    pass


def parse_solution_file(path: Path, record_id: int) -> dict[str, Any]:
    headers: dict[str, str] = {}
    lines: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line.startswith("%"):
            key, sep, value = line[1:].partition(":")
            if sep:
                headers[key.strip().lower()] = value.strip()
        elif line:
            lines.append(line)

    missing = [k for k in REQUIRED if k not in headers]
    if missing:
        raise LabelError(f"{path.name}: missing header(s) {', '.join(missing)}")
    if not lines:
        raise LabelError(f"{path.name}: no solution lines")

    if headers["error_line"].lower() in ("none", "null", "-", ""):
        error_line, error_type = None, None
    else:
        try:
            error_line = int(headers["error_line"])
        except ValueError:
            raise LabelError(f"{path.name}: error_line must be a number or 'none'") from None
        if not 2 <= error_line <= len(lines):
            raise LabelError(f"{path.name}: error_line {error_line} is not a line 2..{len(lines)}")
        error_type = headers.get("error_type", "other")
        if error_type not in ERROR_TYPES:
            raise LabelError(f"{path.name}: unknown error_type '{error_type}'")

    return {
        "id": record_id,
        "family": headers["problem"],
        "source": path.name,
        "lines": lines,
        "error_line": error_line,
        "error_type": error_type,
        **({"note": headers["note"]} if "note" in headers else {}),
    }


def collect(folder: Path) -> list[dict[str, Any]]:
    files = sorted(folder.glob("*.txt"))
    if not files:
        raise LabelError(f"no .txt files in {folder}")
    return [parse_solution_file(f, i) for i, f in enumerate(files)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("folder", type=Path)
    parser.add_argument("--out", type=Path, default=Path("data/real/labels.jsonl"))
    args = parser.parse_args()
    try:
        records = collect(args.folder)
    except LabelError as exc:
        parser.exit(1, f"error: {exc}\n")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    n_errors = sum(r["error_line"] is not None for r in records)
    print(f"wrote {len(records)} solutions ({n_errors} with a mistake) to {args.out}")


if __name__ == "__main__":
    main()
