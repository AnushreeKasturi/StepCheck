from pathlib import Path

import pytest

from stepcheck.eval.collect import LabelError, collect, parse_solution_file


def write(folder: Path, name: str, text: str) -> Path:
    path = folder / name
    path.write_text(text, encoding="utf-8")
    return path


def test_parses_headers_and_latex_without_escaping(tmp_path: Path) -> None:
    f = write(
        tmp_path,
        "s07_p3.txt",
        "% problem: p3\n% error_line: 2\n% error_type: other\n% note: multiplied some terms\n"
        "\\frac{x}{3} + 5 = 2x - 5\n\nx + 5 = 6x - 5\n10 = 5x\nx = 2\n",
    )
    r = parse_solution_file(f, 4)
    assert r["lines"][0] == r"\frac{x}{3} + 5 = 2x - 5"
    assert (r["id"], r["family"], r["error_line"], r["error_type"]) == (4, "p3", 2, "other")
    assert r["note"] == "multiplied some terms"


def test_correct_solution(tmp_path: Path) -> None:
    f = write(tmp_path, "a.txt", "% problem: p1\n% error_line: none\n2x = 4\nx = 2\n")
    r = parse_solution_file(f, 0)
    assert r["error_line"] is None and r["error_type"] is None


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("% error_line: 2\nx = 1\nx = 2\n", "missing header"),
        ("% problem: p1\n% error_line: 5\nx = 1\nx = 2\n", "not a line"),
        ("% problem: p1\n% error_line: 1\nx = 1\nx = 2\n", "not a line"),  # line 1 is the problem
        (
            "% problem: p1\n% error_line: 2\n% error_type: typo\nx = 1\nx = 2\n",
            "unknown error_type",
        ),
        ("% problem: p1\n% error_line: two\nx = 1\n", "must be a number"),
    ],
)
def test_label_mistakes_are_reported(tmp_path: Path, text: str, message: str) -> None:
    with pytest.raises(LabelError, match=message):
        parse_solution_file(write(tmp_path, "bad.txt", text), 0)


def test_collect_is_sorted_by_file_name(tmp_path: Path) -> None:
    write(tmp_path, "s02_p1.txt", "% problem: p1\n% error_line: none\nx = 1\n")
    write(tmp_path, "s01_p1.txt", "% problem: p1\n% error_line: none\nx = 1\n")
    assert [r["source"] for r in collect(tmp_path)] == ["s01_p1.txt", "s02_p1.txt"]
