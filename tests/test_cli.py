from pathlib import Path

import pytest

from stepcheck.cli import main, read_solution

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.mark.parametrize(
    ("fixture", "exit_code", "expected"),
    [
        ("correct_quadratic.txt", 0, "CORRECT"),
        ("sign_error.txt", 1, "[sign error] Line 2: Sign error"),
        ("divide_by_x.txt", 1, "You divided both sides by x"),
        ("simplify_error.txt", 1, "[power of sum]"),
        ("squaring_correct.txt", 0, "correctly rejects extraneous {1}"),
    ],
)
def test_fixtures(
    fixture: str, exit_code: int, expected: str, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["check", str(FIXTURES / fixture)]) == exit_code
    assert expected in capsys.readouterr().out


def test_comments_and_blank_lines_are_skipped(tmp_path: Path) -> None:
    f = tmp_path / "s.txt"
    f.write_text("% problem\n\n2x = 4\n\nx = 2\n", encoding="utf-8")
    assert read_solution(f) == ["2x = 4", "x = 2"]


def test_missing_file(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["check", "does-not-exist.txt"]) == 3
    assert "stepcheck:" in capsys.readouterr().err


def test_two_variables_is_bad_input(tmp_path: Path) -> None:
    f = tmp_path / "s.txt"
    f.write_text("x + y = 2\ny = 2 - x\n", encoding="utf-8")
    assert main(["check", str(f)]) == 3
