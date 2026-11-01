from pathlib import Path

import pytest

from stepcheck.cli import main, read_solution
from stepcheck.ocr.base import OcrLine, OcrResult

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


class _FakeOcr:
    name = "fake"

    def recognize(self, image: Path) -> OcrResult:
        lines = ("5 - (x - 3) = 2", "5 - x - 3 = 2", r"x = \frac{0}{")
        return OcrResult(tuple(OcrLine(t, 0.9) for t in lines), "fake", 12.0)


@pytest.fixture
def photo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    import stepcheck.ocr.pipeline as pipeline

    monkeypatch.setattr(pipeline, "get_backend", lambda name: _FakeOcr())
    image = tmp_path / "photo.jpg"
    image.write_bytes(b"not really a jpeg")
    return image


def test_ocr_prints_lines_flags_unreadable_and_saves(
    photo: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "lines.txt"
    assert main(["ocr", str(photo), "--out", str(out)]) == 0
    printed = capsys.readouterr().out
    assert "5 - x - 3 = 2" in printed
    assert "?? " in printed  # the unreadable third line is flagged
    assert read_solution(out) == ["5 - (x - 3) = 2", "5 - x - 3 = 2", r"x = \frac{0}{"]


def test_check_image_reads_then_checks(photo: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["check-image", str(photo)]) == 1
    assert "[sign error]" in capsys.readouterr().out


def test_unknown_backend_is_bad_input(photo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import stepcheck.ocr.pipeline as pipeline

    def unknown(name: str) -> None:
        raise ValueError(f"unknown OCR backend '{name}'")

    monkeypatch.setattr(pipeline, "get_backend", unknown)
    assert main(["ocr", str(photo), "--backend", "nope"]) == 3
