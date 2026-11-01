from pathlib import Path

import pytest

from stepcheck.ocr.base import OcrLine, OcrResult
from stepcheck.ocr.pipeline import clean, get_backend, read_solution


class FakeBackend:
    name = "fake"

    def __init__(self, *blocks: tuple[str, float]) -> None:
        self.blocks = blocks

    def recognize(self, image: Path) -> OcrResult:
        lines = tuple(
            OcrLine(text, conf, (0, i * 10, 100, i * 10 + 9))
            for i, (text, conf) in enumerate(self.blocks)
        )
        return OcrResult(lines, self.name, 5.0)


def test_blocks_are_split_and_normalized(tmp_path: Path) -> None:
    image = tmp_path / "photo.jpg"
    image.write_bytes(b"")
    backend = FakeBackend(
        (r"$2x + 3 = 7$", 0.9), (r"\begin{aligned} 2x &= 4 \\ x &= 2 \end{aligned}", 0.8)
    )
    result = read_solution(image, backend)
    assert [line.latex for line in result.lines] == ["2x + 3 = 7", "2x = 4", "x = 2"]
    assert [line.confidence for line in result.lines] == [0.9, 0.8, 0.8]
    assert result.lines[2].box == result.lines[1].box  # split lines keep their block's box


def test_unparseable_line_gets_zero_confidence() -> None:
    raw = OcrResult((OcrLine(r"x = \frac{3}{", 0.95),), "fake", 1.0)
    assert clean(raw).lines[0].confidence == 0.0


def test_missing_image(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        read_solution(tmp_path / "nope.jpg", FakeBackend())


def test_unknown_backend() -> None:
    with pytest.raises(ValueError, match="unknown OCR backend"):
        get_backend("tesseract")
