import os
from pathlib import Path

import pytest

from stepcheck.ocr.pix2text_backend import group_into_lines


def piece(text: str, line: int, x: float, score: float = 0.99) -> dict[str, object]:
    y = 100.0 * line
    return {
        "type": "isolated",
        "text": text,
        "position": [[x, y], [x + 50, y], [x + 50, y + 40], [x, y + 40]],
        "score": score,
        "line_number": line,
    }


def test_pieces_on_one_line_are_joined_left_to_right() -> None:
    # Pix2Text reports "x = 2", the word "or", and "x = 3" as separate pieces.
    lines = group_into_lines(
        [
            piece("x=3", 1, 300),
            piece("x=2", 1, 10),
            piece("or", 1, 150, 0.7),
            piece("x^2=5x-6", 0, 10),
        ]
    )
    assert [line.latex for line in lines] == ["x^2=5x-6", "x=2 or x=3"]
    assert lines[1].confidence == 0.7  # as reliable as its weakest piece
    assert lines[1].box == (10, 100, 350, 140)


def test_empty_pieces_are_ignored() -> None:
    assert group_into_lines([piece("  ", 0, 0)]) == []


@pytest.mark.skipif(
    not os.environ.get("STEPCHECK_OCR_TESTS"),
    reason="runs the real model (slow); set STEPCHECK_OCR_TESTS=1",
)
def test_real_model_on_rendered_handwriting(tmp_path: Path) -> None:
    from PIL import Image, ImageDraw, ImageFont

    from stepcheck.ocr.pipeline import get_backend, read_solution

    lines = ["2x + 3 = 7", "2x = 4", "x = 2"]
    font = ImageFont.truetype("C:/Windows/Fonts/Inkfree.ttf", 56)
    image = Image.new("RGB", (700, 120 * len(lines) + 40), "white")
    draw = ImageDraw.Draw(image)
    for i, text in enumerate(lines):
        draw.text((50, 30 + 120 * i), text, fill="black", font=font)
    path = tmp_path / "solution.png"
    image.save(path)

    result = read_solution(path, get_backend("pix2text"))
    assert [line.latex.replace(" ", "") for line in result.lines] == [
        line.replace(" ", "") for line in lines
    ]
