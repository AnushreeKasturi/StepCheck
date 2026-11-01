import pytest

from stepcheck.engine.parse import parse_math
from stepcheck.ocr.normalize import normalize, normalize_line


@pytest.mark.parametrize(
    ("raw", "clean"),
    [
        ("$x = 3$", "x = 3"),
        ("$$x^{2} = 9$$", "x^{2} = 9"),
        (r"\[ 2x = 6 \]", "2x = 6"),
        (r"\Rightarrow x = 3", "x = 3"),
        ("⇒ x = 3", "x = 3"),
        (r"\therefore x = 3", "x = 3"),
        (r"\boxed{x = 3}", "x = 3"),
        (r"x = 2 \text {or} x = 3", r"x = 2 \text{or} x = 3"),
        (r"\mathrm{x} + 1", "x + 1"),
        (r"\textsc{x}=O", "x=0"),  # real Pix2Text output for a handwritten "x = 0"
        ("x = 1O", "x = 10"),
        (r"x = 2 \text{or} x = 3", r"x = 2 \text{or} x = 3"),  # words are left alone
        ("x = 3.", "x = 3"),
        ("2·x", r"2 \cdot x"),
        ("x² = 4", "x^2 = 4"),
        (r"\displaystyle \frac{x}{2}", r"\frac{x}{2}"),
        ("  x   +   1 ", "x + 1"),
    ],
)
def test_normalize_line(raw: str, clean: str) -> None:
    assert normalize_line(raw) == clean


def test_aligned_block_becomes_one_line_per_step() -> None:
    block = r"\begin{aligned} 2x + 3 &= 7 \\ 2x &= 4 \\ x &= 2 \end{aligned}"
    assert normalize(block) == ["2x + 3 = 7", "2x = 4", "x = 2"]


def test_array_block_with_column_spec() -> None:
    block = r"\begin{array}{l} x^{2} = 9 \\[2pt] x = \pm 3 \end{array}"
    assert normalize(block) == ["x^{2} = 9", r"x = \pm 3"]


def test_regression_pix2text_array_with_braced_rows() -> None:
    # Real Pix2Text output for a four-line solution: a space before {l}, and every row wrapped
    # in {{...}}. The braces hid the '=' from the parser, so lines 2-4 came out unreadable.
    raw = (
        r"\begin{array} {l} {{5-\left( x-3 \right)=2}} \\ {{5-x-3=2}} \\ {{2-x=2}} \\ {{x=0}}"
        r" \\ \end{array}"
    )
    lines = normalize(raw)
    assert lines == [r"5-\left( x-3 \right)=2", "5-x-3=2", "2-x=2", "x=0"]
    for line in lines:
        parse_math(line)


@pytest.mark.parametrize(
    ("raw", "clean"),
    [("{{x=0}}", "x=0"), ("{x}+{1}", "{x}+{1}"), ("{", "{")],  # only braces around everything
)
def test_strip_outer_braces(raw: str, clean: str) -> None:
    assert normalize_line(raw) == clean


def test_empty_pieces_are_dropped() -> None:
    assert normalize(r"x = 1 \\ \\ ") == ["x = 1"]


@pytest.mark.parametrize(
    "raw", [r"\Rightarrow x = 2 \text {or} x = 3", r"\boxed{x = -1}.", "x² - 5x = -6"]
)
def test_normalized_output_parses(raw: str) -> None:
    for line in normalize(raw):
        parse_math(line)
