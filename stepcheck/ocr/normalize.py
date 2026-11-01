"""Turn OCR output into the LaTeX dialect the engine's parser reads.

Every rule here exists because OCR output or student handwriting actually looks like that;
each one has a test.
"""

import re

_DELIMITERS = [r"^\$\$|\$\$$", r"^\$|\$$", r"^\\\[|\\\]$", r"^\\\(|\\\)$"]
_BLOCK = re.compile(
    r"\\begin\{(aligned|align\*?|array|gathered|split|cases)\}\s*(\{[^}]*\})?|\\end\{\w+\*?\}"
)
# Arrows and "therefore" that students put in front of a step: they are not part of the math.
_LEADING_CONNECTIVES = re.compile(
    r"^(\\Rightarrow|\\implies|\\Longrightarrow|\\rightarrow|\\to|\\therefore|⇒|→|∴|=>)\s*"
)
_UNICODE = {
    "\N{MIDDLE DOT}": r" \cdot ",
    "\N{MULTIPLICATION SIGN}": r" \times ",
    "\N{DIVISION SIGN}": r" \div ",
    "\N{SUPERSCRIPT TWO}": "^2",
    "\N{SUPERSCRIPT THREE}": "^3",
    "\N{SQUARE ROOT}": r"\sqrt",
}
_FONT_LETTER = r"\\(mathrm|mathit|mathbf|textsc|textit|textrm|text)\{\s*([a-zA-Z])\s*\}"
_DROP = [r"\\displaystyle", r"\\textstyle", r"\\checkmark", r"\\quad", r"\\qquad"]


def split_lines(latex: str) -> list[str]:
    """One recognized block can contain several steps (`aligned` environments, `\\\\`)."""
    body = _BLOCK.sub("", latex)
    return [part for part in re.split(r"\\\\(?:\[[^\]]*\])?", body) if part.strip()]


def _closing_brace(s: str) -> int:
    """Index of the brace that closes the one at s[0], or -1."""
    depth = 0
    for i, ch in enumerate(s):
        depth += {"{": 1, "}": -1}.get(ch, 0)
        if depth == 0:
            return i
    return -1


def strip_outer_braces(s: str) -> str:
    """{{5 - x = 2}} -> 5 - x = 2. Braces around a whole line hide its '=' from the parser
    (Pix2Text wraps every row of an array like this)."""
    while s.startswith("{") and _closing_brace(s) == len(s) - 1:
        s = s[1:-1].strip()
    return s


def normalize_line(latex: str) -> str:
    s = latex.strip()
    for pattern in _DELIMITERS:
        s = re.sub(pattern, "", s).strip()
    s = strip_outer_braces(s)
    s = s.replace("&", " ")  # alignment markers from aligned environments
    for pattern in _DROP:
        s = re.sub(pattern, " ", s)
    s = re.sub(r"\\boxed\s*\{(.*)\}", r"\1", s)  # students box the final answer
    s = re.sub(r"\\(text|mathrm|textrm|operatorname)\s+\{", r"\\\1{", s)  # "\text {or}"
    # A single letter in another font is just that letter: \mathrm{x}, \textsc{x} (Pix2Text)
    s = re.sub(_FONT_LETTER, r"\2", s)
    # A capital O is never a variable in this algebra: it is a misread zero (seen from Pix2Text).
    s = re.sub(r"(?<![A-Za-z\\])O(?![A-Za-z])", "0", s)
    for symbol, latex_form in _UNICODE.items():
        s = s.replace(symbol, latex_form)
    s = _LEADING_CONNECTIVES.sub("", s.strip())
    s = re.sub(r"[.,;]+$", "", s.strip())  # sentence punctuation after the last step
    return re.sub(r"\s+", " ", s).strip()


def normalize(latex: str) -> list[str]:
    """Recognized text -> zero or more clean lines."""
    return [line for line in (normalize_line(p) for p in split_lines(latex)) if line]
