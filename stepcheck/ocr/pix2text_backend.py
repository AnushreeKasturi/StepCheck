"""Pix2Text: open-source math OCR that runs locally (no API, nothing leaves the machine).

Install with `pip install -e ".[ocr]"`. The first run downloads the models (~1 GB).
"""

import contextlib
import io
import logging
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from stepcheck.ocr.base import Box, OcrLine, OcrResult, OcrUnavailable

_NOISY_LOGGERS = ("pix2text", "cnstd", "cnocr", "ultralytics", "transformers", "optimum")


def _box(position: Any) -> Box:
    xs = [float(p[0]) for p in position]
    ys = [float(p[1]) for p in position]
    return int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))


def _union(boxes: list[Box]) -> Box:
    return (
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    )


def group_into_lines(elements: list[dict[str, Any]]) -> list[OcrLine]:
    """Pix2Text returns pieces (formulas and words such as "or") tagged with a line number.

    Joins each line's pieces left to right. A line is only as reliable as its weakest piece,
    so it gets the lowest score.
    """
    by_line: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for el in elements:
        if str(el.get("text", "")).strip():
            by_line[int(el.get("line_number", 0))].append(el)
    lines = []
    for number in sorted(by_line):
        pieces = sorted(by_line[number], key=lambda el: _box(el["position"])[0])
        text = " ".join(str(el["text"]).strip() for el in pieces)
        score = min(float(el.get("score", 0.0)) for el in pieces)
        lines.append(OcrLine(text, score, _union([_box(el["position"]) for el in pieces])))
    return lines


class Pix2TextBackend:
    name = "pix2text"

    def __init__(self) -> None:
        try:
            import pix2text  # noqa: F401
        except ImportError as exc:
            raise OcrUnavailable('Pix2Text is not installed: run pip install -e ".[ocr]"') from exc
        self._model: Any = None

    def _load(self) -> Any:
        if self._model is None:
            from pix2text import Pix2Text

            for name in _NOISY_LOGGERS:
                logging.getLogger(name).setLevel(logging.ERROR)
            with _quiet():
                self._model = Pix2Text.from_config(enable_table=False)
        return self._model

    def recognize(self, image: Path) -> OcrResult:
        model = self._load()
        start = time.perf_counter()
        with _quiet():
            elements = model.recognize_text_formula(str(image), return_text=False)
        latency = (time.perf_counter() - start) * 1000
        return OcrResult(tuple(group_into_lines(list(elements))), self.name, latency)


@contextlib.contextmanager
def _quiet() -> Any:
    """Pix2Text and its dependencies print progress bars and banners; keep the CLI readable."""
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        yield
