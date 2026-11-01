"""Photo -> clean, parseable lines with confidences. Backends only recognize; this does the rest."""

from pathlib import Path

from stepcheck.engine.parse import parse_line
from stepcheck.ocr.base import OcrBackend, OcrLine, OcrResult
from stepcheck.ocr.normalize import normalize

BACKENDS = ("pix2text",)


def get_backend(name: str) -> OcrBackend:
    """Backends are imported lazily: their heavy dependencies are an optional install."""
    if name == "pix2text":
        from stepcheck.ocr.pix2text_backend import Pix2TextBackend

        return Pix2TextBackend()
    raise ValueError(f"unknown OCR backend '{name}' (choose from {', '.join(BACKENDS)})")


def clean(raw: OcrResult) -> OcrResult:
    """Normalize every recognized block into lines; unreadable lines get confidence 0."""
    lines: list[OcrLine] = []
    for block in raw.lines:
        for text in normalize(block.latex):
            readable = parse_line(text, 0).parsed is not None
            lines.append(OcrLine(text, block.confidence if readable else 0.0, block.box))
    return OcrResult(tuple(lines), raw.backend, raw.latency_ms)


def read_solution(image: Path, backend: OcrBackend) -> OcrResult:
    if not image.is_file():
        raise FileNotFoundError(image)
    return clean(backend.recognize(image))
