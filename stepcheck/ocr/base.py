"""The OCR interface: a photo of a handwritten solution in, one LaTeX string per line out."""

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

Box = tuple[int, int, int, int]  # left, top, right, bottom in image pixels


@dataclass(frozen=True)
class OcrLine:
    latex: str
    confidence: float  # 0..1; 0 means "the parser cannot read this", so the user must check it
    box: Box | None = None  # where the line is on the photo, for highlighting in the UI


@dataclass(frozen=True)
class OcrResult:
    lines: tuple[OcrLine, ...]
    backend: str
    latency_ms: float


class OcrUnavailable(RuntimeError):
    """The backend's optional dependencies or local model are not installed."""


class OcrBackend(Protocol):
    name: str

    def recognize(self, image: Path) -> OcrResult: ...
