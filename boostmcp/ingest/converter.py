from __future__ import annotations

import logging
from pathlib import Path

from markitdown import MarkItDown

logger = logging.getLogger(__name__)

_converter = MarkItDown(enable_plugins=False)
_PDF_SUFFIXES = {".pdf"}


def _pdf_via_pymupdf(path: Path) -> str:
    import fitz  # pymupdf

    doc = fitz.open(path)
    try:
        parts = [page.get_text("text") for page in doc]
    finally:
        doc.close()
    text = "\n\n".join(p.strip() for p in parts if p and p.strip())
    if not text.strip():
        raise ValueError("no text extracted")
    return text


def _convert_markitdown(path: Path) -> str:
    result = _converter.convert_local(str(path))
    text = result.text_content or ""
    if not text.strip():
        raise ValueError("empty output")
    return text


def _convert_pdf(path: Path) -> str:
    errors: list[str] = []

    try:
        return _convert_markitdown(path)
    except Exception as exc:
        errors.append(f"markitdown: {exc}")
        logger.info("PDF markitdown failed for %s, trying pymupdf: %s", path.name, exc)

    try:
        return _pdf_via_pymupdf(path)
    except Exception as exc:
        errors.append(f"pymupdf: {exc}")
        logger.warning("PDF pymupdf failed for %s: %s", path.name, exc)

    raise ValueError(
        f"could not convert PDF {path.name}. "
        + "; ".join(errors[:2])
    )


def convert_file_to_markdown(path: Path) -> str:
    """Convert a local file to markdown. PDFs use pymupdf fallback when markitdown fails."""
    suffix = path.suffix.lower()
    if suffix in _PDF_SUFFIXES:
        return _convert_pdf(path)
    text = _convert_markitdown(path)
    return text
