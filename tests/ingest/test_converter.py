from pathlib import Path
from unittest.mock import patch

import pytest

from docgraph.ingest.converter import _convert_pdf, convert_file_to_markdown

FIXTURES = Path(__file__).parent.parent / "fixtures"


def test_convert_md_file():
    md = convert_file_to_markdown(FIXTURES / "sample.md")
    assert "# Sample" in md
    assert "test document" in md


def test_convert_pdf_falls_back_to_pymupdf(tmp_path):
    pdf = tmp_path / "book.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")

    with patch(
        "docgraph.ingest.converter._convert_markitdown",
        side_effect=RuntimeError("Invalid octal"),
    ):
        with patch(
            "docgraph.ingest.converter._pdf_via_pymupdf",
            return_value="# Chapter 1\n\nNeural networks.",
        ) as pymupdf_mock:
            text = _convert_pdf(pdf)

    assert "Chapter 1" in text
    pymupdf_mock.assert_called_once_with(pdf)


def test_convert_pdf_raises_when_all_backends_fail(tmp_path):
    pdf = tmp_path / "bad.pdf"
    pdf.write_bytes(b"%PDF-1.4")

    with patch(
        "docgraph.ingest.converter._convert_markitdown",
        side_effect=RuntimeError("markitdown fail"),
    ):
        with patch(
            "docgraph.ingest.converter._pdf_via_pymupdf",
            side_effect=ValueError("no text"),
        ):
            with pytest.raises(ValueError, match="could not convert PDF"):
                _convert_pdf(pdf)
