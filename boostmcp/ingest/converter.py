from __future__ import annotations

from pathlib import Path

from markitdown import MarkItDown

_converter = MarkItDown(enable_plugins=False)


def convert_file_to_markdown(path: Path) -> str:
    """Convert local file to markdown using MarkItDown convert_local."""
    result = _converter.convert_local(str(path))
    text = result.text_content or ""
    if not text.strip():
        raise ValueError(f"conversion produced empty output: {path.name}")
    return text
