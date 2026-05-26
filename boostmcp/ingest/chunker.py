from __future__ import annotations


def chunk_markdown(
    text: str,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> list[str]:
    """Split markdown into char-based chunks (chunk_size ≈ tokens * 4)."""
    text = text.strip()
    if not text:
        return []
    char_size = chunk_size * 4
    char_overlap = chunk_overlap * 4
    if len(text) <= char_size:
        return [text]
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + char_size
        chunks.append(text[start:end].strip())
        if end >= len(text):
            break
        start = end - char_overlap
    return [c for c in chunks if c]
