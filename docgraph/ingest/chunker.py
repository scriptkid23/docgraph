from __future__ import annotations

# Try increasingly finer separators so we cut at the most natural boundary that
# still fits in the size budget. Ordered coarse → fine.
_SEPARATORS: tuple[str, ...] = (
    "\n## ",
    "\n### ",
    "\n#### ",
    "\n# ",
    "\n\n",
    "\n",
    ". ",
    " ",
    "",
)


def _split_recursive(
    text: str, max_chars: int, start_idx: int = 0
) -> list[str]:
    """Recursively split text down to pieces ≤ max_chars, preferring natural boundaries."""
    if len(text) <= max_chars:
        return [text]
    for i in range(start_idx, len(_SEPARATORS)):
        sep = _SEPARATORS[i]
        if sep == "":
            return [text[j : j + max_chars] for j in range(0, len(text), max_chars)]
        if sep not in text:
            continue
        parts = text.split(sep)
        out: list[str] = []
        for idx, part in enumerate(parts):
            # Re-attach the separator (except for the first part) so heading
            # markers and sentence terminators survive the split. Recurse with
            # i+1 so we move on to finer separators — otherwise the re-attached
            # separator at the start of `piece` would re-trigger the same split
            # forever on a single oversized section.
            piece = part if idx == 0 else sep + part
            if len(piece) <= max_chars:
                out.append(piece)
            else:
                out.extend(_split_recursive(piece, max_chars, i + 1))
        return out
    return [text]


def _merge_with_overlap(
    pieces: list[str], max_chars: int, overlap_chars: int
) -> list[str]:
    """Greedy-merge adjacent pieces up to max_chars; carry overlap_chars between chunks."""
    chunks: list[str] = []
    current = ""
    for piece in pieces:
        if not piece:
            continue
        if not current:
            current = piece
            continue
        if len(current) + len(piece) <= max_chars:
            current += piece
            continue
        chunks.append(current.strip())
        if overlap_chars > 0 and len(current) > overlap_chars:
            tail = current[-overlap_chars:]
            current = tail + piece
        else:
            current = piece
    if current.strip():
        chunks.append(current.strip())
    return [c for c in chunks if c]


def chunk_markdown(
    text: str,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> list[str]:
    """Split markdown into chunks, preferring heading/paragraph boundaries.

    chunk_size and chunk_overlap are in approximate tokens (× 4 for char budget).
    """
    text = text.strip()
    if not text:
        return []
    char_size = max(1, chunk_size * 4)
    char_overlap = max(0, chunk_overlap * 4)
    if char_overlap >= char_size:
        char_overlap = char_size // 2
    if len(text) <= char_size:
        return [text]
    pieces = _split_recursive(text, char_size)
    return _merge_with_overlap(pieces, char_size, char_overlap)
