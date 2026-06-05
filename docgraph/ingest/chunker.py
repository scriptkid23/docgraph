from __future__ import annotations

import logging
from dataclasses import dataclass

from docgraph.ingest.markdown_blocks import Span, SpanKind, segment_blocks
from docgraph.ingest.markdown_sections import Section, walk_sections
from docgraph.ingest.tokenizer import CharRatioCounter, TokenCounter

logger = logging.getLogger(__name__)

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

_CODE_SEPARATORS: tuple[str, ...] = (
    "\nclass ",
    "\ndef ",
    "\nfunc ",
    "\nfunction ",
    "\n\n",
    "\n",
    " ",
    "",
)


@dataclass
class MarkdownChunk:
    text: str
    heading_path: list[str]
    tokens: int = 0
    partial: bool = False
    overflow: bool = False


def _budget_units(tokens: int, counter: TokenCounter) -> int:
    return max(1, tokens)


def _size(text: str, counter: TokenCounter) -> int:
    return counter.count(text)


def _split_recursive(
    text: str,
    max_units: int,
    separators: tuple[str, ...],
    counter: TokenCounter,
    start_idx: int = 0,
) -> list[str]:
    if _size(text, counter) <= max_units:
        return [text]
    for i in range(start_idx, len(separators)):
        sep = separators[i]
        if sep == "":
            return _hard_split(text, max_units, counter)
        if sep not in text:
            continue
        parts = text.split(sep)
        out: list[str] = []
        for idx, part in enumerate(parts):
            piece = part if idx == 0 else sep + part
            if _size(piece, counter) <= max_units:
                out.append(piece)
            else:
                out.extend(_split_recursive(piece, max_units, separators, counter, i + 1))
        return out
    return [text]


def _hard_split(text: str, max_units: int, counter: TokenCounter) -> list[str]:
    if _size(text, counter) <= max_units:
        return [text]
    out: list[str] = []
    remaining = text
    while remaining and _size(remaining, counter) > max_units:
        piece = counter.truncate(remaining, max_units)
        if not piece:
            break
        out.append(piece)
        remaining = remaining[len(piece) :]
    if remaining.strip():
        out.append(remaining)
    return out


def _merge_with_overlap(
    pieces: list[str],
    max_units: int,
    overlap_units: int,
    counter: TokenCounter,
) -> list[str]:
    chunks: list[str] = []
    current = ""
    for piece in pieces:
        if not piece:
            continue
        if not current:
            current = piece
            continue
        combined = current + piece
        if _size(combined, counter) <= max_units:
            current = combined
            continue
        chunks.append(current.strip())
        if overlap_units > 0 and _size(current, counter) > overlap_units:
            tail = counter.truncate_tail(current, overlap_units)
            current = tail + piece
        else:
            current = piece
    if current.strip():
        chunks.append(current.strip())
    return [c for c in chunks if c]


def _chunk_text(
    text: str,
    separators: tuple[str, ...],
    chunk_size: int,
    chunk_overlap: int,
    counter: TokenCounter,
) -> list[str]:
    text = text.strip()
    if not text:
        return []
    max_units = _budget_units(chunk_size, counter)
    overlap_units = max(0, _budget_units(chunk_overlap, counter))
    if overlap_units >= max_units:
        overlap_units = max_units // 2
    if _size(text, counter) <= max_units:
        return [text]
    pieces = _split_recursive(text, max_units, separators, counter)
    return _merge_with_overlap(pieces, max_units, overlap_units, counter)


def _format_prefix(heading_path: list[str]) -> str:
    if not heading_path:
        return ""
    return " > ".join(heading_path) + "\n\n"


def _split_table_rows(span: Span, max_units: int, counter: TokenCounter) -> list[str]:
    lines = span.text.splitlines()
    if len(lines) < 2:
        return [span.text]
    header = span.header_row or lines[0]
    divider = lines[1]
    rows = lines[2:]
    prefix = header + "\n" + divider + "\n"
    prefix_tokens = _size(prefix, counter)
    body_budget = max(1, max_units - prefix_tokens)
    parts: list[str] = []
    batch: list[str] = []
    for row in rows:
        candidate = prefix + "\n".join(batch + [row])
        if batch and _size(candidate, counter) > max_units:
            parts.append(prefix + "\n".join(batch))
            batch = [row]
        else:
            batch.append(row)
    if batch:
        parts.append(prefix + "\n".join(batch))
    return parts or [span.text]


def _split_code_fence(span: Span, max_units: int, counter: TokenCounter) -> list[str]:
    text = span.text
    if _size(text, counter) <= max_units:
        return [text]
    lines = text.splitlines()
    if not lines:
        return [text]
    open_line = lines[0]
    close_marker = open_line[:3]
    body_lines = lines[1:-1] if lines[-1].strip().startswith(close_marker) else lines[1:]
    parts: list[str] = []
    batch: list[str] = []
    for line in body_lines:
        candidate = open_line + "\n" + "\n".join(batch + [line]) + "\n" + close_marker
        if batch and _size(candidate, counter) > max_units:
            parts.append(open_line + "\n" + "\n".join(batch) + "\n" + close_marker)
            batch = [line]
        else:
            batch.append(line)
    if batch:
        parts.append(open_line + "\n" + "\n".join(batch) + "\n" + close_marker)
    return parts or [text]


def _chunk_section_spans(
    section: Section,
    chunk_size: int,
    chunk_overlap: int,
    counter: TokenCounter,
    *,
    atomic_blocks_enabled: bool,
    heading_prefix_enabled: bool,
) -> list[MarkdownChunk]:
    prefix = _format_prefix(section.heading_path) if heading_prefix_enabled else ""
    prefix_tokens = _size(prefix, counter) if prefix else 0
    body_budget = max(1, chunk_size - prefix_tokens)
    overlap = max(0, chunk_overlap)
    if overlap >= body_budget:
        overlap = body_budget // 2

    raw_chunks: list[tuple[str, bool, bool]] = []

    spans = section.spans or segment_blocks(section.body)
    buffer = ""

    def flush_buffer() -> None:
        nonlocal buffer
        if not buffer.strip():
            buffer = ""
            return
        for piece in _chunk_text(buffer, _SEPARATORS, body_budget, overlap, counter):
            raw_chunks.append((piece, False, _size(piece, counter) > body_budget))
        buffer = ""

    for span in spans:
        if not atomic_blocks_enabled or span.kind == SpanKind.FLOW:
            if buffer:
                buffer += "\n\n" + span.text
            else:
                buffer = span.text
            continue

        flush_buffer()
        if span.kind == SpanKind.CODE_FENCE:
            parts = _split_code_fence(span, body_budget, counter)
        elif span.kind == SpanKind.TABLE:
            parts = _split_table_rows(span, body_budget, counter)
        else:
            parts = [span.text]

        for part in parts:
            overflow = _size(part, counter) > body_budget
            if overflow:
                logger.warning(
                    "atomic block exceeds budget (%d > %d tokens)",
                    _size(part, counter),
                    body_budget,
                )
            raw_chunks.append((part, len(parts) > 1, overflow))

    flush_buffer()

    if not raw_chunks and section.body.strip():
        for piece in _chunk_text(section.body, _SEPARATORS, body_budget, overlap, counter):
            raw_chunks.append((piece, False, _size(piece, counter) > body_budget))

    out: list[MarkdownChunk] = []
    for body, partial, overflow in raw_chunks:
        text = prefix + body if prefix else body
        tokens = _size(text, counter)
        out.append(
            MarkdownChunk(
                text=text.strip(),
                heading_path=list(section.heading_path),
                tokens=tokens,
                partial=partial,
                overflow=overflow,
            )
        )
    return out


def chunk_markdown_sections(
    text: str,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
    *,
    counter: TokenCounter | None = None,
    heading_prefix_enabled: bool = True,
    atomic_blocks_enabled: bool = True,
) -> list[MarkdownChunk]:
    text = text.strip()
    if not text:
        return []
    counter = counter or CharRatioCounter()
    chunks: list[MarkdownChunk] = []
    for section in walk_sections(text):
        chunks.extend(
            _chunk_section_spans(
                section,
                chunk_size,
                chunk_overlap,
                counter,
                atomic_blocks_enabled=atomic_blocks_enabled,
                heading_prefix_enabled=heading_prefix_enabled,
            )
        )
    if not chunks and text:
        fallback = Section(heading_path=[], body=text)
        chunks.extend(
            _chunk_section_spans(
                fallback,
                chunk_size,
                chunk_overlap,
                counter,
                atomic_blocks_enabled=atomic_blocks_enabled,
                heading_prefix_enabled=heading_prefix_enabled,
            )
        )
    return chunks


def chunk_markdown(
    text: str,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
    **kwargs,
) -> list[str]:
    """Split markdown into chunks (legacy API — returns plain strings)."""
    return [
        c.text
        for c in chunk_markdown_sections(
            text, chunk_size, chunk_overlap, **kwargs
        )
    ]


def chunk_code(
    text: str,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
    **kwargs,
) -> list[str]:
    counter = kwargs.get("counter") or CharRatioCounter()
    return _chunk_text(text, _CODE_SEPARATORS, chunk_size, chunk_overlap, counter)
