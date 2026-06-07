from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

from docgraph.ingest.chunker import (
    _budget_units,
    _hard_split,
    _size,
    chunk_code,
)
from docgraph.ingest.tokenizer import CharRatioCounter, TokenCounter
from docgraph.ingest.ts_node_specs import split_node_types

logger = logging.getLogger(__name__)


@lru_cache(maxsize=32)
def _get_parser(language: str) -> Any | None:
    try:
        from tree_sitter_languages import get_parser
    except ImportError:
        return None
    try:
        return get_parser(language)
    except Exception:
        return None


def chunk_code_treesitter(
    text: str,
    language: str,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
    *,
    counter: TokenCounter | None = None,
) -> list[str]:
    counter = counter or CharRatioCounter()
    parser = _get_parser(language)
    if parser is None:
        return chunk_code(text, chunk_size, chunk_overlap, counter=counter)

    node_types = split_node_types(language)
    if not node_types:
        return chunk_code(text, chunk_size, chunk_overlap, counter=counter)

    source = text.encode("utf-8")
    tree = parser.parse(source)
    spans = _collect_declaration_spans(tree.root_node, source, node_types)
    if not spans:
        return chunk_code(text, chunk_size, chunk_overlap, counter=counter)

    return _pack_spans(spans, chunk_size, chunk_overlap, counter)


def _collect_declaration_spans(
    root: Any, source: bytes, node_types: frozenset[str]
) -> list[str]:
    spans: list[str] = []
    preamble: list[str] = []

    def decode(start: int, end: int) -> str:
        return source[start:end].decode("utf-8", errors="replace")

    for child in root.children:
        if child.type in node_types:
            if preamble:
                prefix = "".join(preamble)
                spans.append(prefix + decode(child.start_byte, child.end_byte))
                preamble.clear()
            else:
                spans.append(decode(child.start_byte, child.end_byte))
        else:
            snippet = decode(child.start_byte, child.end_byte)
            if snippet.strip():
                preamble.append(snippet)

    if preamble and not spans:
        return ["".join(preamble)]
    return [s for s in spans if s.strip()]


def _pack_spans(
    spans: list[str],
    chunk_size: int,
    chunk_overlap: int,
    counter: TokenCounter,
) -> list[str]:
    max_units = _budget_units(chunk_size, counter)
    overlap_units = max(0, _budget_units(chunk_overlap, counter))
    if overlap_units >= max_units:
        overlap_units = max_units // 2

    chunks: list[str] = []
    current = ""
    for span in spans:
        span = span.strip()
        if not span:
            continue
        if _size(span, counter) > max_units:
            if current.strip():
                chunks.append(current.strip())
                current = ""
            sub = _split_oversize_declaration(span, max_units, overlap_units, counter)
            chunks.extend(sub)
            continue
        if not current:
            current = span
            continue
        combined = current + "\n\n" + span
        if _size(combined, counter) <= max_units:
            current = combined
        else:
            chunks.append(current.strip())
            current = span
    if current.strip():
        chunks.append(current.strip())
    return [c for c in chunks if c]


def _split_oversize_declaration(
    span: str,
    max_units: int,
    overlap_units: int,
    counter: TokenCounter,
) -> list[str]:
    lines = span.splitlines()
    header = lines[0] if lines else span[:80]
    body_lines = lines[1:] if len(lines) > 1 else []
    if not body_lines:
        return _hard_split(span, max_units, counter)

    header_prefix = header + "\n"
    body_budget = max(1, max_units - _size(header_prefix, counter))
    parts: list[str] = []
    batch: list[str] = []
    cont = f"# (continued from {header.strip()})\n"

    for line in body_lines:
        candidate = header_prefix + "\n".join(batch + [line])
        if batch and _size(candidate, counter) > body_budget:
            parts.append(header_prefix + "\n".join(batch))
            batch = [line]
        else:
            batch.append(line)
    if batch:
        parts.append(header_prefix + "\n".join(batch))

    if len(parts) <= 1:
        return parts or _hard_split(span, max_units, counter)

    out = [parts[0]]
    for part in parts[1:]:
        out.append(cont + part)
    return out
