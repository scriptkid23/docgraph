from __future__ import annotations

import json

from docgraph.ingest.chunker import chunk_code


def chunk_json(
    text: str,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
    **kwargs,
) -> list[str]:
    text = text.strip()
    if not text:
        return []
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return chunk_code(text, chunk_size, chunk_overlap, **kwargs)

    pieces: list[str] = []
    if isinstance(data, dict):
        for key, value in data.items():
            pieces.append(json.dumps({key: value}, indent=2, ensure_ascii=False))
    elif isinstance(data, list):
        for item in data:
            pieces.append(json.dumps(item, indent=2, ensure_ascii=False))
    else:
        return [text]

    return _pack_pieces(pieces, chunk_size, chunk_overlap, **kwargs)


def chunk_yaml(
    text: str,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
    **kwargs,
) -> list[str]:
    text = text.strip()
    if not text:
        return []
    lines = text.splitlines()
    blocks: list[str] = []
    current: list[str] = []
    for line in lines:
        if line and not line[0].isspace() and ":" in line.split("#", 1)[0]:
            if current:
                blocks.append("\n".join(current))
            current = [line]
        else:
            current.append(line)
    if current:
        blocks.append("\n".join(current))
    if not blocks:
        return chunk_code(text, chunk_size, chunk_overlap, **kwargs)
    return _pack_pieces(blocks, chunk_size, chunk_overlap, **kwargs)


def _pack_pieces(
    pieces: list[str],
    chunk_size: int,
    chunk_overlap: int,
    **kwargs,
) -> list[str]:
    from docgraph.ingest.chunker import _budget_units, _merge_with_overlap, _split_recursive
    from docgraph.ingest.tokenizer import CharRatioCounter

    counter = kwargs.get("counter") or CharRatioCounter()
    max_units = _budget_units(chunk_size, counter)
    overlap_units = _budget_units(chunk_overlap, counter)
    if overlap_units >= max_units:
        overlap_units = max_units // 2

    chunks: list[str] = []
    current = ""
    for piece in pieces:
        piece = piece.strip()
        if not piece:
            continue
        if not current:
            if counter.count(piece) <= max_units:
                current = piece
            else:
                sub = _split_recursive(piece, max_units, ("\n\n", "\n", " ", ""), counter)
                chunks.extend(
                    _merge_with_overlap(sub, max_units, overlap_units, counter)
                )
            continue
        combined = current + "\n\n" + piece
        if counter.count(combined) <= max_units:
            current = combined
        else:
            chunks.append(current.strip())
            current = piece
    if current.strip():
        chunks.append(current.strip())
    return [c for c in chunks if c]
