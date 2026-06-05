from __future__ import annotations

import json
import re
import sqlite3
from typing import Any, Optional

from docgraph.config import Config

# Token characters preserved: alphanumeric, underscore, dot, hyphen.
# Anything else (operators, quotes, punctuation) becomes a space.
_TOKEN_KEEP = re.compile(r"[^\w\s_.-]", flags=re.UNICODE)


def _sanitize_query(text: str) -> str:
    """Convert raw user input to a safe FTS5 MATCH expression.

    Wraps each token in double quotes so FTS5 treats it as a phrase literal.
    This neutralizes operators (AND, OR, NOT, NEAR, *, +, etc.) and special
    chars. Empty / whitespace / fully-stripped input returns "".
    """
    if not text:
        return ""
    cleaned = _TOKEN_KEEP.sub(" ", text)
    tokens = [t for t in cleaned.split() if t]
    if not tokens:
        return ""
    return " ".join(f'"{t}"' for t in tokens)


def _decode_tags(raw: Any) -> list[str]:
    """Tolerant decoder — same logic as ChromaStore._decode_tags."""
    if not raw:
        return []
    if isinstance(raw, list):
        return [str(t) for t in raw]
    s = str(raw)
    if s.startswith("["):
        try:
            v = json.loads(s)
            if isinstance(v, list):
                return [str(t) for t in v]
        except json.JSONDecodeError:
            pass
    return [t for t in s.split(",") if t]


_FNAME_NORMALIZE = re.compile(r"[._-]")


def _normalize_filename(fname: str) -> str:
    """Replace dot / underscore / hyphen with spaces for FTS indexing.

    The FTS5 tokenizer is configured with ``tokenchars '_.-'`` which means
    ``config.md`` would be stored as a single token and never match a query
    for ``config``.  Replacing those chars with spaces before indexing lets
    the tokenizer split ``config.md`` into ``config`` and ``md`` so that a
    filename like *config.md* scores higher for the query "config" than a
    chunk whose *text* happens to contain the word.
    """
    return _FNAME_NORMALIZE.sub(" ", fname)


class FtsStore:
    """SQLite FTS5 sparse index wrapper.

    Mirrors the chunk IDs and folder/tags metadata stored in Chroma so
    that hybrid search can fuse results. Text is duplicated here (FTS5
    stores all column values) — same as it was before, just simpler.
    """

    def __init__(self, cfg: Config) -> None:
        self._path = cfg.sqlite_path

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def count_chunks(self) -> int:
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) AS n FROM chunks_fts").fetchone()
        return int(row["n"])

    def upsert_chunks(self, chunks: list[dict[str, Any]]) -> None:
        """Insert chunks into chunks_fts.

        Plain INSERT (no UPSERT). Caller must delete_by_doc_id before
        reinserting chunks for an existing document.

        Filename is normalized (`._-` → space) so `config.md` tokenizes
        as `config md` — otherwise `tokenchars '_.-'` would keep it as
        one unbreakable token and queries for `config` wouldn't match.
        """
        if not chunks:
            return
        rows = [
            (
                c["chunk_id"],
                c["doc_id"],
                c.get("folder", ""),
                c.get("tags", "[]"),
                c.get("chunk_index", 0),
                c["text"],
                _normalize_filename(c.get("filename", "")),
            )
            for c in chunks
        ]
        with self._connect() as conn:
            conn.executemany(
                """INSERT INTO chunks_fts
                   (chunk_id, doc_id, folder, tags, chunk_index, text, filename)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                rows,
            )

    def delete_by_doc_id(self, doc_id: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM chunks_fts WHERE doc_id = ?", (doc_id,))

    def clear(self) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM chunks_fts")

    def search(
        self,
        query: str,
        top_k: int = 30,
        folder: Optional[str] = None,
        tags: Optional[list[str]] = None,
    ) -> list[dict[str, Any]]:
        match_expr = _sanitize_query(query)
        if not match_expr:
            return []
        # Column order: chunk_id(0), doc_id(1), folder(2), tags(3),
        # chunk_index(4), text(5), filename(6).
        # bm25() weights map to ALL columns including UNINDEXED, so we
        # supply 7 weights: filename gets 2.0 to rank filename matches higher.
        sql = (
            "SELECT chunk_id, doc_id, folder, tags, chunk_index, "
            "       bm25(chunks_fts, 1,1,1,1,1,1,2) AS bm25_score "
            "FROM chunks_fts WHERE chunks_fts MATCH ? "
        )
        params: list = [match_expr]
        if folder:
            sql += "AND folder = ? "
            params.append(folder)
        # SQLite bm25() returns smaller (more-negative) = better.
        # Fetch top_k * 2 because tag post-filter may reduce results.
        sql += "ORDER BY bm25_score LIMIT ?"
        params.append(top_k * 2)
        required_tags = set(tags or ())
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        out: list[dict[str, Any]] = []
        for r in rows:
            chunk_tags = _decode_tags(r["tags"])
            if required_tags and not required_tags.issubset(chunk_tags):
                continue
            out.append(
                {
                    "chunk_id": r["chunk_id"],
                    "doc_id": r["doc_id"],
                    "folder": r["folder"],
                    "tags": chunk_tags,
                    "chunk_index": int(r["chunk_index"]),
                    # Flip sign so "higher score = more relevant" matches
                    # vector cosine convention.
                    "bm25_score": -float(r["bm25_score"]),
                }
            )
        return out[:top_k]
