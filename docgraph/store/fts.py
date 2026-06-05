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

# Extra score subtracted from bm25() (which is negative) when a query token
# matches the filename column.  Subtracting makes the value more-negative =
# "better" in SQLite bm25 convention, so flipped bm25_score ends up higher.
_FILENAME_BONUS = 1.0


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

    Uses ``chunks_fts`` (contentless FTS5) for BM25 ranking and a companion
    ``chunks_meta`` regular table to store the UNINDEXED metadata (chunk_id,
    doc_id, folder, tags, chunk_index, filename) and the text needed to issue
    FTS5 row-level delete commands.

    The companion table is keyed on the FTS5 rowid so JOIN is O(1) per row.

    **Filename weighting**: because SQLite FTS5's ``bm25()`` column weights
    only differentiate TF within the same document — and two documents each
    containing the query token once get equal IDF — we apply a post-query
    filename bonus: for each query token that also matches in the ``filename``
    column (via a secondary ``filename:"token"`` MATCH query) we subtract
    ``_FILENAME_BONUS`` from the raw bm25 score (making it more-negative =
    better).  After flipping sign for the caller the bonus becomes positive.
    """

    _INIT_SQL = """
        CREATE TABLE IF NOT EXISTS chunks_meta (
            rowid    INTEGER PRIMARY KEY,
            chunk_id TEXT    NOT NULL,
            doc_id   TEXT    NOT NULL,
            folder   TEXT    NOT NULL DEFAULT '',
            tags     TEXT    NOT NULL DEFAULT '[]',
            chunk_index INTEGER NOT NULL DEFAULT 0,
            filename TEXT    NOT NULL DEFAULT '',
            text     TEXT    NOT NULL DEFAULT ''
        );
        CREATE INDEX IF NOT EXISTS idx_chunks_meta_doc_id ON chunks_meta(doc_id);
    """

    def __init__(self, cfg: Config) -> None:
        self._path = cfg.sqlite_path
        self._ensure_meta_table()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def _ensure_meta_table(self) -> None:
        """Create chunks_meta and its index if they don't exist yet."""
        with self._connect() as conn:
            conn.executescript(self._INIT_SQL)

    def count_chunks(self) -> int:
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) AS n FROM chunks_meta").fetchone()
        return int(row["n"])

    def upsert_chunks(self, chunks: list[dict[str, Any]]) -> None:
        """Insert chunks into both FTS5 and the companion metadata table.

        This is a plain INSERT (no UPSERT) — contentless FTS5 has no UPSERT
        semantics. Callers must call ``delete_by_doc_id`` before reinserting
        chunks for an existing document.

        The filename is normalized (``._-`` replaced with spaces) in the FTS5
        row so that ``config.md`` tokenizes as ``config md``.  The original
        filename is preserved in ``chunks_meta`` for display purposes.
        """
        if not chunks:
            return
        with self._connect() as conn:
            for c in chunks:
                chunk_id = c["chunk_id"]
                doc_id = c["doc_id"]
                folder = c.get("folder", "")
                tags = c.get("tags", "[]")
                chunk_index = c.get("chunk_index", 0)
                text = c["text"]
                filename = c.get("filename", "")
                fts_filename = _normalize_filename(filename)
                cur = conn.execute(
                    """INSERT INTO chunks_fts
                       (chunk_id, doc_id, folder, tags, chunk_index, text, filename)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (chunk_id, doc_id, folder, tags, chunk_index, text, fts_filename),
                )
                rowid = cur.lastrowid
                conn.execute(
                    """INSERT INTO chunks_meta
                       (rowid, chunk_id, doc_id, folder, tags, chunk_index, filename, text)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    # Store original filename (not normalized) in meta for display.
                    # Store fts_filename in meta.text for delete command accuracy.
                    (rowid, chunk_id, doc_id, folder, tags, chunk_index, filename, text),
                )

    def delete_by_doc_id(self, doc_id: str) -> None:
        """Remove all FTS5 rows for a given doc_id.

        Contentless FTS5 requires issuing the special 'delete' command with
        the original column values for each row — we read those from
        ``chunks_meta`` and then remove from both tables.
        """
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT rowid, chunk_id, doc_id, folder, tags, chunk_index, filename, text "
                "FROM chunks_meta WHERE doc_id = ?",
                (doc_id,),
            ).fetchall()
            for r in rows:
                # The FTS5 delete command requires the exact values that were
                # inserted.  filename was normalized on insert, so re-normalize here.
                fts_filename = _normalize_filename(r["filename"])
                conn.execute(
                    "INSERT INTO chunks_fts"
                    "(chunks_fts, rowid, chunk_id, doc_id, folder, tags, chunk_index, text, filename)"
                    " VALUES ('delete', ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        r["rowid"], r["chunk_id"], r["doc_id"],
                        r["folder"], r["tags"], r["chunk_index"],
                        r["text"], fts_filename,
                    ),
                )
            conn.execute("DELETE FROM chunks_meta WHERE doc_id = ?", (doc_id,))

    def clear(self) -> None:
        """Remove all rows from both FTS5 and the companion metadata table."""
        with self._connect() as conn:
            conn.execute("INSERT INTO chunks_fts(chunks_fts) VALUES ('delete-all')")
            conn.execute("DELETE FROM chunks_meta")

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

        # Build column-scoped filename match expression from the same tokens.
        # Example: '"config"' → 'filename:"config"'
        tokens = [t for t in match_expr.split() if t]
        fname_expr = " OR ".join(f"filename:{t}" for t in tokens)

        # Primary query: BM25 over all indexed columns, JOIN meta for metadata.
        sql = (
            "SELECT chunks_fts.rowid, m.chunk_id, m.doc_id, m.folder, m.tags, "
            "       m.chunk_index, bm25(chunks_fts, 1.0, 2.0) AS bm25_score "
            "FROM chunks_fts "
            "JOIN chunks_meta m ON m.rowid = chunks_fts.rowid "
            "WHERE chunks_fts MATCH ? "
        )
        params: list = [match_expr]
        if folder:
            sql += "AND m.folder = ? "
            params.append(folder)
        # SQLite bm25() returns smaller=better (often negative values);
        # ORDER BY ASC picks the best match first.  Fetch top_k * 2 because
        # tag post-filtering may reduce the result set.
        sql += "ORDER BY bm25_score LIMIT ?"
        params.append(top_k * 2)

        required_tags = set(tags or ())
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()

            # Secondary query: find rowids where query tokens hit the filename
            # column specifically, so we can apply a bonus.
            fname_rowids: set[int] = set()
            if fname_expr and rows:
                try:
                    fname_rows = conn.execute(
                        "SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH ?",
                        (fname_expr,),
                    ).fetchall()
                    fname_rowids = {r["rowid"] for r in fname_rows}
                except sqlite3.OperationalError:
                    pass  # malformed expr — skip bonus

        scored: list[tuple[float, dict[str, Any]]] = []
        for r in rows:
            chunk_tags = _decode_tags(r["tags"])
            if required_tags and not required_tags.issubset(chunk_tags):
                continue
            raw_bm25 = float(r["bm25_score"])
            # Apply filename bonus: subtract from already-negative bm25 score
            # so the result sorts first (more-negative = better in SQLite bm25).
            if r["rowid"] in fname_rowids:
                raw_bm25 -= _FILENAME_BONUS
            scored.append(
                (
                    raw_bm25,
                    {
                        "chunk_id": r["chunk_id"],
                        "doc_id": r["doc_id"],
                        "folder": r["folder"],
                        "tags": chunk_tags,
                        "chunk_index": int(r["chunk_index"]),
                        # Flip sign so "higher score = more relevant" matches
                        # vector convention (cosine similarity: higher = better).
                        "bm25_score": -raw_bm25,
                    },
                )
            )
        scored.sort(key=lambda x: x[0])
        return [item for _, item in scored[:top_k]]
