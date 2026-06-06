from __future__ import annotations

import pytest

from docgraph.config import Config
from docgraph.store.fts import FtsStore
from docgraph.store.sqlite import SQLiteStore


@pytest.fixture
def fts(tmp_path):
    cfg = Config(data_dir=tmp_path)
    cfg.ensure_dirs()
    sqlite = SQLiteStore(cfg)
    sqlite.init_schema()
    f = FtsStore(cfg)
    f.upsert_chunks([
        {"chunk_id": "doc_a_0", "doc_id": "doc_a", "folder": "x", "tags": "[]",
         "chunk_index": 0, "text": "normal content here", "filename": "a.md"},
        {"chunk_id": "doc_b_0", "doc_id": "doc_b", "folder": "y", "tags": "[]",
         "chunk_index": 0, "text": "more normal content", "filename": "b.md"},
    ])
    return f


@pytest.mark.parametrize("evil", [
    "'; DROP TABLE chunks_fts; --",
    "' OR '1'='1",
    "'); DELETE FROM documents; --",
    "\\'; SELECT * FROM sqlite_master; --",
    "'; ATTACH DATABASE '/tmp/x.db' AS evil; --",
    "x' UNION SELECT * FROM sqlite_master; --",
])
def test_sql_injection_in_query_does_not_corrupt_data(fts, evil):
    """Search query input must not allow SQL injection.

    Task 5's FTS5 sanitizer escapes quotes and Task 6's CRUD layer uses
    parameter binding. The combination should prevent both classic SQL
    injection AND FTS5 syntax injection.
    """
    hits = fts.search(evil, top_k=10)
    assert isinstance(hits, list)
    assert fts.count_chunks() == 2, "data was corrupted by injection"


@pytest.mark.parametrize("syntactic", [
    "*", "(", ")", '"', ":", "^", "+", "-",
    "AND", "OR", "NOT", "NEAR", "MATCH",
    '"unclosed', '"foo" AND OR',
    "a" * 10000,
    "\x00\x01\x02",
])
def test_fts5_special_chars_do_not_crash(fts, syntactic):
    """FTS5-reserved syntax and control chars must not raise.

    The sanitizer (Task 5) wraps the query in a quoted form so reserved
    FTS5 operators are treated as literal text.
    """
    hits = fts.search(syntactic, top_k=10)
    assert isinstance(hits, list)


def test_folder_filter_uses_parameter_binding(fts):
    """folder filter must be parameter-bound, not string-interpolated."""
    evil = "x'; DROP TABLE chunks_fts; --"
    hits = fts.search("normal", top_k=10, folder=evil)
    assert hits == []
    assert fts.count_chunks() == 2
