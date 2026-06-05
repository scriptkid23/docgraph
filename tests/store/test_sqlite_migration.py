from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from docgraph.config import Config
from docgraph.store.sqlite import SQLiteStore


def _fts_table_exists(db_path: Path) -> bool:
    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='chunks_fts'"
        ).fetchone()
    return row is not None


def test_migrate_creates_chunks_fts_table(tmp_path):
    cfg = Config(data_dir=tmp_path)
    cfg.ensure_dirs()
    store = SQLiteStore(cfg)
    store.init_schema()
    assert _fts_table_exists(cfg.sqlite_path)


def test_migrate_idempotent_when_chunks_fts_exists(tmp_path):
    cfg = Config(data_dir=tmp_path)
    cfg.ensure_dirs()
    store = SQLiteStore(cfg)
    store.init_schema()
    # Second invocation must not raise
    store.init_schema()
    assert _fts_table_exists(cfg.sqlite_path)


def test_migrate_from_old_schema_preserves_documents(tmp_path):
    cfg = Config(data_dir=tmp_path)
    cfg.ensure_dirs()
    db = cfg.sqlite_path
    # Simulate old DB without chunks_fts
    with sqlite3.connect(db) as conn:
        conn.executescript(
            """
            CREATE TABLE documents (
                id TEXT PRIMARY KEY, filename TEXT NOT NULL,
                folder TEXT NOT NULL DEFAULT '', tags TEXT NOT NULL DEFAULT '[]',
                status TEXT NOT NULL DEFAULT 'ready', chunk_count INTEGER NOT NULL DEFAULT 0,
                error_message TEXT, original_path TEXT NOT NULL DEFAULT '',
                markdown_path TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            );
            INSERT INTO documents (id, filename) VALUES ('doc_old_001', 'legacy.md');
            """
        )
    store = SQLiteStore(cfg)
    store.init_schema()
    docs = store.list_documents()
    assert len(docs) == 1
    assert docs[0].filename == "legacy.md"
    assert _fts_table_exists(db)
