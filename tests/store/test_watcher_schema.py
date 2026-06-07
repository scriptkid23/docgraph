import sqlite3
from pathlib import Path

import pytest

from docgraph.config import Config
from docgraph.store.sqlite import SQLiteStore


@pytest.fixture
def cfg(tmp_path: Path) -> Config:
    c = Config(data_dir=tmp_path)
    c.ensure_dirs()
    return c


def _columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}


def test_documents_has_new_columns(cfg: Config):
    store = SQLiteStore(cfg)
    store.init_schema()
    with sqlite3.connect(cfg.sqlite_path) as conn:
        cols = _columns(conn, "documents")
    assert "watched_path" in cols
    assert "materialize" in cols
    assert "mtime_ns" in cols


def test_watched_dirs_table_exists(cfg: Config):
    store = SQLiteStore(cfg)
    store.init_schema()
    with sqlite3.connect(cfg.sqlite_path) as conn:
        cols = _columns(conn, "watched_dirs")
    assert {"id", "path", "folder", "tags", "ignore_globs", "created_at"} <= cols


def test_watcher_state_table_exists(cfg: Config):
    store = SQLiteStore(cfg)
    store.init_schema()
    with sqlite3.connect(cfg.sqlite_path) as conn:
        cols = _columns(conn, "watcher_state")
    assert {"key", "value"} <= cols


def test_partial_unique_index_on_watched_path(cfg: Config):
    store = SQLiteStore(cfg)
    store.init_schema()
    with sqlite3.connect(cfg.sqlite_path) as conn:
        idx = conn.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='index' AND name='idx_documents_watched_path'"
        ).fetchone()
    assert idx is not None


def test_schema_migration_idempotent(cfg: Config):
    store = SQLiteStore(cfg)
    store.init_schema()
    store.init_schema()  # second init must not raise


def test_partial_unique_allows_multiple_null_watched_path(cfg: Config):
    store = SQLiteStore(cfg)
    store.init_schema()
    with sqlite3.connect(cfg.sqlite_path) as conn:
        conn.execute(
            "INSERT INTO documents (id, filename, folder, tags, source_type, status, watched_path) "
            "VALUES ('d1', 'a', '', '[]', 'file', 'ready', NULL)"
        )
        conn.execute(
            "INSERT INTO documents (id, filename, folder, tags, source_type, status, watched_path) "
            "VALUES ('d2', 'b', '', '[]', 'file', 'ready', NULL)"
        )
        conn.commit()


def test_partial_unique_rejects_duplicate_non_null_watched_path(cfg: Config):
    store = SQLiteStore(cfg)
    store.init_schema()
    with sqlite3.connect(cfg.sqlite_path) as conn:
        conn.execute(
            "INSERT INTO documents (id, filename, folder, tags, source_type, status, watched_path) "
            "VALUES ('d1', 'a', '', '[]', 'watched', 'ready', '/tmp/x.md')"
        )
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO documents (id, filename, folder, tags, source_type, status, watched_path) "
                "VALUES ('d2', 'b', '', '[]', 'watched', 'ready', '/tmp/x.md')"
            )
        conn.commit()
