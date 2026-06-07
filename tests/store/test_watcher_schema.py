import sqlite3
from pathlib import Path

import pytest

from docgraph.config import Config
from docgraph.models import WatchedDirRecord
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


def test_insert_and_list_watched_dirs(cfg: Config):
    store = SQLiteStore(cfg)
    store.init_schema()
    wd = WatchedDirRecord(
        id="wd_a",
        path="/tmp/notes",
        folder="notes",
        tags=["personal"],
        ignore_globs=["draft/*"],
        created_at="2026-06-07T14:00:00Z",
    )
    store.insert_watched_dir(wd)
    dirs = store.list_watched_dirs()
    assert len(dirs) == 1
    assert dirs[0].id == "wd_a"
    assert dirs[0].tags == ["personal"]
    assert dirs[0].ignore_globs == ["draft/*"]


def test_get_watched_dir_by_id(cfg: Config):
    store = SQLiteStore(cfg)
    store.init_schema()
    wd = WatchedDirRecord(id="wd_b", path="/tmp/x", created_at="2026-06-07T14:00:00Z")
    store.insert_watched_dir(wd)
    got = store.get_watched_dir("wd_b")
    assert got is not None and got.path == "/tmp/x"
    assert store.get_watched_dir("wd_missing") is None


def test_delete_watched_dir(cfg: Config):
    store = SQLiteStore(cfg)
    store.init_schema()
    store.insert_watched_dir(WatchedDirRecord(id="wd_c", path="/tmp/y", created_at="2026-06-07T14:00:00Z"))
    deleted = store.delete_watched_dir("wd_c")
    assert deleted is True
    assert store.get_watched_dir("wd_c") is None
    assert store.delete_watched_dir("wd_c") is False  # already gone


def test_watcher_state_get_set(cfg: Config):
    store = SQLiteStore(cfg)
    store.init_schema()
    assert store.get_watcher_state("enabled") is None
    store.set_watcher_state("enabled", "true")
    assert store.get_watcher_state("enabled") == "true"
    store.set_watcher_state("enabled", "false")  # upsert
    assert store.get_watcher_state("enabled") == "false"
