from __future__ import annotations

import argparse

from docgraph.config import Config
from docgraph.models import DocumentRecord
from docgraph.store import ChromaStore, FtsStore, SQLiteStore


def _seed_chroma(cfg, n):
    sqlite = SQLiteStore(cfg)
    sqlite.init_schema()
    chroma = ChromaStore(cfg)
    for i in range(n):
        sqlite.insert_document(DocumentRecord(
            id=f"doc_{i}", filename=f"f{i}.md", folder="x", tags=["t"]
        ))
        chroma.upsert_chunks([{
            "id": f"doc_{i}_0", "embedding": [0.1] * 768, "text": f"text {i}",
            "metadata": {"doc_id": f"doc_{i}", "filename": f"f{i}.md",
                         "folder": "x", "tags": '["t"]', "chunk_index": 0},
        }])


def test_rebuild_fts_populates_index(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("DOCGRAPH_DATA_DIR", str(tmp_path))
    cfg = Config(data_dir=tmp_path)
    cfg.ensure_dirs()
    _seed_chroma(cfg, n=3)
    from docgraph.cli import cmd_rebuild_fts
    rc = cmd_rebuild_fts(argparse.Namespace())
    assert rc == 0
    fts = FtsStore(cfg)
    assert fts.count_chunks() == 3
    out = capsys.readouterr().out
    assert "Chroma has 3 chunks" in out
    assert "Done" in out


def test_rebuild_fts_empty_chroma(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("DOCGRAPH_DATA_DIR", str(tmp_path))
    cfg = Config(data_dir=tmp_path)
    cfg.ensure_dirs()
    sqlite = SQLiteStore(cfg)
    sqlite.init_schema()
    ChromaStore(cfg)
    from docgraph.cli import cmd_rebuild_fts
    rc = cmd_rebuild_fts(argparse.Namespace())
    assert rc == 0
    out = capsys.readouterr().out
    assert "Nothing to rebuild" in out


def test_rebuild_fts_exits_nonzero_on_exception(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("DOCGRAPH_DATA_DIR", str(tmp_path))
    cfg = Config(data_dir=tmp_path)
    cfg.ensure_dirs()
    _seed_chroma(cfg, n=2)

    # Monkeypatch FtsStore.rebuild_from_chroma to raise
    from docgraph.store.fts import FtsStore as _FtsStore
    async def boom(self, chroma, sqlite, batch_size=1000, progress_callback=None):
        raise RuntimeError("simulated rebuild failure")
    monkeypatch.setattr(_FtsStore, "rebuild_from_chroma", boom)

    from docgraph.cli import cmd_rebuild_fts
    rc = cmd_rebuild_fts(argparse.Namespace())
    assert rc == 1
    err = capsys.readouterr().err
    assert "simulated rebuild failure" in err or "failed" in err.lower()


def test_rebuild_fts_exits_two_on_silent_partial(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("DOCGRAPH_DATA_DIR", str(tmp_path))
    cfg = Config(data_dir=tmp_path)
    cfg.ensure_dirs()
    _seed_chroma(cfg, n=5)

    # Patch rebuild to report only 3 of 5 indexed (silent partial).
    from docgraph.store.fts import FtsStore as _FtsStore
    async def partial(self, chroma, sqlite, batch_size=1000, progress_callback=None):
        # Don't actually populate — just return wrong count.
        return 3
    monkeypatch.setattr(_FtsStore, "rebuild_from_chroma", partial)

    from docgraph.cli import cmd_rebuild_fts
    rc = cmd_rebuild_fts(argparse.Namespace())
    assert rc == 2
    captured = capsys.readouterr()
    assert "Partial rebuild" in captured.err, (
        f"Partial rebuild message should be on stderr, got stdout={captured.out!r} "
        f"stderr={captured.err!r}"
    )


def test_rebuild_fts_idempotent(tmp_path, monkeypatch, capsys):
    """Running rebuild twice in a row leaves count = N, not 2N
    (rebuild_from_chroma clears the FTS table first)."""
    monkeypatch.setenv("DOCGRAPH_DATA_DIR", str(tmp_path))
    cfg = Config(data_dir=tmp_path)
    cfg.ensure_dirs()
    _seed_chroma(cfg, n=3)
    from docgraph.cli import cmd_rebuild_fts

    rc1 = cmd_rebuild_fts(argparse.Namespace())
    assert rc1 == 0
    rc2 = cmd_rebuild_fts(argparse.Namespace())
    assert rc2 == 0

    fts = FtsStore(cfg)
    assert fts.count_chunks() == 3, (
        "rebuild was not idempotent — count doubled to "
        f"{fts.count_chunks()} after second run"
    )
