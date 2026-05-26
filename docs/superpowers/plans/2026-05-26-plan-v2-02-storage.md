# Plan v2-02 — Storage Layer

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement SQLite metadata store, ChromaDB vector store, and local file store for originals and converted markdown.

**Architecture:** Three focused modules under `boostmcp/store/`. SQLite owns document records and chunk metadata. ChromaDB owns vectors for similarity search. FileStore saves uploaded originals and converted markdown on disk.

**Tech Stack:** Python sqlite3, chromadb, pathlib

**Depends on:** Plan v2-01  
**Blocks:** Plans v2-04, v2-05, v2-06

**Spec refs:** §5.2 Runtime Data Directory, §5.5 Chunk Metadata

---

## File Structure (after this plan)

```
boostmcp/store/
├── __init__.py
├── files.py
├── sqlite.py
└── chroma.py
tests/store/
├── test_files.py
├── test_sqlite.py
└── test_chroma.py
```

Add to `pyproject.toml` dependencies: `chromadb = "^0.6"`

---

### Task 1: File store

**Files:**
- Create: `boostmcp/store/files.py`
- Create: `tests/store/test_files.py`

- [ ] **Step 1: Write failing test**

```python
# tests/store/test_files.py
from pathlib import Path

from boostmcp.config import Config
from boostmcp.store.files import FileStore, sanitize_filename


def test_sanitize_filename():
    assert sanitize_filename("../../etc/passwd") == "passwd"
    assert sanitize_filename("my doc (1).pdf") == "my_doc__1_.pdf"


def test_save_original_and_markdown(tmp_data_dir):
    cfg = Config(data_dir=tmp_data_dir)
    cfg.ensure_dirs()
    store = FileStore(cfg)
    orig = store.save_original("doc_1", "report.pdf", b"PDF bytes")
    md = store.save_markdown("doc_1", "# Title\n\nBody")
    assert orig.exists()
    assert md.exists()
    assert store.read_markdown("doc_1") == "# Title\n\nBody"
    store.delete_doc_files("doc_1")
    assert not orig.exists()
    assert not md.exists()
```

- [ ] **Step 2: Run test — expect FAIL**

```bash
poetry run pytest tests/store/test_files.py -v
```

- [ ] **Step 3: Implement**

```python
# boostmcp/store/files.py
from __future__ import annotations

import re
from pathlib import Path

from boostmcp.config import Config


def sanitize_filename(name: str) -> str:
    base = Path(name).name
    base = re.sub(r"[^\w.\- ]", "_", base)
    base = base.replace(" ", "_")
    return base or "upload"


class FileStore:
    def __init__(self, cfg: Config) -> None:
        self._cfg = cfg

    def save_original(self, doc_id: str, filename: str, content: bytes) -> Path:
        safe = sanitize_filename(filename)
        path = self._cfg.originals_dir / f"{doc_id}_{safe}"
        path.write_bytes(content)
        return path

    def save_markdown(self, doc_id: str, markdown: str) -> Path:
        path = self._cfg.markdown_dir / f"{doc_id}.md"
        path.write_text(markdown, encoding="utf-8")
        return path

    def read_markdown(self, doc_id: str) -> str:
        path = self._cfg.markdown_dir / f"{doc_id}.md"
        return path.read_text(encoding="utf-8")

    def delete_doc_files(self, doc_id: str) -> None:
        for p in self._cfg.originals_dir.glob(f"{doc_id}_*"):
            p.unlink(missing_ok=True)
        md = self._cfg.markdown_dir / f"{doc_id}.md"
        md.unlink(missing_ok=True)
```

- [ ] **Step 4: Run test — expect PASS**

- [ ] **Step 5: Commit**

```bash
git add boostmcp/store/files.py tests/store/test_files.py
git commit -m "feat: add file store for originals and markdown"
```

---

### Task 2: SQLite metadata store

**Files:**
- Create: `boostmcp/store/sqlite.py`
- Create: `tests/store/test_sqlite.py`

- [ ] **Step 1: Write failing test**

```python
# tests/store/test_sqlite.py
from boostmcp.config import Config
from boostmcp.models import DocumentRecord, DocumentStatus
from boostmcp.store.sqlite import SQLiteStore


def test_insert_and_get_document(tmp_data_dir):
    cfg = Config(data_dir=tmp_data_dir)
    cfg.ensure_dirs()
    db = SQLiteStore(cfg)
    db.init_schema()
    doc = DocumentRecord(
        id="doc_1",
        filename="a.md",
        folder="proj",
        tags=["tag1"],
        original_path="/tmp/a.md",
    )
    db.insert_document(doc)
    got = db.get_document("doc_1")
    assert got is not None
    assert got.filename == "a.md"
    assert got.folder == "proj"
    assert got.tags == ["tag1"]


def test_list_documents_filter_by_folder(tmp_data_dir):
    cfg = Config(data_dir=tmp_data_dir)
    cfg.ensure_dirs()
    db = SQLiteStore(cfg)
    db.init_schema()
    db.insert_document(DocumentRecord(id="d1", filename="a.md", folder="A"))
    db.insert_document(DocumentRecord(id="d2", filename="b.md", folder="B"))
    rows = db.list_documents(folder="A")
    assert len(rows) == 1
    assert rows[0].id == "d1"


def test_update_status(tmp_data_dir):
    cfg = Config(data_dir=tmp_data_dir)
    cfg.ensure_dirs()
    db = SQLiteStore(cfg)
    db.init_schema()
    db.insert_document(DocumentRecord(id="d1", filename="a.md"))
    db.update_status("d1", DocumentStatus.READY, chunk_count=5)
    got = db.get_document("d1")
    assert got.status == DocumentStatus.READY
    assert got.chunk_count == 5
```

- [ ] **Step 2: Run test — expect FAIL**

- [ ] **Step 3: Implement**

```python
# boostmcp/store/sqlite.py
from __future__ import annotations

import json
import sqlite3
from typing import Optional

from boostmcp.config import Config
from boostmcp.models import DocumentRecord, DocumentStatus


class SQLiteStore:
    def __init__(self, cfg: Config) -> None:
        self._path = cfg.sqlite_path

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._path)
        conn.row_factory = sqlite3.Row
        return conn

    def init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    folder TEXT NOT NULL DEFAULT '',
                    tags TEXT NOT NULL DEFAULT '[]',
                    status TEXT NOT NULL DEFAULT 'processing',
                    chunk_count INTEGER NOT NULL DEFAULT 0,
                    error_message TEXT,
                    original_path TEXT NOT NULL DEFAULT '',
                    markdown_path TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL DEFAULT (datetime('now'))
                );
            """)

    def insert_document(self, doc: DocumentRecord) -> None:
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO documents
                   (id, filename, folder, tags, status, chunk_count,
                    error_message, original_path, markdown_path)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    doc.id, doc.filename, doc.folder, json.dumps(doc.tags),
                    doc.status.value, doc.chunk_count, doc.error_message,
                    doc.original_path, doc.markdown_path,
                ),
            )

    def _row_to_doc(self, row: sqlite3.Row) -> DocumentRecord:
        return DocumentRecord(
            id=row["id"],
            filename=row["filename"],
            folder=row["folder"],
            tags=json.loads(row["tags"]),
            status=DocumentStatus(row["status"]),
            chunk_count=row["chunk_count"],
            error_message=row["error_message"],
            original_path=row["original_path"],
            markdown_path=row["markdown_path"],
        )

    def get_document(self, doc_id: str) -> Optional[DocumentRecord]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM documents WHERE id = ?", (doc_id,)
            ).fetchone()
        return self._row_to_doc(row) if row else None

    def list_documents(
        self,
        folder: Optional[str] = None,
        tag: Optional[str] = None,
        status: Optional[DocumentStatus] = None,
    ) -> list[DocumentRecord]:
        query = "SELECT * FROM documents WHERE 1=1"
        params: list = []
        if folder is not None:
            query += " AND folder = ?"
            params.append(folder)
        if status is not None:
            query += " AND status = ?"
            params.append(status.value)
        query += " ORDER BY created_at DESC"
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        docs = [self._row_to_doc(r) for r in rows]
        if tag:
            docs = [d for d in docs if tag in d.tags]
        return docs

    def update_status(
        self,
        doc_id: str,
        status: DocumentStatus,
        chunk_count: int = 0,
        error_message: Optional[str] = None,
        markdown_path: str = "",
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """UPDATE documents SET status=?, chunk_count=?,
                   error_message=?, markdown_path=? WHERE id=?""",
                (status.value, chunk_count, error_message, markdown_path, doc_id),
            )

    def update_tags_folder(self, doc_id: str, tags: list[str], folder: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE documents SET tags=?, folder=? WHERE id=?",
                (json.dumps(tags), folder, doc_id),
            )

    def delete_document(self, doc_id: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
```

- [ ] **Step 4: Run tests — expect PASS**

- [ ] **Step 5: Commit**

```bash
git add boostmcp/store/sqlite.py tests/store/test_sqlite.py
git commit -m "feat: add SQLite metadata store for documents"
```

---

### Task 3: ChromaDB vector store

**Files:**
- Create: `boostmcp/store/chroma.py`
- Create: `tests/store/test_chroma.py`

- [ ] **Step 1: Add chromadb dependency**

```bash
poetry add chromadb
```

- [ ] **Step 2: Write failing test**

```python
# tests/store/test_chroma.py
from boostmcp.config import Config
from boostmcp.store.chroma import ChromaStore


def test_upsert_and_search(tmp_data_dir):
    cfg = Config(data_dir=tmp_data_dir)
    cfg.ensure_dirs()
    store = ChromaStore(cfg)
    vec = [0.1] * 768
    store.upsert_chunks([
        {
            "id": "doc_1_0",
            "embedding": vec,
            "text": "Ollama embedding config",
            "metadata": {
                "doc_id": "doc_1",
                "filename": "readme.md",
                "folder": "BoostMCP",
                "tags": "design,v2",
                "chunk_index": 0,
            },
        }
    ])
    results = store.search(
        query_embedding=vec,
        top_k=1,
        folder="BoostMCP",
    )
    assert len(results) == 1
    assert results[0]["text"] == "Ollama embedding config"
    store.delete_by_doc_id("doc_1")
    assert store.search(query_embedding=vec, top_k=1) == []
```

- [ ] **Step 3: Implement**

```python
# boostmcp/store/chroma.py
from __future__ import annotations

from typing import Any, Optional

import chromadb

from boostmcp.config import Config


COLLECTION_NAME = "boostmcp_chunks"


class ChromaStore:
    def __init__(self, cfg: Config) -> None:
        self._client = chromadb.PersistentClient(path=str(cfg.chroma_path))
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    def upsert_chunks(self, chunks: list[dict[str, Any]]) -> None:
        if not chunks:
            return
        self._collection.upsert(
            ids=[c["id"] for c in chunks],
            embeddings=[c["embedding"] for c in chunks],
            documents=[c["text"] for c in chunks],
            metadatas=[c["metadata"] for c in chunks],
        )

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        folder: Optional[str] = None,
        tag: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        where: dict[str, Any] = {}
        if folder:
            where["folder"] = folder
        kwargs: dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": top_k,
        }
        if where:
            kwargs["where"] = where
        result = self._collection.query(**kwargs)
        out: list[dict[str, Any]] = []
        if not result["ids"] or not result["ids"][0]:
            return out
        for i, chunk_id in enumerate(result["ids"][0]):
            meta = result["metadatas"][0][i]
            tags_str = meta.get("tags", "")
            tags = [t for t in tags_str.split(",") if t]
            if tag and tag not in tags:
                continue
            dist = result["distances"][0][i] if result.get("distances") else 0.0
            score = 1.0 - dist  # cosine distance → similarity
            out.append({
                "id": chunk_id,
                "text": result["documents"][0][i],
                "doc_id": meta.get("doc_id", ""),
                "filename": meta.get("filename", ""),
                "folder": meta.get("folder", ""),
                "tags": tags,
                "chunk_index": int(meta.get("chunk_index", 0)),
                "score": score,
                "source_page": meta.get("source_page"),
            })
        return out

    def delete_by_doc_id(self, doc_id: str) -> None:
        existing = self._collection.get(where={"doc_id": doc_id})
        if existing["ids"]:
            self._collection.delete(ids=existing["ids"])
```

- [ ] **Step 4: Run tests — expect PASS**

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml poetry.lock boostmcp/store/chroma.py tests/store/test_chroma.py
git commit -m "feat: add ChromaDB vector store for chunk search"
```

---

### Task 4: Store package init and integration smoke test

**Files:**
- Create: `boostmcp/store/__init__.py`

- [ ] **Step 1: Export public API**

```python
# boostmcp/store/__init__.py
from boostmcp.store.chroma import ChromaStore
from boostmcp.store.files import FileStore
from boostmcp.store.sqlite import SQLiteStore

__all__ = ["ChromaStore", "FileStore", "SQLiteStore"]
```

- [ ] **Step 2: Run all store tests**

```bash
poetry run pytest tests/store/ -v
```

Expected: all PASS

- [ ] **Step 3: Commit**

```bash
git add boostmcp/store/__init__.py
git commit -m "chore: export storage layer public API"
```
