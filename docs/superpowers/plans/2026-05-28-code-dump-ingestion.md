# Code-Dump Ingestion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ingest a Repomix (plain/XML) codebase dump as one uploaded file — split it back into per-file sections, chunk each file with code-aware heuristics, and tag every chunk with its in-repo `file_path` + `language` so search hits are attributable.

**Architecture:** A new `code_dump` module detects and parses Repomix dumps. The chunker gains a code-aware variant sharing the existing recursive splitter. The indexer reads a bounded prefix of each uploaded file; if it looks like a Repomix dump it routes to a new `index_code_dump` path, otherwise the existing convert→markdown path runs unchanged. `file_path` metadata is threaded through Chroma → search → the MCP response.

**Tech Stack:** Python 3.13, pytest (asyncio + respx), ChromaDB, dataclasses.

Spec: `docs/superpowers/specs/2026-05-28-code-dump-ingestion-design.md`

---

## File Structure

- **Create** `docgraph/ingest/code_dump.py` — `infer_language`, `detect_repomix`, `parse_repomix`.
- **Modify** `docgraph/ingest/chunker.py` — parametrize `_split_recursive`, add `_CODE_SEPARATORS`, `chunk_code`, shared `_chunk`.
- **Modify** `docgraph/models.py` — optional `file_path` on `ChunkRecord` + `SearchResult`.
- **Modify** `docgraph/store/chroma.py` — surface `file_path` from chunk metadata.
- **Modify** `docgraph/mcp/search.py` — map `file_path` onto `SearchResult`.
- **Modify** `docgraph/mcp/server.py` — include `file_path` in the search payload.
- **Modify** `docgraph/ingest/indexer.py` — `_read_text_prefix`, `index_code_dump`, routing in `index_document`.
- **Create** `tests/ingest/test_code_dump.py`; **extend** `tests/ingest/test_chunker.py`, `tests/test_models.py`, `tests/store/test_chroma.py`, `tests/ingest/test_indexer.py`.

---

## Task 1: Language inference

**Files:**
- Create: `docgraph/ingest/code_dump.py`
- Test: `tests/ingest/test_code_dump.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/ingest/test_code_dump.py
from docgraph.ingest.code_dump import infer_language


def test_infer_language_known_extensions():
    assert infer_language("src/main.py") == "python"
    assert infer_language("app/index.ts") == "typescript"
    assert infer_language("pkg/server.go") == "go"


def test_infer_language_unknown_returns_none():
    assert infer_language("data.bin") is None
    assert infer_language("Makefile") is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/ingest/test_code_dump.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'docgraph.ingest.code_dump'`

- [ ] **Step 3: Write minimal implementation**

```python
# docgraph/ingest/code_dump.py
from __future__ import annotations

import re

_LANGUAGE_BY_EXT = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".rb": "ruby",
    ".php": "php",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".hpp": "cpp",
    ".cs": "csharp",
    ".sh": "shell",
    ".sql": "sql",
    ".md": "markdown",
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".html": "html",
    ".css": "css",
}


def infer_language(path: str) -> str | None:
    """Map a file path's extension to a language name, or None if unknown."""
    dot = path.rfind(".")
    if dot == -1:
        return None
    return _LANGUAGE_BY_EXT.get(path[dot:].lower())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/ingest/test_code_dump.py -q`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add docgraph/ingest/code_dump.py tests/ingest/test_code_dump.py
git commit -m "feat: add language inference for code-dump ingestion"
```

---

## Task 2: Detect Repomix dumps

**Files:**
- Modify: `docgraph/ingest/code_dump.py`
- Test: `tests/ingest/test_code_dump.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/ingest/test_code_dump.py  (append)
from docgraph.ingest.code_dump import detect_repomix


def test_detect_repomix_plain():
    text = "================\nFile: src/a.py\n================\nprint(1)\n"
    assert detect_repomix(text) is True


def test_detect_repomix_xml():
    text = '<file path="src/a.py">\nprint(1)\n</file>'
    assert detect_repomix(text) is True


def test_detect_repomix_false_on_prose():
    text = "# My Book\n\nChapter one. Some ordinary prose here.\n"
    assert detect_repomix(text) is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/ingest/test_code_dump.py -q`
Expected: FAIL — `ImportError: cannot import name 'detect_repomix'`

- [ ] **Step 3: Write minimal implementation**

Append to `docgraph/ingest/code_dump.py` (after the imports, before/after `infer_language`):

```python
# Repomix plain-style per-file header:
#   ================
#   File: path/to/file.ext
#   ================
# The '=' rule length varies by Repomix version, so match 3 or more.
_PLAIN_HEADER = re.compile(
    r"^={3,}[ \t]*\nFile: (?P<path>.+?)[ \t]*\n={3,}[ \t]*$",
    re.MULTILINE,
)

# Repomix XML-style block: <file path="...">content</file>.
# Repomix does NOT escape file content, so scan with regex, not an XML parser.
_XML_FILE = re.compile(
    r'<file path="(?P<path>[^"]+)">\n?(?P<body>.*?)\n?</file>',
    re.DOTALL,
)


def detect_repomix(text: str) -> bool:
    """True if text looks like a Repomix dump (plain or XML style)."""
    return bool(_XML_FILE.search(text) or _PLAIN_HEADER.search(text))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/ingest/test_code_dump.py -q`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add docgraph/ingest/code_dump.py tests/ingest/test_code_dump.py
git commit -m "feat: detect Repomix plain/XML dumps"
```

---

## Task 3: Parse Repomix dumps into (path, content) pairs

**Files:**
- Modify: `docgraph/ingest/code_dump.py`
- Test: `tests/ingest/test_code_dump.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/ingest/test_code_dump.py  (append)
from docgraph.ingest.code_dump import parse_repomix


def test_parse_repomix_plain():
    text = (
        "================\nFile: src/a.py\n================\n"
        "def a():\n    return 1\n\n"
        "================\nFile: src/b.py\n================\n"
        "def b():\n    return 2\n"
    )
    files = parse_repomix(text)
    assert [p for p, _ in files] == ["src/a.py", "src/b.py"]
    assert "def a():" in files[0][1]
    assert "def b():" in files[1][1]


def test_parse_repomix_xml():
    text = (
        '<file path="src/a.py">\ndef a():\n    return 1\n</file>\n'
        '<file path="src/b.ts">\nexport const b = 2\n</file>\n'
    )
    files = parse_repomix(text)
    assert [p for p, _ in files] == ["src/a.py", "src/b.ts"]
    assert "def a():" in files[0][1]


def test_parse_repomix_skips_empty_sections():
    text = (
        "================\nFile: empty.py\n================\n\n"
        "================\nFile: real.py\n================\nx = 1\n"
    )
    files = parse_repomix(text)
    assert [p for p, _ in files] == ["real.py"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/ingest/test_code_dump.py -q`
Expected: FAIL — `ImportError: cannot import name 'parse_repomix'`

- [ ] **Step 3: Write minimal implementation**

Append to `docgraph/ingest/code_dump.py`:

```python
def parse_repomix(text: str) -> list[tuple[str, str]]:
    """Split a Repomix dump into (file_path, content) pairs.

    Tries XML style first, then plain style. Empty file sections are skipped.
    """
    xml_matches = list(_XML_FILE.finditer(text))
    if xml_matches:
        out: list[tuple[str, str]] = []
        for m in xml_matches:
            body = m.group("body").strip("\n")
            if body.strip():
                out.append((m.group("path").strip(), body))
        return out

    headers = list(_PLAIN_HEADER.finditer(text))
    out = []
    for i, m in enumerate(headers):
        start = m.end()
        end = headers[i + 1].start() if i + 1 < len(headers) else len(text)
        body = text[start:end].strip("\n")
        if body.strip():
            out.append((m.group("path").strip(), body))
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/ingest/test_code_dump.py -q`
Expected: PASS (8 passed)

- [ ] **Step 5: Commit**

```bash
git add docgraph/ingest/code_dump.py tests/ingest/test_code_dump.py
git commit -m "feat: parse Repomix dumps into per-file sections"
```

---

## Task 4: Code-aware chunking

**Files:**
- Modify: `docgraph/ingest/chunker.py`
- Test: `tests/ingest/test_chunker.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/ingest/test_chunker.py  (append; add chunk_code to the import line)
from docgraph.ingest.chunker import chunk_markdown, chunk_code


def test_chunk_code_keeps_small_function_intact():
    code = "def foo():\n    return 1\n"
    chunks = chunk_code(code, chunk_size=512, chunk_overlap=64)
    assert chunks == ["def foo():\n    return 1"]


def test_chunk_code_splits_large_file_without_error():
    code = "def big():\n" + "    x = 1\n" * 20_000
    chunks = chunk_code(code, chunk_size=128, chunk_overlap=16)
    assert len(chunks) > 1
    assert all(c.strip() for c in chunks)


def test_chunk_code_prefers_def_boundaries():
    # Each function fits the budget alone (char_size = 64*4 = 256) but the two
    # together exceed it, so the split lands on the def boundary.
    def fn(n: int) -> str:
        return f"def f{n}():\n" + "    pass\n" * 16

    code = fn(1) + fn(2)
    chunks = chunk_code(code, chunk_size=64, chunk_overlap=0)
    assert len(chunks) == 2
    assert chunks[0].startswith("def f1")
    assert chunks[1].startswith("def f2")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/ingest/test_chunker.py -q`
Expected: FAIL — `ImportError: cannot import name 'chunk_code'`

- [ ] **Step 3: Write the implementation**

In `docgraph/ingest/chunker.py`, add the code separators after `_SEPARATORS`:

```python
_CODE_SEPARATORS: tuple[str, ...] = (
    "\nclass ",
    "\ndef ",
    "\nfunc ",
    "\nfunction ",
    "\n\n",
    "\n",
    " ",
    "",
)
```

Change `_split_recursive` to take the separators as a parameter. Replace its signature and the two references to `_SEPARATORS`:

```python
def _split_recursive(
    text: str, max_chars: int, separators: tuple[str, ...], start_idx: int = 0
) -> list[str]:
    """Recursively split text down to pieces ≤ max_chars, preferring natural boundaries."""
    if len(text) <= max_chars:
        return [text]
    for i in range(start_idx, len(separators)):
        sep = separators[i]
        if sep == "":
            return [text[j : j + max_chars] for j in range(0, len(text), max_chars)]
        if sep not in text:
            continue
        parts = text.split(sep)
        out: list[str] = []
        for idx, part in enumerate(parts):
            # Re-attach the separator (except for the first part) so heading
            # markers and sentence terminators survive the split. Recurse with
            # i+1 so we move on to finer separators — otherwise the re-attached
            # separator at the start of `piece` would re-trigger the same split
            # forever on a single oversized section.
            piece = part if idx == 0 else sep + part
            if len(piece) <= max_chars:
                out.append(piece)
            else:
                out.extend(_split_recursive(piece, max_chars, separators, i + 1))
        return out
    return [text]
```

Replace the existing `chunk_markdown` function with a shared helper plus two thin wrappers:

```python
def _chunk(
    text: str,
    separators: tuple[str, ...],
    chunk_size: int,
    chunk_overlap: int,
) -> list[str]:
    text = text.strip()
    if not text:
        return []
    char_size = max(1, chunk_size * 4)
    char_overlap = max(0, chunk_overlap * 4)
    if char_overlap >= char_size:
        char_overlap = char_size // 2
    if len(text) <= char_size:
        return [text]
    pieces = _split_recursive(text, char_size, separators)
    return _merge_with_overlap(pieces, char_size, char_overlap)


def chunk_markdown(
    text: str,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> list[str]:
    """Split markdown into chunks, preferring heading/paragraph boundaries.

    chunk_size and chunk_overlap are in approximate tokens (× 4 for char budget).
    """
    return _chunk(text, _SEPARATORS, chunk_size, chunk_overlap)


def chunk_code(
    text: str,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> list[str]:
    """Split source code into chunks, preferring class/def/function boundaries."""
    return _chunk(text, _CODE_SEPARATORS, chunk_size, chunk_overlap)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/ingest/test_chunker.py -q`
Expected: PASS (7 passed — 4 existing markdown tests + 3 new code tests)

- [ ] **Step 5: Commit**

```bash
git add docgraph/ingest/chunker.py tests/ingest/test_chunker.py
git commit -m "feat: add code-aware chunking sharing the recursive splitter"
```

---

## Task 5: Add `file_path` to models

**Files:**
- Modify: `docgraph/models.py`
- Test: `tests/test_models.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_models.py  (append; add SearchResult/ChunkRecord to the import line)
from docgraph.models import DocumentRecord, DocumentStatus, SearchResult, ChunkRecord


def test_search_result_file_path_defaults_none():
    r = SearchResult(
        text="x", doc_id="d", filename="f", folder="",
        tags=[], chunk_index=0, score=1.0,
    )
    assert r.file_path is None


def test_chunk_record_accepts_file_path():
    c = ChunkRecord(
        id="d_0", doc_id="d", text="code", chunk_index=0,
        filename="dump.txt", folder="", tags=[], file_path="src/a.py",
    )
    assert c.file_path == "src/a.py"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_models.py -q`
Expected: FAIL — `TypeError: __init__() got an unexpected keyword argument 'file_path'`

- [ ] **Step 3: Write the implementation**

In `docgraph/models.py`, add `file_path: Optional[str] = None` to both dataclasses (after the existing `source_page` field):

```python
@dataclass
class ChunkRecord:
    id: str
    doc_id: str
    text: str
    chunk_index: int
    filename: str
    folder: str
    tags: list[str]
    source_page: Optional[int] = None
    file_path: Optional[str] = None


@dataclass
class SearchResult:
    text: str
    doc_id: str
    filename: str
    folder: str
    tags: list[str]
    chunk_index: int
    score: float
    source_page: Optional[int] = None
    file_path: Optional[str] = None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_models.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add docgraph/models.py tests/test_models.py
git commit -m "feat: add optional file_path to ChunkRecord and SearchResult"
```

---

## Task 6: Thread `file_path` through Chroma → search → MCP

**Files:**
- Modify: `docgraph/store/chroma.py:77-87`
- Modify: `docgraph/mcp/search.py:43-54`
- Modify: `docgraph/mcp/server.py:36-48`
- Test: `tests/store/test_chroma.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/store/test_chroma.py  (append)
def test_search_returns_file_path(tmp_data_dir):
    from docgraph.config import Config
    from docgraph.store.chroma import ChromaStore

    cfg = Config(data_dir=tmp_data_dir)
    cfg.ensure_dirs()
    store = ChromaStore(cfg)
    store.upsert_chunks([{
        "id": "d_0",
        "embedding": [0.1] * 768,
        "text": "def a(): return 1",
        "metadata": {
            "doc_id": "d", "filename": "dump.txt", "folder": "",
            "tags": "[]", "chunk_index": 0, "file_path": "src/a.py",
        },
    }])
    results = store.search(query_embedding=[0.1] * 768, top_k=1)
    assert results[0]["file_path"] == "src/a.py"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/store/test_chroma.py::test_search_returns_file_path -q`
Expected: FAIL — `KeyError: 'file_path'`

- [ ] **Step 3: Write the implementation**

In `docgraph/store/chroma.py`, add `file_path` to the result dict built in `search` (inside the `out.append({...})` block):

```python
            out.append({
                "id": chunk_id,
                "text": result["documents"][0][i],
                "doc_id": meta.get("doc_id", ""),
                "filename": meta.get("filename", ""),
                "folder": meta.get("folder", ""),
                "tags": chunk_tags,
                "chunk_index": int(meta.get("chunk_index", 0)),
                "score": score,
                "source_page": meta.get("source_page"),
                "file_path": meta.get("file_path"),
            })
```

In `docgraph/mcp/search.py`, add `file_path` to the `SearchResult(...)` construction:

```python
            SearchResult(
                text=r["text"],
                doc_id=r["doc_id"],
                filename=r["filename"],
                folder=r["folder"],
                tags=r["tags"],
                chunk_index=r["chunk_index"],
                score=r["score"],
                source_page=r.get("source_page"),
                file_path=r.get("file_path"),
            )
```

In `docgraph/mcp/server.py`, add `file_path` to the `payload` dict in `search_documents`:

```python
        payload = [
            {
                "text": r.text,
                "doc_id": r.doc_id,
                "filename": r.filename,
                "folder": r.folder,
                "tags": r.tags,
                "chunk_index": r.chunk_index,
                "score": round(r.score, 4),
                "source_page": r.source_page,
                "file_path": r.file_path,
            }
            for r in results
        ]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/store/test_chroma.py tests/mcp -q`
Expected: PASS (new test passes; existing chroma + mcp tests still pass)

- [ ] **Step 5: Commit**

```bash
git add docgraph/store/chroma.py docgraph/mcp/search.py docgraph/mcp/server.py tests/store/test_chroma.py
git commit -m "feat: surface file_path through search results and MCP payload"
```

---

## Task 7: Route code dumps in the indexer

**Files:**
- Modify: `docgraph/ingest/indexer.py`
- Test: `tests/ingest/test_indexer.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/ingest/test_indexer.py  (append)
import json as _json


@pytest.mark.asyncio
@respx.mock
async def test_index_document_routes_repomix_dump(tmp_data_dir):
    cfg = Config(data_dir=tmp_data_dir)
    cfg.ensure_dirs()
    sqlite = SQLiteStore(cfg)
    sqlite.init_schema()
    files = FileStore(cfg)
    chroma = ChromaStore(cfg)
    embedder = OllamaEmbedder(cfg.ollama_url, cfg.ollama_model)

    def _embed(request):
        n = len(_json.loads(request.content)["input"])
        return httpx.Response(200, json={"embeddings": [[0.1] * 768] * n})

    respx.post(f"{cfg.ollama_url}/api/embed").mock(side_effect=_embed)

    dump = (
        "================\nFile: src/a.py\n================\n"
        "def a():\n    return 1\n\n"
        "================\nFile: src/b.py\n================\n"
        "def b():\n    return 2\n"
    )
    doc = DocumentRecord(id="doc_cd", filename="repo.txt", folder="", tags=[])
    sqlite.insert_document(doc)
    orig = files.save_original("doc_cd", "repo.txt", dump.encode("utf-8"))

    indexer = Indexer(cfg, sqlite, files, chroma, embedder)
    await indexer.index_document("doc_cd", orig)

    updated = sqlite.get_document("doc_cd")
    assert updated.status == DocumentStatus.READY
    assert updated.chunk_count >= 2
    results = chroma.search(query_embedding=[0.1] * 768, top_k=5)
    file_paths = {r["file_path"] for r in results}
    assert {"src/a.py", "src/b.py"} <= file_paths
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/ingest/test_indexer.py::test_index_document_routes_repomix_dump -q`
Expected: FAIL — the dump is sent through markitdown/markdown chunking, so chunks have no `file_path` (assertion on `file_paths` fails).

- [ ] **Step 3: Write the implementation**

In `docgraph/ingest/indexer.py`, extend the imports:

```python
from docgraph.ingest.chunker import chunk_code, chunk_markdown
from docgraph.ingest.code_dump import detect_repomix, infer_language, parse_repomix
```

Add a module-level helper near the top (after the `logger = ...` line):

```python
def _read_text_prefix(path: Path, limit: int = 65536) -> str:
    """Read up to `limit` chars of a file as text, ignoring decode errors."""
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        return f.read(limit)
```

Replace the body of `index_document` so it detects and routes:

```python
    async def index_document(self, doc_id: str, original_path: Path) -> None:
        doc = self._sqlite.get_document(doc_id)
        if doc is None:
            raise ValueError(f"document not found: {doc_id}")
        logger.info(
            "indexing file doc_id=%s path=%s", doc_id, original_path.name
        )
        try:
            prefix = await asyncio.to_thread(_read_text_prefix, original_path)
            if detect_repomix(prefix):
                full_text = await asyncio.to_thread(
                    original_path.read_text, "utf-8", "ignore"
                )
                await self.index_code_dump(doc_id, full_text)
                return
            self._progress(doc_id, 5, "Converting to text (5%)")
            markdown = await asyncio.to_thread(
                convert_file_to_markdown, original_path
            )
            logger.debug(
                "converted doc_id=%s markdown_chars=%d", doc_id, len(markdown)
            )
            self._progress(doc_id, 20, "Converted to markdown (20%)")
            await self.index_markdown(doc_id, markdown)
        except Exception as exc:
            self._sqlite.update_status(
                doc_id,
                DocumentStatus.ERROR,
                error_message=str(exc),
            )
            raise
```

Add the new `index_code_dump` method (place it right after `index_markdown`):

```python
    async def index_code_dump(self, doc_id: str, text: str) -> None:
        doc = self._sqlite.get_document(doc_id)
        if doc is None:
            raise ValueError(f"document not found: {doc_id}")
        try:
            self._progress(doc_id, 10, "Parsing code dump (10%)")
            parsed = parse_repomix(text)
            if not parsed:
                raise ValueError("could not parse code dump")
            md_path = self._files.save_markdown(doc_id, text)
            self._progress(doc_id, 28, "Splitting code into chunks (28%)")
            chunk_texts: list[str] = []
            chunk_files: list[str] = []
            for file_path, content in parsed:
                pieces = await asyncio.to_thread(
                    chunk_code,
                    content,
                    self._cfg.chunk_size,
                    self._cfg.chunk_overlap,
                )
                for piece in pieces:
                    chunk_texts.append(piece)
                    chunk_files.append(file_path)
            if not chunk_texts:
                raise ValueError("no chunks produced from code dump")
            if len(chunk_texts) > self._cfg.max_chunks_per_doc:
                raise ValueError(
                    f"document exceeds max_chunks_per_doc "
                    f"({len(chunk_texts)} > {self._cfg.max_chunks_per_doc}); "
                    f"increase DOCGRAPH_MAX_CHUNKS_PER_DOC or split the source"
                )
            self._progress(
                doc_id, 42, f"Chunked into {len(chunk_texts)} parts (42%)"
            )
            vectors = await self._embed_with_progress(doc_id, chunk_texts)
            self._progress(doc_id, 96, "Saving to index (96%)")
            chroma_chunks = []
            for i, (piece, vec, file_path) in enumerate(
                zip(chunk_texts, vectors, chunk_files)
            ):
                metadata = {
                    "doc_id": doc_id,
                    "filename": doc.filename,
                    "folder": doc.folder,
                    # JSON-encoded so tags containing commas survive round-tripping.
                    "tags": json.dumps(doc.tags),
                    "chunk_index": i,
                    "file_path": file_path,
                    "language": infer_language(file_path) or "",
                }
                chroma_chunks.append({
                    "id": f"{doc_id}_{i}",
                    "embedding": vec,
                    "text": piece,
                    "metadata": metadata,
                })
            self._chroma.upsert_chunks(chroma_chunks)
            self._sqlite.update_status(
                doc_id,
                DocumentStatus.READY,
                chunk_count=len(chunk_texts),
                markdown_path=str(md_path),
            )
        except Exception as exc:
            self._sqlite.update_status(
                doc_id,
                DocumentStatus.ERROR,
                error_message=str(exc),
            )
            raise
```

> Note: `language` is stored as `""` when unknown because ChromaDB metadata values may not be `None`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/ingest/test_indexer.py -q`
Expected: PASS (existing `test_index_document_success` still passes; new routing test passes)

- [ ] **Step 5: Commit**

```bash
git add docgraph/ingest/indexer.py tests/ingest/test_indexer.py
git commit -m "feat: route Repomix code dumps to code-aware indexing"
```

---

## Task 8: Full-suite verification

- [ ] **Step 1: Run the entire test suite**

Run: `python -m pytest -q`
Expected: PASS — all tests green (no regressions in embed/web/mcp/store/ingest).

- [ ] **Step 2: If anything fails, fix it before proceeding**

Use superpowers:systematic-debugging on any failure. Do not mark complete with failing tests.

---

## Self-Review Notes

- **Spec §4.1 (detect/parse):** Tasks 1–3. **§4.2 (chunker):** Task 4. **§4.3 (indexer routing):** Task 7. **§4.4 (metadata threading):** Tasks 5–6. **§6 (error handling):** `parse_repomix` empty → `ValueError` in Task 7; unknown language → `""`; `max_chunks_per_doc` guard preserved in Task 7. **§7 (testing):** covered across Tasks 1–7 plus full-suite Task 8.
- **Type consistency:** `parse_repomix` returns `list[tuple[str, str]]` (used in Task 7); `chunk_code(text, chunk_size, chunk_overlap)` signature matches Task 4 definition; `file_path` field name is identical across models, chroma, search, server, indexer.
- **No placeholders:** every code step contains complete code; every run step has an exact command + expected result.
