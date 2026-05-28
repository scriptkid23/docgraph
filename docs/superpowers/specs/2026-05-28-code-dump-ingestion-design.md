# Code-Dump Ingestion (Repomix plain/XML) — Design

**Date:** 2026-05-28
**Status:** Approved (design)

## 1. Problem & Goal

DocGraph converts each uploaded file to markdown, then chunks it on **markdown**
boundaries (`## headings`, blank lines) before embedding with nomic-embed-text.
This works for prose (books, PDFs, URLs) but mangles source code: a code file has
no markdown headings, so it gets sliced mid-function at arbitrary line breaks,
which degrades retrieval quality.

The user's workflow is to run **Repomix** (or Gitingest) on a repository, producing
one combined dump file, and upload that to DocGraph for codebase Q&A. Today that dump
is treated as one opaque markdown blob.

> Note: The "8192-token truncation" and "single giant vector" concerns from generic
> code-RAG advice do **not** apply to DocGraph — it already chunks into ~512-token
> pieces, well under nomic's 8192 limit. The real gap is markdown-only splitting.

**Goal:** Recognize an uploaded Repomix dump, split it back into per-file sections,
chunk each file with code-aware heuristics, and tag every chunk with its in-repo
`file_path` + `language` so search hits are attributable to a specific file.

## 2. Scope

**In scope:**
- Repomix **plain** and **XML** dump formats.
- Code-aware heuristic chunking (no AST / tree-sitter).
- One uploaded dump = one DocGraph document (existing upload/doc model unchanged).
- `file_path` + `language` metadata threaded into the search response.

**Out of scope (explicitly deferred):**
- Tree-sitter / AST-based splitting.
- Repo-folder walking (pointing DocGraph at a local directory).
- Gitingest format and Repomix **Markdown** style.
- Per-file documents (we keep one document per dump).

## 3. Decisions (from brainstorming)

| Question | Decision |
|---|---|
| Use case | Ingest source-code repos for codebase Q&A. |
| Entry point | Repomix/Gitingest **dump** uploaded as a single file. |
| Dump format | Repomix **plain / XML** (auto-detect, support both). |
| Within-file chunking | **A. Heuristic separators** — extend the existing recursive splitter; no new deps. |
| Document granularity | One document per dump; chunks carry per-file metadata. |
| `file_path` in search | In scope now (thin vertical slice through to the MCP response). |

## 4. Components

### 4.1 `docgraph/ingest/code_dump.py` (new)
- `detect_repomix(text: str) -> bool`
  - True if the text contains XML `<file path="...">` tags **or** the plain-style
    `================`/`File:` separator pattern.
  - Designed to be cheap — run on a bounded prefix of the file.
- `parse_repomix(text: str) -> list[tuple[str, str]]`
  - Returns `(file_path, content)` pairs for both plain and XML styles.
  - Skips the Repomix preamble / directory-summary block and empty sections.
  - Preserves file content verbatim (including blank lines / indentation).

### 4.2 `docgraph/ingest/chunker.py` (extend)
- Generalize `_split_recursive` to accept a `separators: tuple[str, ...]` parameter
  (single shared implementation; keeps the recursion-depth fix from the markdown path).
- Add `chunk_code(text, chunk_size, chunk_overlap) -> list[str]` using code separators:
  ```python
  _CODE_SEPARATORS = ("\nclass ", "\ndef ", "\nfunc ", "\nfunction ",
                      "\n\n", "\n", " ", "")
  ```
- `chunk_markdown` keeps its current separators. Both reuse `_merge_with_overlap`.

### 4.3 `docgraph/ingest/indexer.py` (extend)
- In `index_document`, read a **bounded text prefix** of the original file
  (utf-8, errors ignored). If `detect_repomix(prefix)` → route to new
  `index_code_dump(doc_id, full_text)`; else fall through to the existing
  `convert_file_to_markdown → chunk_markdown` path (unchanged).
- `index_code_dump`: `parse_repomix` → for each file `chunk_code(content)` and tag
  the resulting chunks with `file_path` + `language` → flatten →
  `_embed_with_progress` → upsert.
- Binaries (PDF/docx) never match the Repomix signature, so they are unaffected.

### 4.4 Metadata threading (`models.py`, `chroma.py`, `search.py`)
- Add optional `file_path` (and `language` where useful) to chunk metadata in the indexer.
- `chroma.search` result dict reads `meta.get("file_path")`.
- `search.py` maps it onto a new optional `SearchResult.file_path` field.
- `ChunkRecord` / `SearchResult` in `models.py` gain optional `file_path` (default `None`)
  so non-code documents are unaffected.
- Surface `file_path` in the MCP search response.

## 5. Data Flow

```
upload → save original → index_document
  ├─ read bounded prefix → detect_repomix?
  │   yes → parse_repomix → [(path, content)...]
  │          → per file: chunk_code(content); tag chunks with file_path + language
  │          → flatten → _embed_with_progress
  │          → upsert (metadata: doc_id, file_path, language, chunk_index, filename, folder, tags)
  │   no  → convert_file_to_markdown → chunk_markdown → embed         (unchanged path)
```

## 6. Error Handling

- Signature matched but `parse_repomix` finds 0 files → `ValueError("could not parse code dump")`,
  recorded as `ERROR` status (existing try/except pattern in `index_document`).
- Empty file section → skipped.
- Unknown file extension → `language = None`; still chunked (code separators degrade
  gracefully to blank-line / newline splitting).
- A whole-repo dump can exceed `max_chunks_per_doc` (default 5000). The existing guard
  and clear error message cover this; the user raises `DOCGRAPH_MAX_CHUNKS_PER_DOC` or
  splits the source. No code change.

## 7. Testing

- `tests/ingest/test_code_dump.py`
  - `detect_repomix`: true on plain + XML samples; false on prose / plain markdown.
  - `parse_repomix`: correct `(path, content)` pairs for plain and XML; preserves blank
    lines; handles trailing separators; skips preamble.
- `tests/ingest/test_chunker.py`
  - `chunk_code` keeps a small function intact; prefers `def`/`class` boundaries on a
    large file; produces no empty chunks; no recursion error on a very large file.
- `tests/ingest/test_indexer.py` (extend)
  - An uploaded Repomix dump routes to the code path and produces chunks carrying
    `file_path` metadata (mock embedder, matching existing indexer tests).

## 8. Notes / Open Items

- `chunk_size` default stays 512 tokens; code is denser than prose but the char≈token×4
  heuristic is close enough. Configurable via existing `DOCGRAPH_CHUNK_SIZE`.
- `language` inference is a simple extension → name map; unknown extensions are allowed.
