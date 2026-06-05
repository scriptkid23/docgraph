# Plan chunk-02 — Markdown Heading-Path Contextual Prefix

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Prepend the markdown heading breadcrumb (e.g. `Configuration > Embedding > OpenAI`) to every chunk's text before embedding, AND store it as `heading_path` metadata for citations.

**Why:** This is the highest-ROI structural improvement in 2025–2026 RAG ("contextual chunk headers" / deterministic Anthropic Contextual Retrieval). It disambiguates orphan paragraphs from similar sections elsewhere in the corpus, lifts hybrid-retrieval BM25 scores (heading words ride into the body), and makes citations natural.

**Architecture:** Replace the current "split-then-merge" pipeline with a heading-aware pre-pass that walks the markdown line-by-line, maintains a heading stack, and emits *sections* tagged with their breadcrumb. Each section is then chunked with the existing recursive splitter, with the breadcrumb prepended to every emitted chunk and **subtracted from the per-chunk token budget** so deeply-nested sections never overflow the embedder.

**Tech Stack:** Pure stdlib. (We don't pull in `markdown-it-py` because the heading-stack walker is ~40 lines.)

**Depends on:** chunk-01 (tokenizer-based budget — we need `counter.count(prefix)` to subtract correctly).

**Spec:** `docs/superpowers/specs/2026-06-03-chunker-improvements-design.md` §3.

---

## File Structure

- **Create** `docgraph/ingest/markdown_sections.py` — heading-stack walker that yields `(heading_path: list[str], body: str)` tuples.
- **Modify** `docgraph/ingest/chunker.py` — new public `chunk_markdown_sections()` that wraps the walker + per-section chunking + breadcrumb prefixing.
- **Modify** `docgraph/ingest/indexer.py` — call the new chunker, store `heading_path` in metadata.
- **Modify** `docgraph/store/chroma.py` and `docgraph/mcp/search.py` — surface `heading_path` in `SearchResult`.
- **Modify** `docgraph/mcp/server.py` — include `heading_path` in the search response payload.
- **Modify** `docgraph/models.py` — add `heading_path: list[str] | None` on `SearchResult`.
- **Modify** `frontend/` — render breadcrumb on each search hit (optional UI polish, separate task).
- **Modify** `skills/document/SKILL.md` — citation format becomes `[filename → Heading > Subheading, chunk N]`.

---

## Task 1: Heading-stack walker

**Files:**
- Create: `docgraph/ingest/markdown_sections.py`
- Test: `tests/ingest/test_markdown_sections.py`

- [ ] **Step 1: Failing test**

```python
# tests/ingest/test_markdown_sections.py
from docgraph.ingest.markdown_sections import walk_sections


def test_walk_sections_emits_heading_path_per_section():
    md = """# Title
intro

## A
body of A

### A.1
body of A.1

## B
body of B
"""
    sections = list(walk_sections(md))
    paths = [s.heading_path for s in sections]
    assert paths == [
        ["Title"],
        ["Title", "A"],
        ["Title", "A", "A.1"],
        ["Title", "B"],
    ]
    assert sections[2].body.strip() == "body of A.1"


def test_walk_sections_skips_atx_inside_code_fence():
    md = """# Title
text

```python
# this is not a heading
print("hi")
```

## Real Heading
body
"""
    sections = list(walk_sections(md))
    headings = [s.heading_path[-1] for s in sections]
    assert headings == ["Title", "Real Heading"]


def test_walk_sections_handles_no_heading_doc():
    md = "Just a paragraph with no heading at all.\n\nAnother paragraph."
    sections = list(walk_sections(md))
    assert len(sections) == 1
    assert sections[0].heading_path == []
    assert "Just a paragraph" in sections[0].body
```

- [ ] **Step 2: Run — expect FAIL.**

- [ ] **Step 3: Implement**

```python
# docgraph/ingest/markdown_sections.py
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterator

_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_FENCE = re.compile(r"^(```|~~~)")


@dataclass
class Section:
    heading_path: list[str]
    body: str


def walk_sections(md: str) -> Iterator[Section]:
    """Yield Section(heading_path, body) tuples honoring the heading hierarchy.

    - Lines inside fenced code blocks are never treated as headings.
    - Sections are emitted in document order.
    - The body excludes the heading line itself.
    - Pre-heading content (before any heading) is emitted with heading_path=[].
    """
    stack: list[str] = []
    pending: list[str] = []
    in_fence = False

    def flush() -> Iterator[Section]:
        if pending:
            body = "\n".join(pending).strip("\n")
            if body.strip():
                yield Section(heading_path=list(stack), body=body)
            pending.clear()

    for line in md.splitlines():
        if _FENCE.match(line):
            in_fence = not in_fence
            pending.append(line)
            continue
        if in_fence:
            pending.append(line)
            continue
        m = _HEADING.match(line)
        if m:
            yield from flush()
            level = len(m.group(1))
            title = m.group(2).strip()
            # truncate stack to (level - 1), then push the new heading
            stack = stack[: level - 1]
            while len(stack) < level - 1:
                stack.append("")
            stack.append(title)
            continue
        pending.append(line)
    yield from flush()
```

- [ ] **Step 4: Run — expect PASS.**

- [ ] **Step 5: Commit** — `feat(chunk-02): heading-stack walker for markdown`

---

## Task 2: Section-aware chunker

**Files:**
- Modify: `docgraph/ingest/chunker.py`
- Test: `tests/ingest/test_chunker.py`

- [ ] **Step 1: Failing test**

```python
def test_chunk_markdown_sections_prepends_breadcrumb():
    from docgraph.ingest.chunker import chunk_markdown_sections

    md = "# Top\n\n## Sub\n\n" + "word " * 800
    chunks = chunk_markdown_sections(md, chunk_size=128, chunk_overlap=16)
    assert chunks, "expected at least one chunk"
    for c in chunks:
        assert c.text.startswith("Top > Sub\n\n"), c.text[:50]
        assert c.heading_path == ["Top", "Sub"]


def test_chunk_markdown_sections_subtracts_prefix_from_budget():
    from docgraph.ingest.chunker import chunk_markdown_sections

    md = "# A\n\n## B\n\n## C\n\n## D\n\n## E\n\n" + "x " * 5000
    # deeply nested would overshoot if prefix isn't subtracted
    chunks = chunk_markdown_sections(md, chunk_size=64, chunk_overlap=8)
    for c in chunks:
        # rough check: total length under (chunk_size + slack) tokens
        assert len(c.text) <= 64 * 4 + 100
```

- [ ] **Step 2: Run — expect FAIL.**

- [ ] **Step 3: Implement**

  - Define a `MarkdownChunk` dataclass: `text: str`, `heading_path: list[str]`, `chunk_index: int`.
  - For each `Section` from the walker:
    1. Build `prefix = " > ".join(section.heading_path) + "\n\n"` (empty when no headings).
    2. Compute `prefix_tokens = counter.count(prefix)`.
    3. Run the existing recursive splitter on `section.body` with `chunk_size - prefix_tokens` budget.
    4. Yield `MarkdownChunk(text=prefix + body_chunk, heading_path=section.heading_path, ...)`.
  - Backward-compat: keep `chunk_markdown(text)` returning `list[str]` (calling the new path internally and dropping breadcrumbs); deprecate over time.

- [ ] **Step 4: Run — expect PASS** plus existing chunker tests stay green.

- [ ] **Step 5: Commit** — `feat(chunk-02): section-aware chunker with breadcrumb prefix`

---

## Task 3: Indexer stores `heading_path`

**Files:**
- Modify: `docgraph/ingest/indexer.py`
- Test: `tests/ingest/test_indexer.py`

- [ ] **Step 1: Failing test**

```python
async def test_index_markdown_records_heading_path(indexer, doc_id):
    await indexer.index_markdown(doc_id, "# T\n\n## S\n\n" + "x " * 1000)
    chunks = chroma.get_by_doc_id(doc_id)
    paths = {tuple(json.loads(c["metadata"]["heading_path"])) for c in chunks}
    assert ("T", "S") in paths
```

- [ ] **Step 2: Run — expect FAIL.**

- [ ] **Step 3: Implement**

  - Switch `index_markdown` from `chunk_markdown` to `chunk_markdown_sections`.
  - Store `metadata["heading_path"] = json.dumps(chunk.heading_path)` (list of strings, JSON-encoded for Chroma compatibility, mirroring how `tags` is handled).

- [ ] **Step 4: Run — expect PASS.**

- [ ] **Step 5: Commit** — `feat(chunk-02): indexer writes heading_path metadata`

---

## Task 4: Surface `heading_path` in search response

**Files:**
- Modify: `docgraph/store/chroma.py`, `docgraph/models.py`, `docgraph/mcp/search.py`, `docgraph/mcp/server.py`
- Test: `tests/store/test_chroma.py`, `tests/mcp/test_search.py`

- [ ] **Step 1: Failing tests** — verify `heading_path` round-trips through `ChromaStore.search` → `SearchService.search` → MCP tool payload.

- [ ] **Step 2: Run — expect FAIL.**

- [ ] **Step 3: Implement**

  - In `ChromaStore.search`, decode `heading_path` (mirror `_decode_tags` style — JSON list).
  - Add `heading_path: list[str] | None = None` to `SearchResult`.
  - In `mcp/server.py`, include `heading_path` in the JSON tool result.
  - Backward compat: `_decode_heading_path` returns `[]` when key absent.

- [ ] **Step 4: Run — expect PASS.**

- [ ] **Step 5: Commit** — `feat(chunk-02): surface heading_path through search to MCP clients`

---

## Task 5: Update `/document` skill citation format

**Files:**
- Modify: `skills/document/SKILL.md`

- [ ] Citation format becomes `[filename → Top > Sub, chunk N]` when `heading_path` is non-empty, falling back to current `[filename, chunk N]` otherwise.
- [ ] Commit — `docs(chunk-02): citation format includes heading breadcrumb`

---

## Task 6: Web UI breadcrumb (optional polish)

**Files:**
- Modify: `frontend/src/components/SearchResults.tsx` (or equivalent)

- [ ] Render heading breadcrumb above each result text. Skip if backend returns empty list.
- [ ] Commit — `feat(web)(chunk-02): show heading breadcrumb in search results`

---

## Acceptance Criteria

- [ ] Every newly-indexed markdown chunk starts with `Heading > Subheading\n\n` when at least one heading is present.
- [ ] `heading_path` metadata is JSON-encoded and survives a re-read from Chroma.
- [ ] No chunk exceeds `chunk_size` tokens because the prefix length is subtracted from the per-section budget.
- [ ] Existing tests pass; backward-compat `chunk_markdown(text)` still returns `list[str]`.
- [ ] MCP search tool returns `heading_path` in its payload.
