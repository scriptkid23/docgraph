# Plan chunk-04 — Language-Aware Dispatch in Repomix

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Inside a Repomix dump, route each parsed file to the chunker that matches its language: markdown files use `chunk_markdown_sections`, JSON/YAML use a structure-aware chunker (or just `chunk_markdown` as a safe default), code files use `chunk_code` (today) and later the tree-sitter chunker (chunk-05).

**Why:** `Indexer.index_code_dump` currently calls `chunk_code` for every file, including `.md`, `.yaml`, `.json`. This means a `README.md` inside a Repomix dump never benefits from heading-aware splitting; a YAML config gets chopped on `\n\n` instead of top-level keys; JSON gets cut mid-object. Smallest-effort, biggest-clarity fix in the lineup.

**Architecture:** Replace the unconditional `chunk_code` call in `index_code_dump` with a `dispatch_chunker(language)` lookup. Provide a small registry mapping language → chunker function. JSON/YAML use a leaf-aware chunker that prefers top-level key boundaries.

**Depends on:** chunk-01 (tokenizer), chunk-02 (heading-aware md). Optional: chunk-03 (atomic fences for `.md` inside dumps).
**Blocks:** chunk-05 (tree-sitter chunker plugs into the same dispatcher).

---

## File Structure

- **Modify** `docgraph/ingest/code_dump.py` — extend `_LANGUAGE_BY_EXT` if needed; expose `LANGUAGE_BY_EXT` as a constant.
- **Create** `docgraph/ingest/lang_dispatch.py` — `dispatch_chunker(language) -> Callable`.
- **Create** `docgraph/ingest/structured_chunker.py` — `chunk_json`, `chunk_yaml` that prefer top-level boundaries.
- **Modify** `docgraph/ingest/indexer.py:index_code_dump` — call dispatcher per file.
- **Tests:** `tests/ingest/test_lang_dispatch.py`, `tests/ingest/test_structured_chunker.py`, extend `tests/ingest/test_indexer.py`.

---

## Task 1: JSON / YAML structure-aware chunker

**Files:**
- Create: `docgraph/ingest/structured_chunker.py`
- Test: `tests/ingest/test_structured_chunker.py`

- [ ] **Step 1: Failing test**

```python
# tests/ingest/test_structured_chunker.py
from docgraph.ingest.structured_chunker import chunk_json, chunk_yaml


def test_chunk_json_splits_at_top_level_keys():
    text = '{\n  "a": 1,\n  "b": 2,\n  "c": "long" \n}'
    chunks = chunk_json(text, chunk_size=8, chunk_overlap=0)
    assert any('"a"' in c for c in chunks)
    assert any('"c"' in c for c in chunks)


def test_chunk_yaml_splits_at_top_level_keys():
    text = "alpha:\n  x: 1\nbeta:\n  y: 2\ngamma:\n  z: 3\n"
    chunks = chunk_yaml(text, chunk_size=8, chunk_overlap=0)
    assert sum("alpha:" in c for c in chunks) == 1
    assert sum("beta:" in c for c in chunks) == 1
    assert sum("gamma:" in c for c in chunks) == 1
```

- [ ] **Step 2: Run — expect FAIL.**

- [ ] **Step 3: Implement**
  - `chunk_json`: parse with `json.loads`; for top-level objects, render each key/value pair as a string (`"<key>": <value-pretty-printed>`); use the recursive merge to pack pairs into chunks. For top-level arrays, treat each element as a unit. Fall back to `chunk_code` separators if parse fails.
  - `chunk_yaml`: split on lines that start at column 0 with `^[A-Za-z_][\w-]*:` (top-level key heuristic). No PyYAML dep needed for the chunker.

- [ ] **Step 4: Run — expect PASS.**

- [ ] **Step 5: Commit** — `feat(chunk-04): structure-aware chunkers for JSON and YAML`

---

## Task 2: Language dispatcher

**Files:**
- Create: `docgraph/ingest/lang_dispatch.py`
- Test: `tests/ingest/test_lang_dispatch.py`

- [ ] **Step 1: Failing test**

```python
# tests/ingest/test_lang_dispatch.py
from docgraph.ingest.lang_dispatch import dispatch_chunker
from docgraph.ingest.chunker import chunk_code, chunk_markdown
from docgraph.ingest.structured_chunker import chunk_json, chunk_yaml


def test_dispatch_returns_markdown_for_md():
    assert dispatch_chunker("markdown") is chunk_markdown


def test_dispatch_returns_structured_for_json_yaml():
    assert dispatch_chunker("json") is chunk_json
    assert dispatch_chunker("yaml") is chunk_yaml


def test_dispatch_returns_code_for_known_languages():
    for lang in ("python", "go", "rust", "java"):
        assert dispatch_chunker(lang) is chunk_code


def test_dispatch_returns_code_for_unknown():
    # Fail-safe: unknown extension still gets chunked, just less smartly.
    assert dispatch_chunker(None) is chunk_code
    assert dispatch_chunker("brainfuck") is chunk_code
```

- [ ] **Step 2: Run — expect FAIL.**

- [ ] **Step 3: Implement**

```python
# docgraph/ingest/lang_dispatch.py
from __future__ import annotations
from typing import Callable

from docgraph.ingest.chunker import chunk_code, chunk_markdown
from docgraph.ingest.structured_chunker import chunk_json, chunk_yaml

_REGISTRY: dict[str, Callable] = {
    "markdown": chunk_markdown,
    "json": chunk_json,
    "yaml": chunk_yaml,
}


def dispatch_chunker(language: str | None) -> Callable:
    if language and language in _REGISTRY:
        return _REGISTRY[language]
    return chunk_code
```

- [ ] **Step 4: Run — expect PASS.**

- [ ] **Step 5: Commit** — `feat(chunk-04): language-aware chunker dispatcher`

---

## Task 3: Wire dispatcher into `index_code_dump`

**Files:**
- Modify: `docgraph/ingest/indexer.py`
- Test: extend `tests/ingest/test_indexer.py`

- [ ] **Step 1: Failing test** — a Repomix dump containing a `README.md` produces chunks whose first chunk starts with the heading breadcrumb (proving `chunk_markdown_sections` ran for that file).

- [ ] **Step 2: Run — expect FAIL.**

- [ ] **Step 3: Implement**
  - Replace the inner loop in `index_code_dump`:

    ```python
    for file_path, content in parsed:
        language = infer_language(file_path)
        chunker_fn = dispatch_chunker(language)
        pieces = await asyncio.to_thread(
            chunker_fn,
            content,
            self._cfg.chunk_size,
            self._cfg.chunk_overlap,
        )
    ```

  - For markdown, prefer `chunk_markdown_sections` so chunks pick up heading prefixes; for the others, `pieces` stays a `list[str]`.
  - Keep `metadata["language"]` as today; add `metadata["chunker"] = chunker_fn.__name__` for debuggability.

- [ ] **Step 4: Run — expect PASS.**

- [ ] **Step 5: Commit** — `feat(chunk-04): repomix dump dispatches per-file by language`

---

## Acceptance Criteria

- [ ] A `README.md` inside a Repomix dump produces chunks whose body starts with `Heading > Subheading\n\n`.
- [ ] A `package.json` inside a Repomix dump produces chunks whose top-level keys are not split mid-pair.
- [ ] A `.py` inside a Repomix dump still uses `chunk_code` (no regression).
- [ ] `metadata["chunker"]` is populated and visible in MCP search results for debugging.
