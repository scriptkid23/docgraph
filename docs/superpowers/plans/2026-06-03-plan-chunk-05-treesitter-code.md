# Plan chunk-05 — Tree-sitter AST Code Chunker

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Chunk source code at AST function/class/method boundaries using tree-sitter, instead of regex line-prefixes. This gives semantically-complete chunks for every supported language (Python, TypeScript, JavaScript, Go, Rust, Java, C#, C/C++, Ruby, PHP, Swift, Kotlin), and is the single biggest quality win for code-dump ingestion.

**Why:** The current `chunk_code` regex matches only `class `, `def `, `func `, `function ` at line-start. Java/C#/Kotlin/Swift use access modifiers (`public class`, `private fn`); Rust uses `fn` (not `func`); C++ uses return-type prefix. All of those silently fall back to blank-line splits, producing half-functions. Meanwhile the project already ships **tree-sitter** in `.codegraph/`, so the parser infra is paid for. This is the moat.

**Architecture:** Add `docgraph/ingest/treesitter_chunker.py` that lazily loads a `tree_sitter_languages.get_language(lang)` parser, walks the syntax tree, and emits text spans corresponding to top-level declarations (`function_definition`, `class_definition`, `method_declaration`, etc., per-language node-type set). Spans larger than the budget are split internally at child-node boundaries (statement-level). Spans much smaller than the budget are merged to neighbouring siblings (greedy, never crossing a class boundary).

This chunker plugs into the chunk-04 dispatcher behind a `code_chunker = "treesitter"` config flag (default `"regex"` until validated).

**Depends on:** chunk-01 (tokenizer), chunk-04 (dispatcher).
**Optional reuse:** the existing Rust codegraph indexer at `crates/codegraph` already invokes tree-sitter — investigate exposing a Python binding so we don't duplicate parser setup.

**Spec:** `docs/superpowers/specs/2026-06-03-chunker-improvements-design.md` §3.

---

## File Structure

- **Create** `docgraph/ingest/treesitter_chunker.py` — main entry `chunk_code_treesitter(text, language, chunk_size, chunk_overlap, counter)`.
- **Create** `docgraph/ingest/ts_node_specs.py` — per-language map of "split node types" (e.g. `python: {"function_definition", "class_definition", "decorated_definition"}`).
- **Modify** `docgraph/ingest/lang_dispatch.py` — when `cfg.code_chunker == "treesitter"`, prefer the new chunker; fall back to `chunk_code` if parser unavailable for a language.
- **Modify** `docgraph/config.py` — `code_chunker: str = "regex"` (`"regex"` | `"treesitter"`).
- **Modify** `pyproject.toml` — add optional dep `tree-sitter-languages` (or use the codegraph crate via PyO3 binding — see Task 6).
- **Tests:** `tests/ingest/test_treesitter_chunker.py`, fixtures under `tests/fixtures/code/`.

---

## Task 1: Per-language split-node spec

**Files:**
- Create: `docgraph/ingest/ts_node_specs.py`
- Test: `tests/ingest/test_ts_node_specs.py`

- [ ] **Step 1: Failing test**

```python
# tests/ingest/test_ts_node_specs.py
from docgraph.ingest.ts_node_specs import split_node_types


def test_split_node_types_python():
    s = split_node_types("python")
    assert "function_definition" in s
    assert "class_definition" in s
    assert "decorated_definition" in s


def test_split_node_types_typescript():
    s = split_node_types("typescript")
    assert "function_declaration" in s
    assert "class_declaration" in s
    assert "method_definition" in s


def test_split_node_types_unknown_returns_empty():
    assert split_node_types("brainfuck") == set()
```

- [ ] **Step 2: Run — expect FAIL.**

- [ ] **Step 3: Implement** — a static dict keyed by language → `frozenset[str]`:

```python
# docgraph/ingest/ts_node_specs.py
_SPECS: dict[str, frozenset[str]] = {
    "python":     frozenset({"function_definition", "class_definition", "decorated_definition"}),
    "javascript": frozenset({"function_declaration", "class_declaration", "method_definition", "arrow_function"}),
    "typescript": frozenset({"function_declaration", "class_declaration", "method_definition", "interface_declaration"}),
    "go":         frozenset({"function_declaration", "method_declaration", "type_declaration"}),
    "rust":       frozenset({"function_item", "impl_item", "struct_item", "enum_item", "trait_item"}),
    "java":       frozenset({"method_declaration", "class_declaration", "interface_declaration", "constructor_declaration"}),
    "csharp":     frozenset({"method_declaration", "class_declaration", "interface_declaration"}),
    "cpp":        frozenset({"function_definition", "class_specifier", "struct_specifier"}),
    "c":          frozenset({"function_definition", "struct_specifier"}),
    "ruby":       frozenset({"method", "class", "module"}),
    "php":        frozenset({"function_definition", "method_declaration", "class_declaration"}),
}

def split_node_types(language: str) -> frozenset[str]:
    return _SPECS.get(language, frozenset())
```

- [ ] **Step 4: Run — expect PASS.**

- [ ] **Step 5: Commit** — `feat(chunk-05): per-language tree-sitter split-node spec`

---

## Task 2: Parser loader with graceful fallback

**Files:**
- Create: `docgraph/ingest/treesitter_chunker.py`
- Test: `tests/ingest/test_treesitter_chunker.py`

- [ ] **Step 1: Failing test (skipped if `tree_sitter_languages` not installed)**

```python
import pytest
ts_lang = pytest.importorskip("tree_sitter_languages")

from docgraph.ingest.treesitter_chunker import _get_parser


def test_get_parser_returns_parser_for_python():
    p = _get_parser("python")
    assert p is not None
    tree = p.parse(b"def f():\n    return 1\n")
    assert tree.root_node.type in {"module", "program"}


def test_get_parser_returns_none_for_unknown():
    assert _get_parser("brainfuck") is None
```

- [ ] **Step 2: Run — expect FAIL.**

- [ ] **Step 3: Implement**

```python
# docgraph/ingest/treesitter_chunker.py
from __future__ import annotations

from functools import lru_cache
from typing import Any


@lru_cache(maxsize=32)
def _get_parser(language: str) -> Any | None:
    try:
        from tree_sitter_languages import get_parser
    except ImportError:
        return None
    try:
        return get_parser(language)
    except Exception:
        return None
```

- [ ] **Step 4: Run — expect PASS.**

- [ ] **Step 5: Commit** — `feat(chunk-05): cached tree-sitter parser loader`

---

## Task 3: AST walker emits declaration spans

**Files:**
- Modify: `docgraph/ingest/treesitter_chunker.py`
- Test: extend `tests/ingest/test_treesitter_chunker.py`

- [ ] **Step 1: Failing test**

```python
def test_chunker_keeps_each_python_function_intact():
    from docgraph.ingest.treesitter_chunker import chunk_code_treesitter
    code = """
def alpha():
    return 1


def beta(x):
    return x * 2


class Gamma:
    def m(self):
        return 3
"""
    chunks = chunk_code_treesitter(code, "python", chunk_size=4096, chunk_overlap=0)
    joined = "\n---\n".join(chunks)
    # Each declaration should appear in exactly one chunk, fully.
    for sig in ("def alpha", "def beta", "class Gamma"):
        assert joined.count(sig) == 1
    # Methods inside class stay with the class (not split into siblings).
    gamma = next(c for c in chunks if "class Gamma" in c)
    assert "def m(self)" in gamma
```

- [ ] **Step 2: Run — expect FAIL.**

- [ ] **Step 3: Implement**

  - Walk the parse tree's top-level children.
  - For each child: if its `type` ∈ `split_node_types(language)`, emit `(node.start_byte, node.end_byte, node.type)`. Otherwise (imports, top-level statements, comments, blank lines) accumulate into a "preamble" / "between-decl" buffer that gets glued to the next declaration so context (decorators, doc comments) stays attached.
  - Convert byte ranges to text using `code.encode("utf-8")` (tree-sitter is byte-indexed).
  - Always include the declaration's leading attached comments (preceding contiguous comment lines + decorators).

- [ ] **Step 4: Run — expect PASS.**

- [ ] **Step 5: Commit** — `feat(chunk-05): AST walker emits declaration spans with attached comments`

---

## Task 4: Pack spans under the budget; split oversize spans

**Files:**
- Modify: `docgraph/ingest/treesitter_chunker.py`
- Test: extend `tests/ingest/test_treesitter_chunker.py`

- [ ] **Step 1: Failing tests**

```python
def test_chunker_merges_small_siblings_under_budget():
    code = "\n\n".join(f"def f{i}():\n    return {i}" for i in range(10))
    chunks = chunk_code_treesitter(code, "python", chunk_size=64, chunk_overlap=0)
    # ten tiny functions should pack into <10 chunks.
    assert 1 <= len(chunks) < 10


def test_chunker_splits_oversize_function_at_statement_boundary():
    body = "    x = 1\n" * 1000
    code = f"def big():\n{body}\n    return x\n"
    chunks = chunk_code_treesitter(code, "python", chunk_size=128, chunk_overlap=16)
    assert len(chunks) > 1
    # Continuation chunks should still mention the function signature as a header.
    assert all("def big" in c or c.startswith("# (continued") for c in chunks)
```

- [ ] **Step 2: Run — expect FAIL.**

- [ ] **Step 3: Implement**

  - Use the chunk-01 `TokenCounter` to measure each span.
  - Pack: greedy merge adjacent spans while `counter.count(merged) <= chunk_size`. Never merge across class boundaries (a method belongs to its class).
  - Oversize span: descend into its children — for `function_definition`, walk the `block` child statement-by-statement, packing each into a sub-chunk while keeping the signature `def big(...):` as a header on every sub-chunk (prepend `# (continued from <signature>)\n`).
  - Carry overlap: if a span produced N sub-chunks, the boundary overlap is the last `chunk_overlap` tokens of the previous sub-chunk's body (NOT the signature header — that's already context).

- [ ] **Step 4: Run — expect PASS.**

- [ ] **Step 5: Commit** — `feat(chunk-05): pack and split AST spans under token budget`

---

## Task 5: Wire into dispatcher with config flag

**Files:**
- Modify: `docgraph/ingest/lang_dispatch.py`, `docgraph/config.py`
- Test: extend `tests/ingest/test_lang_dispatch.py`

- [ ] **Step 1: Failing test** — when `cfg.code_chunker == "treesitter"` and `tree_sitter_languages` is installed, `dispatch_chunker("python")` returns the tree-sitter chunker; with `"regex"`, returns `chunk_code`.

- [ ] **Step 2: Run — expect FAIL.**

- [ ] **Step 3: Implement**

  - `Config.code_chunker: str = "regex"`.
  - `dispatch_chunker(language, code_chunker="regex")`: when `"treesitter"`, attempt `_get_parser(language)`; if returned, use a closure binding the language to `chunk_code_treesitter`; otherwise fall back to `chunk_code`.
  - The indexer reads `cfg.code_chunker` and threads it through.

- [ ] **Step 4: Run — expect PASS.**

- [ ] **Step 5: Commit** — `feat(chunk-05): config flag selects regex or tree-sitter code chunker`

---

## Task 6: Investigate reuse of codegraph crate (optional)

**Files:** `crates/codegraph/*`, new Python binding via PyO3.

- [ ] Review `crates/codegraph` to see whether it already exposes per-symbol byte ranges via PyO3.
- [ ] If yes, add a `docgraph_codegraph_chunks(text, language) -> list[(start, end, kind)]` binding and prefer it over `tree_sitter_languages` (avoids duplicate parser downloads, faster cold start).
- [ ] If no, keep `tree_sitter_languages` as an optional dep.
- [ ] Document the decision in the spec doc and link from this plan.
- [ ] Commit — `feat(chunk-05): explore reusing codegraph crate for chunking`

---

## Task 7: Multi-language fixture suite

**Files:** `tests/fixtures/code/{example.py,example.ts,example.go,example.rs,example.java,example.cs}`

- [ ] **Step 1: Add 30–80 line representative files for each language.**

- [ ] **Step 2: Parameterised test ensures**:
  - All public declarations show up exactly once.
  - No chunk mixes two distinct top-level declarations except via legitimate packing.
  - Chunks are ≤ `chunk_size` tokens.

- [ ] **Step 3: Run — expect PASS.**

- [ ] **Step 4: Commit** — `test(chunk-05): multi-language tree-sitter fixture suite`

---

## Acceptance Criteria

- [ ] For every supported language, no chunk contains a half-function (verified by AST re-parse: each chunk re-parses without error if treated as a fragment OR carries a `# (continued from ...)` header).
- [ ] Class/method association is preserved: a method's chunk either contains the enclosing class signature or carries a continuation header pointing to it.
- [ ] When `tree_sitter_languages` is unavailable for a language, dispatch falls back to `chunk_code` and the indexer logs INFO once.
- [ ] No regression in existing `test_chunker.py`, `test_indexer.py`.
- [ ] `metadata["chunker"]` reflects the actual chunker used (`"chunk_code_treesitter"` vs `"chunk_code"`).
