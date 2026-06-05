# DocGraph Chunker Improvements — Design Spec

**Date:** 2026-06-03
**Status:** Proposed
**Author:** evaluation follow-up
**Scope:** `docgraph/ingest/chunker.py`, `docgraph/ingest/indexer.py`, `docgraph/mcp/search.py`, plus new helpers under `docgraph/ingest/`.

## 1. Context

The current chunker (`docgraph/ingest/chunker.py`) is a recursive character splitter
modelled on LangChain's `RecursiveCharacterTextSplitter`. It works for English
markdown prototypes but, per the 2026-06-03 evaluation, has six gaps relative to
modern RAG best practice:

1. **Char-based token estimate** (`chars / 4`) silently truncates non-English chunks
   on small-context embedders (e.g. `multilingual-e5-base`, 512-token cap).
2. **Heading hierarchy is discarded** after splitting — no breadcrumb in chunk
   text or metadata, hurting retrieval disambiguation across similar sections.
3. **Code fences and markdown tables can be cut in half** when a paragraph or
   heading-less section exceeds the budget, producing unrenderable chunks.
4. **Repomix dump always uses `chunk_code`** even for `.md`/`.json`/`.yaml` files
   inside the dump.
5. **Code chunker is regex-based** (matches `class`/`def`/`func`/`function`
   prefixes only); Java/C#/Rust/Kotlin/Swift fall back to blank-line splits and
   the project already ships a tree-sitter index in `.codegraph/`.
6. **No content-hash dedup, no MMR diversity, no rerank** in the retrieval path —
   boilerplate from URL crawls bloats the index and identical-section results
   crowd out diverse top-k.

## 2. Goal

Move from "passable structural chunking" to **structure-aware, token-accurate,
language-aware chunking** without sacrificing the project's existing simplicity
or local-first ethos.

Non-goals (this round):
- Semantic chunking by sentence-similarity drops (compute-heavy, deferred).
- Late chunking with full-doc embedding (requires long-context model + ColBERT-
  style storage, deferred).
- LLM-generated contextual headers (Anthropic Contextual Retrieval) — costly,
  the deterministic heading-path variant captures most of the win.

## 3. Plan Inventory

| Plan | Title | Priority | Effort | Depends on |
|---|---|---|---|---|
| [01](../plans/2026-06-03-plan-chunk-01-tokenizer.md) | Real tokenizer-based chunk budgets | P0 | 1 day | — |
| [02](../plans/2026-06-03-plan-chunk-02-heading-prefix.md) | Markdown heading-path contextual prefix | P0 | 1 day | 01 (subtract prefix from budget) |
| [03](../plans/2026-06-03-plan-chunk-03-atomic-fences.md) | Atomic code-fence and table preservation | P1 | 1–2 days | 01 |
| [04](../plans/2026-06-03-plan-chunk-04-lang-dispatch.md) | Language-aware dispatch in Repomix | P1 | 0.5 day | — |
| [05](../plans/2026-06-03-plan-chunk-05-treesitter-code.md) | Tree-sitter AST code chunker | P0 (long-term) | 2–3 days | 04 |
| [06](../plans/2026-06-03-plan-chunk-06-search-quality.md) | Dedup + MMR diversity + optional rerank | P2 | 2 days | — |

Suggested execution order: **01 → 02 → 04 → 03 → 06 → 05**. Plan 05 is the
biggest moat but also the largest effort; do it once 01–04 are stable and the
tokenizer + dispatch infra it needs is in place.

## 4. Cross-cutting Concerns

### 4.1 Backward compatibility

All existing chunks in ChromaDB stay readable. Each plan adds NEW metadata fields
(`heading_path`, `chunk_hash`, `tokens`, `lang_chunker`) without removing any.
Search code defaults to empty string when a field is missing, so legacy chunks
keep working until they are re-indexed.

### 4.2 Re-index trigger

Any of plans 01, 02, 03, 05 changes chunk content/boundaries, so re-indexing is
required to see retrieval improvement. The existing `Indexer.reindex_document`
path covers this; we add a `docgraph reindex --all` CLI command in plan 01 to
make bulk migration explicit.

### 4.3 Configuration knobs

New `Config` fields, all optional with sensible defaults:

| Field | Default | Plan |
|---|---|---|
| `tokenizer_source` | `"auto"` | 01 |
| `heading_prefix_enabled` | `True` | 02 |
| `atomic_blocks_enabled` | `True` | 03 |
| `code_chunker` | `"regex"` (`"treesitter"` opt-in) | 05 |
| `dedup_enabled` | `True` | 06 |
| `mmr_lambda` | `0.5` | 06 |
| `rerank_enabled` | `False` | 06 |
| `rerank_model` | `"bge-reranker-base"` | 06 |

### 4.4 Observability

Each chunk gains a `tokens: int` metadata field once plan 01 lands; this lets us
write a one-shot diagnostic CLI (`docgraph stats chunks`) showing min/median/p95
token counts per doc — crucial for catching silent truncation.

## 5. Test Strategy

Every plan ships with TDD-style tests (`tests/ingest/test_chunker_*.py`,
`tests/store/test_chroma_*.py`, `tests/mcp/test_search_*.py`). Two integration
suites, marked `@pytest.mark.integration`, exercise:

- A multilingual corpus (English + Vietnamese + Chinese) to verify no chunk
  exceeds the embedder's `max_seq_len` after plan 01.
- A "synthetic markdown book" fixture with deeply-nested headings and a fenced
  code block that spans the chunk budget — verifies plans 02 and 03.

## 6. Migration Notes

- Plans 01, 02, 03, 05 require a re-index to take effect. Plan 04 also requires
  re-index for any Repomix dump previously ingested.
- Plan 06 is search-only; no re-index needed.
- The `chunk_count` shown in the Web UI may shift after re-index due to changed
  boundaries — this is expected.
