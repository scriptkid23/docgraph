# Plan chunk-01 — Real Tokenizer-Based Chunk Budgets

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the `chars / 4 ≈ tokens` heuristic with a real tokenizer so chunk budgets are accurate for non-English content (Vietnamese, Chinese, Japanese) and never silently exceed the embedder's `max_seq_len`.

**Why:** With the current heuristic a 2,048-char Vietnamese chunk can tokenize to 1,500+ tokens — well over `multilingual-e5-base`'s 512 cap — and the tokenizer truncates silently. This corrupts retrieval for the whole doc.

**Architecture:** Add `docgraph/ingest/tokenizer.py` exposing a `TokenCounter` protocol. Three implementations: HuggingFace `tokenizers` (preferred for ONNX models), OpenAI `tiktoken` (for `text-embedding-3-*`), and a `CharRatioCounter` fallback that keeps current behaviour when no tokenizer can be loaded. The chunker's char budget becomes a real token budget. Each chunk's actual token count is stored in metadata.

**Tech Stack:** `tokenizers` (HF), optional `tiktoken`.

**Depends on:** —
**Blocks:** chunk-02 (heading prefix needs to subtract its tokens from the body budget), chunk-03 (atomic-fence overflow handling needs accurate counts).

**Spec:** `docs/superpowers/specs/2026-06-03-chunker-improvements-design.md` §3, §4.3.

---

## File Structure

- **Create** `docgraph/ingest/tokenizer.py` — `TokenCounter` protocol + 3 impls + factory `get_token_counter(cfg)`.
- **Modify** `docgraph/ingest/chunker.py` — accept a `TokenCounter`, switch internal budget from chars to tokens.
- **Modify** `docgraph/ingest/indexer.py` — pass tokenizer through; record `tokens` in chunk metadata.
- **Modify** `docgraph/config.py` — add `tokenizer_source: str = "auto"`.
- **Modify** `docgraph/cli.py` — add `docgraph reindex --all` for bulk migration.
- **Create** `tests/ingest/test_tokenizer.py`, extend `tests/ingest/test_chunker.py`.

Add deps: `poetry add tokenizers` (already a transitive dep of HF stack but make it explicit).

---

## Task 1: TokenCounter protocol + char-ratio fallback

**Files:**
- Create: `docgraph/ingest/tokenizer.py`
- Test: `tests/ingest/test_tokenizer.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/ingest/test_tokenizer.py
from docgraph.ingest.tokenizer import CharRatioCounter, TokenCounter


def test_char_ratio_counter_uses_chars_over_4():
    c: TokenCounter = CharRatioCounter()
    assert c.count("hello world") == 3   # 11 / 4 rounded up
    assert c.count("") == 0


def test_char_ratio_counter_truncate_returns_prefix():
    c = CharRatioCounter()
    out = c.truncate("hello world", max_tokens=2)  # 8 chars
    assert out == "hello wo"
```

- [ ] **Step 2: Run — expect FAIL (`ModuleNotFoundError`).**

- [ ] **Step 3: Implement**

```python
# docgraph/ingest/tokenizer.py
from __future__ import annotations

import math
from typing import Protocol


class TokenCounter(Protocol):
    def count(self, text: str) -> int: ...
    def truncate(self, text: str, max_tokens: int) -> str: ...


class CharRatioCounter:
    """Fallback when no real tokenizer is available. ~chars/4 like LangChain's default."""

    def __init__(self, ratio: float = 4.0) -> None:
        self._ratio = ratio

    def count(self, text: str) -> int:
        if not text:
            return 0
        return math.ceil(len(text) / self._ratio)

    def truncate(self, text: str, max_tokens: int) -> str:
        return text[: int(max_tokens * self._ratio)]
```

- [ ] **Step 4: Run — expect PASS (2 passed).**

- [ ] **Step 5: Commit** — `feat(chunk-01): add TokenCounter protocol with char-ratio fallback`

---

## Task 2: HuggingFace tokenizer counter

**Files:**
- Modify: `docgraph/ingest/tokenizer.py`
- Test: `tests/ingest/test_tokenizer.py`

- [ ] **Step 1: Write failing test (skipped if `tokenizers` not installed)**

```python
import pytest

tokenizers = pytest.importorskip("tokenizers")

from docgraph.ingest.tokenizer import HFTokenizerCounter


def test_hf_counter_counts_real_tokens(tmp_path):
    # Use a tiny BPE tokenizer trained inline so the test is hermetic.
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers
    tok = Tokenizer(models.BPE(unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    trainer = trainers.BpeTrainer(special_tokens=["[UNK]"], vocab_size=50)
    tok.train_from_iterator(["hello world foo bar baz"], trainer)
    path = tmp_path / "tok.json"
    tok.save(str(path))

    counter = HFTokenizerCounter.from_file(str(path))
    assert counter.count("hello world") >= 1
    assert counter.count("") == 0
    assert counter.truncate("hello world foo bar baz", max_tokens=2).count(" ") <= 2
```

- [ ] **Step 2: Run — expect FAIL (`ImportError: HFTokenizerCounter`).**

- [ ] **Step 3: Implement**

```python
# append to docgraph/ingest/tokenizer.py
class HFTokenizerCounter:
    def __init__(self, tokenizer) -> None:
        self._tok = tokenizer

    @classmethod
    def from_file(cls, path: str) -> "HFTokenizerCounter":
        from tokenizers import Tokenizer
        return cls(Tokenizer.from_file(path))

    @classmethod
    def from_pretrained(cls, name: str) -> "HFTokenizerCounter":
        from tokenizers import Tokenizer
        return cls(Tokenizer.from_pretrained(name))

    def count(self, text: str) -> int:
        if not text:
            return 0
        return len(self._tok.encode(text).ids)

    def truncate(self, text: str, max_tokens: int) -> str:
        enc = self._tok.encode(text)
        if len(enc.ids) <= max_tokens:
            return text
        # offsets = list[(start, end)] in original string; take char-end of last kept token.
        end_char = enc.offsets[max_tokens - 1][1]
        return text[:end_char]
```

- [ ] **Step 4: Run — expect PASS.**

- [ ] **Step 5: Commit** — `feat(chunk-01): add HuggingFace tokenizer counter`

---

## Task 3: Factory `get_token_counter(cfg)` with auto-detect

**Files:**
- Modify: `docgraph/config.py`, `docgraph/ingest/tokenizer.py`
- Test: `tests/ingest/test_tokenizer.py`

- [ ] **Step 1: Add `tokenizer_source` to `Config`** (default `"auto"`).

- [ ] **Step 2: Write failing test**

```python
def test_factory_returns_char_ratio_when_unknown(monkeypatch):
    from docgraph.config import Config
    from docgraph.ingest.tokenizer import get_token_counter, CharRatioCounter

    cfg = Config(data_dir=...)  # use tmp_path
    cfg.tokenizer_source = "char-ratio"
    assert isinstance(get_token_counter(cfg), CharRatioCounter)
```

- [ ] **Step 3: Implement** (with `auto` mapping per provider):

```python
def get_token_counter(cfg) -> TokenCounter:
    src = (cfg.tokenizer_source or "auto").lower()
    if src == "char-ratio":
        return CharRatioCounter()
    if src == "auto":
        # Map provider+model -> a tokenizer choice.
        if cfg.embed_provider == "local":
            # Local Rust embedder downloads tokenizer.json next to the ONNX model.
            tok_path = cfg.local_model_dir / cfg.local_model / "tokenizer.json"
            if tok_path.exists():
                return HFTokenizerCounter.from_file(str(tok_path))
        if cfg.embed_provider == "openai":
            try:
                from docgraph.ingest.tokenizer import TiktokenCounter
                return TiktokenCounter(cfg.openai_model)
            except Exception:
                pass
        return CharRatioCounter()
    raise ValueError(f"unknown tokenizer_source: {src}")
```

- [ ] **Step 4: Run — expect PASS.**

- [ ] **Step 5: Commit** — `feat(chunk-01): wire token counter factory with auto-detect`

---

## Task 4: Optional `TiktokenCounter` for OpenAI

**Files:**
- Modify: `docgraph/ingest/tokenizer.py`
- Test: `tests/ingest/test_tokenizer.py`

- [ ] **Step 1: Failing test (skip if `tiktoken` not installed)**

```python
tiktoken = pytest.importorskip("tiktoken")

def test_tiktoken_counter_matches_known_token_count():
    from docgraph.ingest.tokenizer import TiktokenCounter
    c = TiktokenCounter("text-embedding-3-small")
    assert c.count("hello world") == 2
```

- [ ] **Step 2: Run — expect FAIL.**

- [ ] **Step 3: Implement**

```python
class TiktokenCounter:
    def __init__(self, model: str) -> None:
        import tiktoken
        try:
            self._enc = tiktoken.encoding_for_model(model)
        except KeyError:
            self._enc = tiktoken.get_encoding("cl100k_base")

    def count(self, text: str) -> int:
        return len(self._enc.encode(text))

    def truncate(self, text: str, max_tokens: int) -> str:
        ids = self._enc.encode(text)
        if len(ids) <= max_tokens:
            return text
        return self._enc.decode(ids[:max_tokens])
```

- [ ] **Step 4: Run — expect PASS.**

- [ ] **Step 5: Commit** — `feat(chunk-01): add tiktoken counter for OpenAI provider`

---

## Task 5: Re-base chunker on token budgets

**Files:**
- Modify: `docgraph/ingest/chunker.py`
- Test: `tests/ingest/test_chunker.py`

- [ ] **Step 1: Failing test — Vietnamese text never exceeds budget**

```python
def test_chunk_markdown_vi_respects_token_cap_with_real_tokenizer(tmp_path):
    from docgraph.ingest.chunker import chunk_markdown
    from docgraph.ingest.tokenizer import HFTokenizerCounter

    counter = HFTokenizerCounter.from_pretrained("intfloat/multilingual-e5-base")
    text = ("Đây là một đoạn văn tiếng Việt rất dài. " * 200)
    chunks = chunk_markdown(text, chunk_size=480, chunk_overlap=48, counter=counter)
    for c in chunks:
        assert counter.count(c) <= 480, f"chunk over budget: {counter.count(c)} tokens"
```

- [ ] **Step 2: Run — expect FAIL** (current chunker produces oversized chunks for VI).

- [ ] **Step 3: Implement**

  - Add a `counter: TokenCounter | None = None` parameter to `chunk_markdown`, `chunk_code`, and the internal `_chunk` helpers.
  - Replace `char_size = chunk_size * 4` with: when a counter is provided, work in token units; when not, fall back to the current char-budget behaviour for backward compat.
  - In `_split_recursive`, the size check becomes `counter.count(text) <= max_tokens` (cache counts per piece to avoid re-encoding in inner loops).
  - In `_merge_with_overlap`, replace `len(...)` comparisons with token counts; replace tail-overlap slicing `current[-overlap_chars:]` with `counter.truncate(reverse, ...)`-equivalent: keep the trailing N tokens by encoding once and decoding the suffix slice.

- [ ] **Step 4: Run — expect PASS** plus all existing tests still green.

- [ ] **Step 5: Commit** — `feat(chunk-01): chunker now budgets in tokens when counter is provided`

---

## Task 6: Indexer threads tokenizer + records `tokens` metadata

**Files:**
- Modify: `docgraph/ingest/indexer.py`
- Test: `tests/ingest/test_indexer.py`

- [ ] **Step 1: Failing test** — chunk metadata includes `tokens: int > 0`:

```python
async def test_index_markdown_records_token_count(indexer, doc_id):
    await indexer.index_markdown(doc_id, "# Title\n\n" + "word " * 800)
    chunks = chroma.get_by_doc_id(doc_id)
    assert all(c["metadata"]["tokens"] > 0 for c in chunks)
```

- [ ] **Step 2: Run — expect FAIL** (no `tokens` field today).

- [ ] **Step 3: Implement**

  - `Indexer.__init__` gains `counter: TokenCounter` (built once via `get_token_counter(cfg)` in the app bootstrap).
  - Pass `counter` into `chunk_markdown` / `chunk_code` calls.
  - Populate `metadata["tokens"] = counter.count(text)` per chunk.
  - In `app/bootstrap` (search service entry), build the counter once and inject.

- [ ] **Step 4: Run — expect PASS.**

- [ ] **Step 5: Commit** — `feat(chunk-01): record tokens metadata on every chunk`

---

## Task 7: CLI `docgraph reindex --all` + diagnostic

**Files:**
- Modify: `docgraph/cli.py`
- Test: `tests/test_cli.py`

- [ ] **Step 1: Failing test** — `docgraph reindex --all --dry-run` prints planned doc IDs.

- [ ] **Step 2: Run — expect FAIL.**

- [ ] **Step 3: Implement**

  - Add `reindex` subcommand: with `--all`, iterate all docs whose status is `READY` and call `Indexer.reindex_document` (sequentially to avoid embedder thrash).
  - Add `--dry-run` for confirmation.
  - Add `docgraph stats chunks` (small): print min/median/p95 tokens grouped by doc.

- [ ] **Step 4: Run — expect PASS.**

- [ ] **Step 5: Commit** — `feat(chunk-01): add reindex --all and chunk stats CLI`

---

## Task 8: Documentation + README

- [ ] Update `README.md` "Configuration" table with `DOCGRAPH_TOKENIZER_SOURCE`.
- [ ] Note in the migration section that re-indexing is required to benefit from accurate token budgets.
- [ ] Commit — `docs(chunk-01): document tokenizer config and re-index flow`

---

## Acceptance Criteria

- [ ] No chunk produced from any markdown input exceeds the configured `chunk_size` measured by the embedder's tokenizer.
- [ ] `tokens` metadata is present on every newly-written chunk.
- [ ] Existing tests in `tests/ingest/test_chunker.py` still pass with the default char-ratio counter.
- [ ] `docgraph reindex --all` migrates a corpus end-to-end without errors.
- [ ] `docgraph stats chunks` reports realistic distributions (e.g. p95 ≤ chunk_size).
