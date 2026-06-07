# Plan chunk-03 — Atomic Code-Fence and Table Preservation

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Never split a fenced code block (```` ``` ```` … ```` ``` ````, `~~~`) or a markdown pipe-table mid-row. When a single atomic block exceeds the chunk budget, either keep it whole (allow overflow with a warning) or split *along block-internal boundaries* (lines for code, rows for tables — with the header row replicated into each part).

**Why:** A half-fence chunk renders broken markdown, mis-tokenizes (the embedder sees lopsided syntax), and starves the LLM of code context. A half-table chunk loses the header row, so the data rows become semantic noise. This is fingerprint #2 of "amateur RAG" in 2026 best-practice surveys.

**Architecture:** Add a pre-pass that segments markdown into a sequence of **atomic** and **flow** spans, each tagged with a type. The recursive chunker runs only on flow spans; atomic spans pass through untouched (or split internally if they bust the budget alone). The pre-pass plugs into the section walker from chunk-02.

**Depends on:** chunk-01 (token budget), chunk-02 (section walker).

**Spec:** `docs/superpowers/specs/2026-06-03-chunker-improvements-design.md` §3.

---

## File Structure

- **Create** `docgraph/ingest/markdown_blocks.py` — `Span` dataclass + `segment_blocks()` returning `list[Span]`.
- **Modify** `docgraph/ingest/markdown_sections.py` — section body becomes `list[Span]` instead of raw string.
- **Modify** `docgraph/ingest/chunker.py` — chunker iterates spans, treats atomic ones as inseparable.
- **Modify** `docgraph/config.py` — `atomic_blocks_enabled: bool = True`.
- **Tests:** `tests/ingest/test_markdown_blocks.py`, extend `tests/ingest/test_chunker.py`.

---

## Task 1: Span segmenter

**Files:**
- Create: `docgraph/ingest/markdown_blocks.py`
- Test: `tests/ingest/test_markdown_blocks.py`

- [ ] **Step 1: Failing test**

```python
# tests/ingest/test_markdown_blocks.py
from docgraph.ingest.markdown_blocks import segment_blocks, SpanKind


def test_segment_keeps_code_fence_atomic():
    md = """intro paragraph

```python
def f():
    return 1
```

trailing paragraph
"""
    spans = segment_blocks(md)
    kinds = [s.kind for s in spans]
    assert SpanKind.CODE_FENCE in kinds
    fence = next(s for s in spans if s.kind == SpanKind.CODE_FENCE)
    assert "def f()" in fence.text
    assert fence.text.startswith("```") and fence.text.rstrip().endswith("```")


def test_segment_keeps_pipe_table_atomic_with_header():
    md = """text before

| col A | col B |
|-------|-------|
| 1     | 2     |
| 3     | 4     |

text after
"""
    spans = segment_blocks(md)
    table = next(s for s in spans if s.kind == SpanKind.TABLE)
    assert table.header_row.startswith("| col A")
    assert "| 1     | 2     |" in table.text


def test_segment_returns_flow_for_plain_text():
    md = "Just paragraphs.\n\nNothing special here."
    spans = segment_blocks(md)
    assert len(spans) == 1
    assert spans[0].kind == SpanKind.FLOW
```

- [ ] **Step 2: Run — expect FAIL.**

- [ ] **Step 3: Implement**

```python
# docgraph/ingest/markdown_blocks.py
from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

_FENCE_OPEN = re.compile(r"^(```|~~~)(.*)$")
_TABLE_DIVIDER = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$")


class SpanKind(str, Enum):
    FLOW = "flow"
    CODE_FENCE = "code_fence"
    TABLE = "table"


@dataclass
class Span:
    kind: SpanKind
    text: str
    header_row: str = ""  # populated only for TABLE spans


def segment_blocks(md: str) -> list[Span]:
    """Segment markdown into atomic (code/table) and flow spans."""
    lines = md.splitlines()
    out: list[Span] = []
    buf: list[str] = []

    def flush_flow():
        if buf:
            text = "\n".join(buf).strip("\n")
            if text:
                out.append(Span(SpanKind.FLOW, text))
            buf.clear()

    i = 0
    while i < len(lines):
        line = lines[i]
        m = _FENCE_OPEN.match(line)
        if m:
            flush_flow()
            fence = m.group(1)
            block = [line]
            i += 1
            while i < len(lines):
                block.append(lines[i])
                if lines[i].strip().startswith(fence):
                    i += 1
                    break
                i += 1
            out.append(Span(SpanKind.CODE_FENCE, "\n".join(block)))
            continue

        # Pipe-table: a "header" line followed by a divider line of dashes/pipes.
        if "|" in line and i + 1 < len(lines) and _TABLE_DIVIDER.match(lines[i + 1]):
            flush_flow()
            header = line
            block = [header, lines[i + 1]]
            i += 2
            while i < len(lines) and lines[i].strip().startswith("|"):
                block.append(lines[i])
                i += 1
            out.append(Span(SpanKind.TABLE, "\n".join(block), header_row=header))
            continue

        buf.append(line)
        i += 1

    flush_flow()
    return out
```

- [ ] **Step 4: Run — expect PASS.**

- [ ] **Step 5: Commit** — `feat(chunk-03): atomic span segmenter for code fences and tables`

---

## Task 2: Wire spans into section walker

**Files:**
- Modify: `docgraph/ingest/markdown_sections.py`
- Test: extend `tests/ingest/test_markdown_sections.py`

- [ ] **Step 1: Failing test** — `Section.spans` is a list of typed `Span` objects.

- [ ] **Step 2: Run — expect FAIL.**

- [ ] **Step 3: Implement**
  - Section dataclass becomes `Section(heading_path, spans: list[Span])`.
  - After collecting body lines, run `segment_blocks` on the joined body string to produce `spans`.
  - Keep `body` as a derived `@property` returning the joined text — preserves backward compat with chunk-02 callers.

- [ ] **Step 4: Run — expect PASS.**

- [ ] **Step 5: Commit** — `feat(chunk-03): section walker tags spans by kind`

---

## Task 3: Chunker treats atomic spans as inseparable

**Files:**
- Modify: `docgraph/ingest/chunker.py`
- Test: extend `tests/ingest/test_chunker.py`

- [ ] **Step 1: Failing test**

```python
def test_chunker_keeps_code_fence_intact():
    from docgraph.ingest.chunker import chunk_markdown_sections

    fence = "```python\n" + ("x = 1\n" * 10) + "```"
    md = "# H\n\nbefore\n\n" + fence + "\n\nafter"
    chunks = chunk_markdown_sections(md, chunk_size=128, chunk_overlap=16)
    matched = [c for c in chunks if "```python" in c.text]
    assert any("```" in c.text and c.text.count("```") >= 2 for c in matched), \
        "code fence was split across chunks"


def test_chunker_table_preserves_header_when_split():
    from docgraph.ingest.chunker import chunk_markdown_sections

    rows = "\n".join(f"| {i} | {i*i} |" for i in range(200))
    md = "# T\n\n| n | n*n |\n|---|---|\n" + rows
    chunks = chunk_markdown_sections(md, chunk_size=64, chunk_overlap=8)
    table_chunks = [c for c in chunks if "| n | n*n |" in c.text]
    # Either the table fit in one chunk (rare) or every chunk that contains
    # rows must also contain the header row.
    row_chunks = [c for c in chunks if "|" in c.text and "n*n" not in c.text and "---" not in c.text]
    if row_chunks:
        for c in row_chunks:
            assert "| n | n*n |" in c.text, "row chunk lost its table header"
```

- [ ] **Step 2: Run — expect FAIL.**

- [ ] **Step 3: Implement**

  - The per-section chunker becomes a "span emitter":
    - Maintain a current chunk buffer (list of strings).
    - For each span:
      - **FLOW** span: feed into the existing recursive splitter; emit pieces into the buffer with the merge-with-overlap logic.
      - **CODE_FENCE** span: if it fits the remaining budget, append to buffer. Otherwise flush the buffer, then either:
        - emit the fence as one oversize chunk (log a warning with `tokens > budget`), OR
        - if the fence's token count > 2 × budget, split by lines (never mid-line); each piece must keep the opening ```` ``` ```` and append a closing ```` ``` ```` so it remains valid markdown. Add `partial=True` metadata.
      - **TABLE** span: if it fits, append. Otherwise flush, then split into row groups (`segment_blocks` already kept the header row). Each row-group chunk gets the original `header_row + divider_row` prepended. Mark `partial=True`.
  - Carry the breadcrumb prefix from chunk-02 into each emitted chunk.

- [ ] **Step 4: Run — expect PASS** plus existing tests stay green.

- [ ] **Step 5: Commit** — `feat(chunk-03): atomic fence/table handling in chunker`

---

## Task 4: Telemetry for oversize atomic blocks

**Files:**
- Modify: `docgraph/ingest/indexer.py`
- Test: extend `tests/ingest/test_indexer.py`

- [ ] **Step 1: Failing test** — when a chunk's `tokens > chunk_size`, indexer writes `metadata["overflow"] = True` and logs at WARNING.

- [ ] **Step 2: Run — expect FAIL.**

- [ ] **Step 3: Implement**
  - Each chunk emitted from the chunker carries a `meta` dict including `partial: bool` and `overflow: bool`.
  - Indexer copies these into Chroma metadata.
  - On overflow: `logger.warning("chunk %s overflows: %d tokens > %d budget", chunk_id, tokens, budget)`.

- [ ] **Step 4: Run — expect PASS.**

- [ ] **Step 5: Commit** — `feat(chunk-03): record partial/overflow flags on atomic-block chunks`

---

## Task 5: Disable flag for benchmarking

**Files:**
- Modify: `docgraph/config.py`, `docgraph/ingest/chunker.py`

- [ ] Add `atomic_blocks_enabled: bool = True` and a `DOCGRAPH_ATOMIC_BLOCKS=false` env override so benchmarks can A/B against the old behaviour.
- [ ] When false, fall back to the chunk-02 flow-only path.
- [ ] Commit — `feat(chunk-03): config flag to disable atomic-block preservation`

---

## Acceptance Criteria

- [ ] No chunk produced from any markdown input contains an unbalanced ```` ``` ```` (open without close, or vice versa).
- [ ] Every chunk that contains pipe-table data rows also contains the table's header row.
- [ ] When an atomic block exceeds the budget, the indexer writes `overflow=True` AND logs a WARNING.
- [ ] All existing chunker tests still pass.
