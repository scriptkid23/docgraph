from docgraph.ingest.markdown_blocks import SpanKind, segment_blocks


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
