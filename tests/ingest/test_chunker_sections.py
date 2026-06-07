from docgraph.ingest.chunker import chunk_markdown_sections


def test_chunk_markdown_sections_prepends_breadcrumb():
    md = "# Top\n\n## Sub\n\n" + "word " * 800
    chunks = chunk_markdown_sections(md, chunk_size=128, chunk_overlap=16)
    assert chunks
    for c in chunks:
        assert c.text.startswith("Top > Sub\n\n"), c.text[:50]
        assert c.heading_path == ["Top", "Sub"]


def test_chunker_keeps_code_fence_intact():
    fence = "```python\n" + ("x = 1\n" * 10) + "```"
    md = "# H\n\nbefore\n\n" + fence + "\n\nafter"
    chunks = chunk_markdown_sections(md, chunk_size=512, chunk_overlap=16)
    matched = [c for c in chunks if "```python" in c.text]
    assert any(
        c.text.count("```") >= 2 for c in matched
    ), "code fence was split across chunks"
