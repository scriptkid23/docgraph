from docgraph.ingest.chunker import chunk_markdown


def test_chunk_markdown_splits_with_overlap():
    text = "word " * 600
    chunks = chunk_markdown(text, chunk_size=512, chunk_overlap=64)
    assert len(chunks) >= 2
    assert all(isinstance(c, str) and len(c) > 0 for c in chunks)


def test_chunk_markdown_preserves_short_text():
    text = "# Hello\n\nShort doc."
    chunks = chunk_markdown(text, chunk_size=512, chunk_overlap=64)
    assert chunks == ["# Hello\n\nShort doc."]
