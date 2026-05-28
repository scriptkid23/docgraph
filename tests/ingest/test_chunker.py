from docgraph.ingest.chunker import chunk_code, chunk_markdown


def test_chunk_markdown_splits_with_overlap():
    text = "word " * 600
    chunks = chunk_markdown(text, chunk_size=512, chunk_overlap=64)
    assert len(chunks) >= 2
    assert all(isinstance(c, str) and len(c) > 0 for c in chunks)


def test_chunk_markdown_preserves_short_text():
    text = "# Hello\n\nShort doc."
    chunks = chunk_markdown(text, chunk_size=512, chunk_overlap=64)
    assert chunks == ["# Hello\n\nShort doc."]


def test_chunk_markdown_handles_oversized_heading_section():
    # Regression: a single H2 section larger than the char budget used to cause
    # infinite recursion because the separator was re-attached to the recursive
    # call, which then re-split on the same separator forever.
    body = "word " * 50_000  # ~250k chars, no inner H2
    text = f"## Chapter 1\n\n{body}"
    chunks = chunk_markdown(text, chunk_size=512, chunk_overlap=64)
    assert len(chunks) > 1
    assert all(isinstance(c, str) and c for c in chunks)
    assert "Chapter 1" in chunks[0]


def test_chunk_markdown_handles_large_multi_chapter_book():
    # ~1000-page book shape: many H2 chapters, each chapter much larger than the
    # per-chunk char budget. Must not hit Python's recursion limit.
    chapter_body = "paragraph text. " * 3_000  # ~48k chars per chapter
    text = "\n\n".join(
        f"## Chapter {i}\n\n{chapter_body}" for i in range(1, 51)
    )
    chunks = chunk_markdown(text, chunk_size=512, chunk_overlap=64)
    assert len(chunks) > 50
    assert all(c.strip() for c in chunks)


def test_chunk_code_keeps_small_function_intact():
    code = "def foo():\n    return 1\n"
    chunks = chunk_code(code, chunk_size=512, chunk_overlap=64)
    assert chunks == ["def foo():\n    return 1"]


def test_chunk_code_splits_large_file_without_error():
    code = "def big():\n" + "    x = 1\n" * 20_000
    chunks = chunk_code(code, chunk_size=128, chunk_overlap=16)
    assert len(chunks) > 1
    assert all(c.strip() for c in chunks)


def test_chunk_code_prefers_def_boundaries():
    def fn(n: int) -> str:
        return f"def f{n}():\n" + "    pass\n" * 16

    code = fn(1) + fn(2)
    chunks = chunk_code(code, chunk_size=64, chunk_overlap=0)
    assert len(chunks) == 2
    assert chunks[0].startswith("def f1")
    assert chunks[1].startswith("def f2")
