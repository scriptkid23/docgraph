from docgraph.ingest.lang_dispatch import dispatch_chunker
from docgraph.ingest.chunker import chunk_code, chunk_markdown_sections
from docgraph.ingest.structured_chunker import chunk_json, chunk_yaml


def test_dispatch_returns_markdown_sections_for_md():
    assert dispatch_chunker("markdown") is chunk_markdown_sections


def test_dispatch_returns_structured_for_json_yaml():
    assert dispatch_chunker("json") is chunk_json
    assert dispatch_chunker("yaml") is chunk_yaml


def test_dispatch_returns_code_for_known_languages():
    for lang in ("python", "go", "rust", "java"):
        assert dispatch_chunker(lang) is chunk_code


def test_dispatch_returns_code_for_unknown():
    assert dispatch_chunker(None) is chunk_code
    assert dispatch_chunker("brainfuck") is chunk_code
