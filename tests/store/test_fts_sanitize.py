from __future__ import annotations

import pytest

from docgraph.store.fts import _sanitize_query


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("máy tính", '"máy" "tính"'),
        ("embed_query", '"embed_query"'),
        ("foo*bar", '"foo" "bar"'),
        ('say "hello"', '"say" "hello"'),
        ("AND OR NOT", '"AND" "OR" "NOT"'),
        ("   ", ""),
        ("", ""),
        ("(foo)", '"foo"'),
        ("v1.5", '"v1.5"'),
        ("a-b", '"a-b"'),
        ("nomic-embed-text", '"nomic-embed-text"'),
        ("***", ""),
        ("query: with colon", '"query" "with" "colon"'),
    ],
)
def test_sanitize(raw, expected):
    assert _sanitize_query(raw) == expected
