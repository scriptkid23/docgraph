from __future__ import annotations

import pytest

from docgraph.store.fts import _sanitize_query


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("máy tính", '"máy" OR "tính"'),
        ("embed_query", '"embed_query"'),
        ("foo*bar", '"foo" OR "bar"'),
        ('say "hello"', '"say" OR "hello"'),
        ("AND OR NOT", '"AND" OR "OR" OR "NOT"'),
        ("   ", ""),
        ("", ""),
        ("(foo)", '"foo"'),
        ("v1.5", '"v1.5"'),
        ("a-b", '"a-b"'),
        ("nomic-embed-text", '"nomic-embed-text"'),
        ("***", ""),
        ("query: with colon", '"query" OR "with" OR "colon"'),
    ],
)
def test_sanitize(raw, expected):
    assert _sanitize_query(raw) == expected
