from docgraph.embed.prefix import apply_embed_prefix, uses_e5_prefixes


def test_uses_e5_prefixes():
    assert uses_e5_prefixes("multilingual-e5-base")
    assert uses_e5_prefixes("MultilingualE5Large")
    assert not uses_e5_prefixes("nomic-embed-text")


def test_apply_embed_prefix_passage():
    out = apply_embed_prefix(["Hello", "World"], "multilingual-e5-base", for_query=False)
    assert out == ["passage: Hello", "passage: World"]


def test_apply_embed_prefix_query():
    out = apply_embed_prefix(["xin chào"], "multilingual-e5-base", for_query=True)
    assert out == ["query: xin chào"]


def test_apply_embed_prefix_skips_existing():
    out = apply_embed_prefix(["query: already"], "multilingual-e5-base", for_query=False)
    assert out == ["query: already"]


def test_apply_embed_prefix_nomic_unchanged():
    texts = ["plain text"]
    assert apply_embed_prefix(texts, "nomic-embed-text", for_query=True) == texts
