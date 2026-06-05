from docgraph.ingest.tokenizer import CharRatioCounter


def test_char_ratio_counter_uses_chars_over_4():
    c = CharRatioCounter()
    assert c.count("hello world") == 3
    assert c.count("") == 0


def test_char_ratio_counter_truncate_returns_prefix():
    c = CharRatioCounter()
    out = c.truncate("hello world", max_tokens=2)
    assert out == "hello wo"


def test_char_ratio_counter_truncate_tail():
    c = CharRatioCounter()
    out = c.truncate_tail("hello world", max_tokens=2)
    assert out == "lo world"
