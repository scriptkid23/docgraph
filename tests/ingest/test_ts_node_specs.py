from docgraph.ingest.ts_node_specs import split_node_types


def test_split_node_types_python():
    s = split_node_types("python")
    assert "function_definition" in s
    assert "class_definition" in s


def test_split_node_types_unknown_returns_empty():
    assert split_node_types("brainfuck") == frozenset()
