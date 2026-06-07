from docgraph.ingest.code_dump import (
    detect_repomix,
    infer_language,
    parse_repomix,
)


def test_infer_language_known_extensions():
    assert infer_language("src/main.py") == "python"
    assert infer_language("app/index.ts") == "typescript"
    assert infer_language("pkg/server.go") == "go"


def test_infer_language_unknown_returns_none():
    assert infer_language("data.bin") is None
    assert infer_language("Makefile") is None


def test_detect_repomix_plain():
    text = "================\nFile: src/a.py\n================\nprint(1)\n"
    assert detect_repomix(text) is True


def test_detect_repomix_xml():
    text = '<file path="src/a.py">\nprint(1)\n</file>'
    assert detect_repomix(text) is True


def test_detect_repomix_false_on_prose():
    text = "# My Book\n\nChapter one. Some ordinary prose here.\n"
    assert detect_repomix(text) is False


def test_parse_repomix_plain():
    text = (
        "================\nFile: src/a.py\n================\n"
        "def a():\n    return 1\n\n"
        "================\nFile: src/b.py\n================\n"
        "def b():\n    return 2\n"
    )
    files = parse_repomix(text)
    assert [p for p, _ in files] == ["src/a.py", "src/b.py"]
    assert "def a():" in files[0][1]
    assert "def b():" in files[1][1]


def test_parse_repomix_xml():
    text = (
        '<file path="src/a.py">\ndef a():\n    return 1\n</file>\n'
        '<file path="src/b.ts">\nexport const b = 2\n</file>\n'
    )
    files = parse_repomix(text)
    assert [p for p, _ in files] == ["src/a.py", "src/b.ts"]
    assert "def a():" in files[0][1]


def test_parse_repomix_skips_empty_sections():
    text = (
        "================\nFile: empty.py\n================\n\n"
        "================\nFile: real.py\n================\nx = 1\n"
    )
    files = parse_repomix(text)
    assert [p for p, _ in files] == ["real.py"]
