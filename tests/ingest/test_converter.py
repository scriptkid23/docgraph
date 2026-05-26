from pathlib import Path

from boostmcp.ingest.converter import convert_file_to_markdown

FIXTURES = Path(__file__).parent.parent / "fixtures"


def test_convert_md_file():
    md = convert_file_to_markdown(FIXTURES / "sample.md")
    assert "# Sample" in md
    assert "test document" in md
