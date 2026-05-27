import pytest

from docgraph.ingest.urls import parse_url_lines, url_display_name, validate_url


def test_parse_url_lines_skips_blanks_and_comments():
    text = """
    https://example.com/a

    # comment
    https://example.com/b
    https://example.com/a
    """
    assert parse_url_lines(text) == [
        "https://example.com/a",
        "https://example.com/b",
    ]


def test_url_display_name():
    name = url_display_name("https://docs.python.org/3/tutorial/")
    assert "docs.python.org" in name
    assert "tutorial" in name


def test_validate_url_accepts_https():
    assert validate_url("https://example.com/page") == "https://example.com/page"


def test_validate_url_rejects_localhost():
    with pytest.raises(ValueError, match="blocked"):
        validate_url("http://localhost/admin")


def test_validate_url_rejects_private_ip():
    with pytest.raises(ValueError, match="blocked"):
        validate_url("http://192.168.1.1/internal")


def test_validate_url_rejects_bad_scheme():
    with pytest.raises(ValueError, match="scheme"):
        validate_url("ftp://example.com")
