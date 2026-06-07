from docgraph.ingest.markdown_sections import walk_sections


def test_walk_sections_emits_heading_path_per_section():
    md = """# Title
intro

## A
body of A

### A.1
body of A.1

## B
body of B
"""
    sections = list(walk_sections(md))
    paths = [s.heading_path for s in sections]
    assert paths == [
        ["Title"],
        ["Title", "A"],
        ["Title", "A", "A.1"],
        ["Title", "B"],
    ]
    assert sections[2].body.strip() == "body of A.1"


def test_walk_sections_skips_atx_inside_code_fence():
    md = """# Title
text

```python
# this is not a heading
print("hi")
```

## Real Heading
body
"""
    sections = list(walk_sections(md))
    headings = [s.heading_path[-1] for s in sections]
    assert headings == ["Title", "Real Heading"]


def test_walk_sections_handles_no_heading_doc():
    md = "Just a paragraph with no heading at all.\n\nAnother paragraph."
    sections = list(walk_sections(md))
    assert len(sections) == 1
    assert sections[0].heading_path == []
    assert "Just a paragraph" in sections[0].body
