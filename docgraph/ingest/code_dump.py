from __future__ import annotations

import re

_LANGUAGE_BY_EXT = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".rb": "ruby",
    ".php": "php",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".hpp": "cpp",
    ".cs": "csharp",
    ".sh": "shell",
    ".sql": "sql",
    ".md": "markdown",
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".html": "html",
    ".css": "css",
}

# Repomix plain-style per-file header:
#   ================
#   File: path/to/file.ext
#   ================
# The '=' rule length varies by Repomix version, so match 3 or more.
_PLAIN_HEADER = re.compile(
    r"^={3,}[ \t]*\nFile: (?P<path>.+?)[ \t]*\n={3,}[ \t]*$",
    re.MULTILINE,
)

# Repomix XML-style block: <file path="...">content</file>.
# Repomix does NOT escape file content, so scan with regex, not an XML parser.
_XML_FILE = re.compile(
    r'<file path="(?P<path>[^"]+)">\n?(?P<body>.*?)\n?</file>',
    re.DOTALL,
)


def infer_language(path: str) -> str | None:
    """Map a file path's extension to a language name, or None if unknown."""
    dot = path.rfind(".")
    if dot == -1:
        return None
    return _LANGUAGE_BY_EXT.get(path[dot:].lower())


def detect_repomix(text: str) -> bool:
    """True if text looks like a Repomix dump (plain or XML style)."""
    return bool(_XML_FILE.search(text) or _PLAIN_HEADER.search(text))


def parse_repomix(text: str) -> list[tuple[str, str]]:
    """Split a Repomix dump into (file_path, content) pairs.

    Tries XML style first, then plain style. Empty file sections are skipped.
    """
    xml_matches = list(_XML_FILE.finditer(text))
    if xml_matches:
        out: list[tuple[str, str]] = []
        for m in xml_matches:
            body = m.group("body").strip("\n")
            if body.strip():
                out.append((m.group("path").strip(), body))
        return out

    headers = list(_PLAIN_HEADER.finditer(text))
    out = []
    for i, m in enumerate(headers):
        start = m.end()
        end = headers[i + 1].start() if i + 1 < len(headers) else len(text)
        body = text[start:end].strip("\n")
        if body.strip():
            out.append((m.group("path").strip(), body))
    return out
