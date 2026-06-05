from __future__ import annotations

import re

# Token characters preserved: alphanumeric, underscore, dot, hyphen.
# Anything else (operators, quotes, punctuation) becomes a space.
_TOKEN_KEEP = re.compile(r"[^\w\s_.-]", flags=re.UNICODE)


def _sanitize_query(text: str) -> str:
    """Convert raw user input to a safe FTS5 MATCH expression.

    Wraps each token in double quotes so FTS5 treats it as a phrase literal.
    This neutralizes operators (AND, OR, NOT, NEAR, *, +, etc.) and special
    chars. Empty / whitespace / fully-stripped input returns "".
    """
    if not text:
        return ""
    cleaned = _TOKEN_KEEP.sub(" ", text)
    tokens = [t for t in cleaned.split() if t]
    if not tokens:
        return ""
    return " ".join(f'"{t}"' for t in tokens)
