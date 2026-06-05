from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterator

from docgraph.ingest.markdown_blocks import Span, segment_blocks

_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_FENCE = re.compile(r"^(```|~~~)")


@dataclass
class Section:
    heading_path: list[str]
    body: str
    spans: list[Span] | None = None

    def __post_init__(self) -> None:
        if self.spans is None:
            self.spans = segment_blocks(self.body)


def walk_sections(md: str) -> Iterator[Section]:
    """Yield Section tuples honoring the heading hierarchy."""
    stack: list[str] = []
    pending: list[str] = []
    in_fence = False

    def flush() -> Iterator[Section]:
        if pending:
            body = "\n".join(pending).strip("\n")
            if body.strip():
                yield Section(heading_path=list(stack), body=body)
            pending.clear()

    for line in md.splitlines():
        if _FENCE.match(line):
            in_fence = not in_fence
            pending.append(line)
            continue
        if in_fence:
            pending.append(line)
            continue
        m = _HEADING.match(line)
        if m:
            yield from flush()
            level = len(m.group(1))
            title = m.group(2).strip()
            stack = stack[: level - 1]
            stack.append(title)
            continue
        pending.append(line)
    yield from flush()
