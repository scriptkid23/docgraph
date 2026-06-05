from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

_FENCE_OPEN = re.compile(r"^(```|~~~)(.*)$")
_TABLE_DIVIDER = re.compile(
    r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$"
)


class SpanKind(str, Enum):
    FLOW = "flow"
    CODE_FENCE = "code_fence"
    TABLE = "table"


@dataclass
class Span:
    kind: SpanKind
    text: str
    header_row: str = ""


def segment_blocks(md: str) -> list[Span]:
    """Segment markdown into atomic (code/table) and flow spans."""
    lines = md.splitlines()
    out: list[Span] = []
    buf: list[str] = []

    def flush_flow() -> None:
        if buf:
            text = "\n".join(buf).strip("\n")
            if text:
                out.append(Span(SpanKind.FLOW, text))
            buf.clear()

    i = 0
    while i < len(lines):
        line = lines[i]
        m = _FENCE_OPEN.match(line)
        if m:
            flush_flow()
            fence = m.group(1)
            block = [line]
            i += 1
            while i < len(lines):
                block.append(lines[i])
                if lines[i].strip().startswith(fence):
                    i += 1
                    break
                i += 1
            out.append(Span(SpanKind.CODE_FENCE, "\n".join(block)))
            continue

        if "|" in line and i + 1 < len(lines) and _TABLE_DIVIDER.match(lines[i + 1]):
            flush_flow()
            header = line
            block = [header, lines[i + 1]]
            i += 2
            while i < len(lines) and lines[i].strip().startswith("|"):
                block.append(lines[i])
                i += 1
            out.append(Span(SpanKind.TABLE, "\n".join(block), header_row=header))
            continue

        buf.append(line)
        i += 1

    flush_flow()
    return out
