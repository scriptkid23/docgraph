from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class DocumentStatus(str, Enum):
    PROCESSING = "processing"
    READY = "ready"
    ERROR = "error"


class SourceType(str, Enum):
    FILE = "file"
    URL = "url"


@dataclass
class DocumentRecord:
    id: str
    filename: str
    folder: str = ""
    tags: list[str] = field(default_factory=list)
    status: DocumentStatus = DocumentStatus.PROCESSING
    chunk_count: int = 0
    progress_pct: int = 0
    progress_phase: str = ""
    error_message: Optional[str] = None
    original_path: str = ""
    markdown_path: str = ""
    source_type: SourceType = SourceType.FILE
    source_url: str = ""


@dataclass
class ChunkRecord:
    id: str
    doc_id: str
    text: str
    chunk_index: int
    filename: str
    folder: str
    tags: list[str]
    source_page: Optional[int] = None
    file_path: Optional[str] = None


@dataclass
class SearchResult:
    text: str
    doc_id: str
    filename: str
    folder: str
    tags: list[str]
    chunk_index: int
    score: float
    source_page: Optional[int] = None
    file_path: Optional[str] = None
