# Plan v2-01 — Foundation & Migration

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bootstrap Python/Poetry project, domain types, config loading, and remove Go v1 code.

**Architecture:** Single Poetry package `boostmcp`. Config in `boostmcp/config.py` reads env vars then `~/.boostmcp/config.yaml`. Domain types in `boostmcp/models.py`. Archive v1 Go code and docs before adding Python.

**Tech Stack:** Python 3.10+, Poetry, PyYAML, pytest

**Depends on:** nothing  
**Blocks:** Plans v2-02 through v2-07

**Spec refs:** §5.1 Project Structure, §6 Configuration, §11 Migration

---

## File Structure (after this plan)

```
boostmcp/
├── pyproject.toml
├── poetry.lock
├── boostmcp/
│   ├── __init__.py
│   ├── __main__.py
│   ├── cli.py              # stub: prints "use serve" until Plan 06
│   ├── config.py
│   └── models.py
├── tests/
│   ├── conftest.py
│   ├── test_config.py
│   └── test_models.py
└── docs/archive/v1/        # moved v1 Go docs
```

---

### Task 1: Archive v1 and remove Go code

**Files:**
- Move: `docs/superpowers/specs/2026-05-26-boostmcp-v1-design.md` → `docs/archive/v1/`
- Move: `docs/superpowers/plans/2026-05-26-*` (v1 plans + index) → `docs/archive/v1/plans/`
- Delete: `internal/`, `cmd/`, `pkg/`, `go.mod`, `go.sum`, `scripts/call_generate.go` (if present)

- [ ] **Step 1: Create archive directories**

```bash
mkdir -p docs/archive/v1/plans
git mv docs/superpowers/specs/2026-05-26-boostmcp-v1-design.md docs/archive/v1/
git mv docs/superpowers/plans/2026-05-26-boostmcp-v1-index.md docs/archive/v1/plans/
git mv docs/superpowers/plans/2026-05-26-plan-01-foundation.md docs/archive/v1/plans/
git mv docs/superpowers/plans/2026-05-26-plan-02-inference.md docs/archive/v1/plans/
git mv docs/superpowers/plans/2026-05-26-plan-03-generator.md docs/archive/v1/plans/
git mv docs/superpowers/plans/2026-05-26-plan-04-narrower.md docs/archive/v1/plans/
git mv docs/superpowers/plans/2026-05-26-plan-05-mcp-server.md docs/archive/v1/plans/
git mv docs/superpowers/plans/2026-05-26-plan-06-e2e-docs.md docs/archive/v1/plans/
```

- [ ] **Step 2: Remove Go source**

```bash
git rm -r internal cmd pkg go.mod go.sum
# if untracked: rm -rf internal cmd pkg go.mod go.sum
```

- [ ] **Step 3: Commit**

```bash
git add docs/archive/v1
git commit -m "chore: archive v1 Go code and docs for Python v2 rewrite"
```

---

### Task 2: Initialize Poetry project

**Files:**
- Create: `pyproject.toml`
- Create: `boostmcp/__init__.py`
- Create: `boostmcp/__main__.py`
- Create: `boostmcp/cli.py`
- Create: `tests/conftest.py`

- [ ] **Step 1: Create pyproject.toml**

```toml
[tool.poetry]
name = "boostmcp"
version = "2.0.0"
description = "Local document RAG server for Cursor via MCP"
authors = ["BoostMCP Contributors"]
readme = "README.md"
packages = [{ include = "boostmcp" }]

[tool.poetry.dependencies]
python = "^3.10"
pyyaml = "^6.0"

[tool.poetry.group.dev.dependencies]
pytest = "^8.0"
pytest-asyncio = "^0.25"

[tool.poetry.scripts]
boostmcp = "boostmcp.cli:main"

[build-system]
requires = ["poetry-core"]
build-backend = "poetry.core.masonry.api"

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
markers = [
    "integration: tests requiring Ollama or external services",
]
```

- [ ] **Step 2: Create package stubs**

```python
# boostmcp/__init__.py
__version__ = "2.0.0"
```

```python
# boostmcp/cli.py
import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser(prog="boostmcp")
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("serve", help="Start MCP + Web UI server")
    args = parser.parse_args()
    if args.command == "serve":
        print("boostmcp serve not implemented yet", file=sys.stderr)
        sys.exit(1)
    parser.print_help()
    sys.exit(1)


if __name__ == "__main__":
    main()
```

```python
# boostmcp/__main__.py
from boostmcp.cli import main

main()
```

```python
# tests/conftest.py
import pytest


@pytest.fixture
def tmp_data_dir(tmp_path):
    return tmp_path / "boostmcp_data"
```

- [ ] **Step 3: Install and verify**

```bash
poetry install
poetry run boostmcp --help
```

Expected: help text with `serve` subcommand

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml poetry.lock boostmcp/ tests/
git commit -m "chore: initialize Poetry project for BoostMCP v2"
```

---

### Task 3: Domain models

**Files:**
- Create: `boostmcp/models.py`
- Create: `tests/test_models.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_models.py
from boostmcp.models import DocumentRecord, DocumentStatus, SearchResult


def test_document_record_defaults():
    doc = DocumentRecord(
        id="doc_abc",
        filename="spec.pdf",
        folder="BoostMCP",
        tags=["design"],
    )
    assert doc.status == DocumentStatus.PROCESSING
    assert doc.chunk_count == 0
    assert doc.error_message is None


def test_search_result_fields():
    r = SearchResult(
        text="hello",
        doc_id="doc_abc",
        filename="spec.pdf",
        folder="BoostMCP",
        tags=["design"],
        chunk_index=2,
        score=0.87,
    )
    assert r.source_page is None
```

- [ ] **Step 2: Run test to verify it fails**

```bash
poetry run pytest tests/test_models.py -v
```

Expected: FAIL — `ModuleNotFoundError: boostmcp.models`

- [ ] **Step 3: Implement models**

```python
# boostmcp/models.py
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class DocumentStatus(str, Enum):
    PROCESSING = "processing"
    READY = "ready"
    ERROR = "error"


@dataclass
class DocumentRecord:
    id: str
    filename: str
    folder: str = ""
    tags: list[str] = field(default_factory=list)
    status: DocumentStatus = DocumentStatus.PROCESSING
    chunk_count: int = 0
    error_message: Optional[str] = None
    original_path: str = ""
    markdown_path: str = ""


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
```

- [ ] **Step 4: Run tests**

```bash
poetry run pytest tests/test_models.py -v
```

Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add boostmcp/models.py tests/test_models.py
git commit -m "feat: add domain models for documents and search results"
```

---

### Task 4: Configuration loading

**Files:**
- Create: `boostmcp/config.py`
- Create: `tests/test_config.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_config.py
import os
from pathlib import Path

import pytest
import yaml

from boostmcp.config import Config, load_config


def test_config_defaults(tmp_data_dir):
    cfg = Config(data_dir=tmp_data_dir)
    assert cfg.web_host == "127.0.0.1"
    assert cfg.web_port == 8080
    assert cfg.embed_provider == "ollama"
    assert cfg.ollama_model == "nomic-embed-text"
    assert cfg.chunk_size == 512
    assert cfg.default_top_k == 5


def test_load_config_from_yaml(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    cfg_file = data_dir / "config.yaml"
    cfg_file.write_text(yaml.dump({
        "server": {"web_port": 9090},
        "embedding": {"provider": "openai", "openai_model": "text-embedding-3-small"},
    }))
    monkeypatch.setenv("BOOSTMCP_DATA_DIR", str(data_dir))
    cfg = load_config()
    assert cfg.web_port == 9090
    assert cfg.embed_provider == "openai"
    assert cfg.openai_model == "text-embedding-3-small"


def test_env_overrides_yaml(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "config.yaml").write_text(yaml.dump({"server": {"web_port": 9090}}))
    monkeypatch.setenv("BOOSTMCP_DATA_DIR", str(data_dir))
    monkeypatch.setenv("BOOSTMCP_WEB_PORT", "7777")
    cfg = load_config()
    assert cfg.web_port == 7777
```

- [ ] **Step 2: Run test to verify it fails**

```bash
poetry run pytest tests/test_config.py -v
```

Expected: FAIL — `ModuleNotFoundError: boostmcp.config`

- [ ] **Step 3: Implement config**

```python
# boostmcp/config.py
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


def _expand_path(p: str) -> Path:
    return Path(os.path.expanduser(p)).resolve()


@dataclass
class Config:
    data_dir: Path
    web_host: str = "127.0.0.1"
    web_port: int = 8080
    embed_provider: str = "ollama"
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "nomic-embed-text"
    openai_api_key: str = ""
    openai_model: str = "text-embedding-3-small"
    chunk_size: int = 512
    chunk_overlap: int = 64
    max_file_size_mb: int = 50
    default_top_k: int = 5
    min_score: float = 0.3

    @property
    def sqlite_path(self) -> Path:
        return self.data_dir / "data.db"

    @property
    def chroma_path(self) -> Path:
        return self.data_dir / "chroma"

    @property
    def originals_dir(self) -> Path:
        return self.data_dir / "files" / "originals"

    @property
    def markdown_dir(self) -> Path:
        return self.data_dir / "files" / "markdown"

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.chroma_path.mkdir(parents=True, exist_ok=True)
        self.originals_dir.mkdir(parents=True, exist_ok=True)
        self.markdown_dir.mkdir(parents=True, exist_ok=True)


def _apply_yaml(cfg: Config, data: dict[str, Any]) -> None:
    if server := data.get("server"):
        cfg.web_host = server.get("web_host", cfg.web_host)
        cfg.web_port = int(server.get("web_port", cfg.web_port))
    if storage := data.get("storage"):
        if data_dir := storage.get("data_dir"):
            cfg.data_dir = _expand_path(data_dir)
    if embedding := data.get("embedding"):
        cfg.embed_provider = embedding.get("provider", cfg.embed_provider)
        cfg.ollama_url = embedding.get("ollama_url", cfg.ollama_url)
        cfg.ollama_model = embedding.get("ollama_model", cfg.ollama_model)
        cfg.openai_api_key = embedding.get("openai_api_key", cfg.openai_api_key)
        cfg.openai_model = embedding.get("openai_model", cfg.openai_model)
    if ingest := data.get("ingest"):
        cfg.chunk_size = int(ingest.get("chunk_size", cfg.chunk_size))
        cfg.chunk_overlap = int(ingest.get("chunk_overlap", cfg.chunk_overlap))
        cfg.max_file_size_mb = int(ingest.get("max_file_size_mb", cfg.max_file_size_mb))
    if search := data.get("search"):
        cfg.default_top_k = int(search.get("default_top_k", cfg.default_top_k))
        cfg.min_score = float(search.get("min_score", cfg.min_score))


def _apply_env(cfg: Config) -> None:
    if v := os.getenv("BOOSTMCP_DATA_DIR"):
        cfg.data_dir = _expand_path(v)
    if v := os.getenv("BOOSTMCP_WEB_PORT"):
        cfg.web_port = int(v)
    if v := os.getenv("BOOSTMCP_EMBED_PROVIDER"):
        cfg.embed_provider = v
    if v := os.getenv("BOOSTMCP_OLLAMA_URL"):
        cfg.ollama_url = v
    if v := os.getenv("BOOSTMCP_OLLAMA_EMBED_MODEL"):
        cfg.ollama_model = v
    if v := os.getenv("BOOSTMCP_OPENAI_API_KEY"):
        cfg.openai_api_key = v
    if v := os.getenv("BOOSTMCP_CHUNK_SIZE"):
        cfg.chunk_size = int(v)
    if v := os.getenv("BOOSTMCP_MAX_FILE_MB"):
        cfg.max_file_size_mb = int(v)


def load_config() -> Config:
    data_dir = _expand_path(os.getenv("BOOSTMCP_DATA_DIR", "~/.boostmcp"))
    cfg = Config(data_dir=data_dir)
    yaml_path = cfg.data_dir / "config.yaml"
    if yaml_path.exists():
        with yaml_path.open(encoding="utf-8") as f:
            _apply_yaml(cfg, yaml.safe_load(f) or {})
    _apply_env(cfg)
    return cfg
```

- [ ] **Step 4: Run tests**

```bash
poetry run pytest tests/test_config.py -v
```

Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add boostmcp/config.py tests/test_config.py
git commit -m "feat: add config loading from YAML and environment"
```

---

### Task 5: Verify foundation

- [ ] **Step 1: Run full test suite**

```bash
poetry run pytest tests/ -v
```

Expected: all tests PASS

- [ ] **Step 2: Verify package import**

```bash
poetry run python -c "from boostmcp.config import load_config; from boostmcp.models import DocumentRecord; print('ok')"
```

Expected: `ok`
