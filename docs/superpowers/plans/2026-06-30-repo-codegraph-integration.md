# Repo Import via Codegraph — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a first-class "Repository" concept to DocGraph that clones a GitHub URL or local path, delegates AST-aware code intelligence to `colbymchenry/codegraph` via subprocess, and indexes the repo's `*.md` files through the existing Chroma pipeline. Surface both worlds through a single MCP entry point.

**Architecture:** New `docgraph/repo/` package owns lifecycle (clone, `codegraph init`, doc indexing, delete). MCP server gains `code_*` tools that shell out to `codegraph <subcommand>` scoped to a per-repo path. `*.md` files inside the cloned repo are indexed into the existing Chroma store with `repo_id` metadata so existing `search_documents` keeps working and can filter by repo. JSON shapes from codegraph are passed through verbatim.

**Tech Stack:** Python 3.10–3.13, FastAPI, asyncio subprocess for codegraph CLI, SQLite (WAL), Chroma, React 19 + Vite 6 + TS 5.7 for frontend.

## Global Constraints

- Python `>=3.10,<3.14` (from `pyproject.toml`).
- Test framework: `pytest` with `pytest-asyncio` (`asyncio_mode = "auto"`), `respx` for httpx mocks.
- All subprocess calls use `asyncio.create_subprocess_exec(*argv)` with arg-list form. **Never** `shell=True`. Argv must be `list[str]`.
- Codegraph binary name defaults to `"codegraph"` and is resolved via `$PATH`. Override via `cfg.codegraph_bin`.
- Per-repo `asyncio.Lock` serializes init / reindex / delete on the same `repo_id`.
- Clone command is exactly `git clone --depth 1 <url> <target_dir>` — default branch only, no auth flags.
- SSRF protection on URLs uses the existing `docgraph.ingest.urls.validate_url` (blocks localhost, private IPv4, link-local, non-http(s) schemes).
- JSON shapes from codegraph are passed through verbatim to MCP callers. Wrapper layer never reshapes the schema.
- Heartbeat phase update at least every 5s during `codegraph init`. No DB row may sit at the same `progress_phase` for >5s while a long-running subprocess is active.
- Existing fixture `tmp_data_dir` from `tests/conftest.py` returns `tmp_path / "docgraph_data"` and is reused everywhere.
- New code follows the codebase convention: `from __future__ import annotations`, dataclasses for records, module-level `logger = logging.getLogger(__name__)`.
- Frontend has no test runner; verification is `npm run build` (which runs `tsc --noEmit && vite build`). UI changes pass when the build is clean.

---

## File map

**Create:**
- `docgraph/repo/__init__.py`
- `docgraph/repo/codegraph_client.py`
- `docgraph/repo/manager.py`
- `docgraph/cli_repos.py`
- `tests/repo/__init__.py`
- `tests/repo/test_codegraph_client.py`
- `tests/repo/test_manager.py`
- `tests/web/test_repos_api.py`
- `tests/mcp/test_code_tools.py`
- `tests/integration/__init__.py`
- `tests/integration/test_repo_e2e.py`
- `frontend/src/components/RepoImportSection.tsx`
- `frontend/src/components/RepoTable.tsx`

**Modify:**
- `docgraph/config.py` — add `repos_dir`, `codegraph_bin`, `codegraph_init_timeout_sec`, `codegraph_query_timeout_sec`; extend `_apply_yaml`, `_apply_env`, `ensure_dirs`.
- `docgraph/models.py` — add `RepoRecord`; add `repo_id: str = ""` to `DocumentRecord`.
- `docgraph/store/sqlite.py` — new `repos` table + indexes in `init_schema`; new `repo_id` column migration on `documents`; new repo CRUD methods; `list_documents_by_repo`; read `repo_id` in `_row_to_doc`.
- `docgraph/store/chroma.py` — `search()` accepts optional `repo_id`; adds it to `where`.
- `docgraph/ingest/indexer.py` — propagate `doc.repo_id` into chunk metadata when non-empty.
- `docgraph/web/deps.py` — `AppState` gains `codegraph: CodegraphClient`, `repos()` factory.
- `docgraph/web/app.py` — add `/api/repos*` endpoints; extend `/api/health` with codegraph block.
- `docgraph/mcp/server.py` — register new tools (`list_repos`, `import_repo`, `code_*`); extend `search_documents` with optional `repo` arg.
- `docgraph/cli.py` — register `import-repo`, `list-repos`, `delete-repo` subcommands (delegate to `cli_repos.run_repos_command`).
- `docgraph/watch/ignore.py` — add `.codegraph` to `HARDCODED_IGNORE_DIRS`.
- `frontend/src/types.ts` — `Repo` type; extend `HealthInfo` with `codegraph` field.
- `frontend/src/api.ts` — `fetchRepos`, `importRepo`, `reindexRepo`, `deleteRepo`.
- `frontend/src/App.tsx` — mount Repositories section; extend polling.
- `frontend/src/components/Header.tsx` — codegraph health banner.
- `frontend/src/components/DocumentTable.tsx` — show source repo when `doc.repo_id` set.
- `Dockerfile` — install `codegraph` in runtime stage.
- `docker-compose.yml` — named volume for repos.
- `README.md` — Repositories section + new env vars.

---

## Task 1: Config additions

**Files:**
- Modify: `docgraph/config.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Consumes: existing `Config` dataclass, `_apply_yaml`, `_apply_env`.
- Produces:
  - `Config.repos_dir: Path` — default `data_dir / "repos"` (computed in `__post_init__` or property)
  - `Config.codegraph_bin: str = "codegraph"`
  - `Config.codegraph_init_timeout_sec: int = 600`
  - `Config.codegraph_query_timeout_sec: int = 30`
  - `ensure_dirs()` creates `repos_dir`
  - YAML key `repos:` with `dir`, `bin`, `init_timeout_sec`, `query_timeout_sec`
  - Env: `DOCGRAPH_REPOS_DIR`, `DOCGRAPH_CODEGRAPH_BIN`, `DOCGRAPH_CODEGRAPH_INIT_TIMEOUT_SEC`, `DOCGRAPH_CODEGRAPH_QUERY_TIMEOUT_SEC`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_config.py`:

```python
def test_config_repos_defaults(tmp_data_dir):
    cfg = Config(data_dir=tmp_data_dir)
    assert cfg.repos_dir == tmp_data_dir / "repos"
    assert cfg.codegraph_bin == "codegraph"
    assert cfg.codegraph_init_timeout_sec == 600
    assert cfg.codegraph_query_timeout_sec == 30


def test_ensure_dirs_creates_repos(tmp_data_dir):
    cfg = Config(data_dir=tmp_data_dir)
    cfg.ensure_dirs()
    assert cfg.repos_dir.is_dir()


def test_repos_yaml_override(tmp_data_dir, monkeypatch):
    import yaml as _yaml
    cfg_path = tmp_data_dir / "config.yaml"
    tmp_data_dir.mkdir(parents=True)
    cfg_path.write_text(_yaml.dump({
        "repos": {
            "bin": "/custom/codegraph",
            "init_timeout_sec": 1200,
            "query_timeout_sec": 45,
        }
    }))
    monkeypatch.setenv("DOCGRAPH_DATA_DIR", str(tmp_data_dir))
    cfg = load_config()
    assert cfg.codegraph_bin == "/custom/codegraph"
    assert cfg.codegraph_init_timeout_sec == 1200
    assert cfg.codegraph_query_timeout_sec == 45


def test_repos_env_override(tmp_data_dir, monkeypatch):
    monkeypatch.setenv("DOCGRAPH_DATA_DIR", str(tmp_data_dir))
    monkeypatch.setenv("DOCGRAPH_CODEGRAPH_BIN", "/usr/local/bin/cg")
    monkeypatch.setenv("DOCGRAPH_CODEGRAPH_INIT_TIMEOUT_SEC", "900")
    cfg = load_config()
    assert cfg.codegraph_bin == "/usr/local/bin/cg"
    assert cfg.codegraph_init_timeout_sec == 900
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `poetry run pytest tests/test_config.py -v -k repos`
Expected: FAIL (`AttributeError: 'Config' object has no attribute 'repos_dir'`).

- [ ] **Step 3: Implement Config additions**

In `docgraph/config.py`, in the `Config` dataclass add (after the existing fields):

```python
    # Repos (codegraph integration)
    codegraph_bin: str = "codegraph"
    codegraph_init_timeout_sec: int = 600
    codegraph_query_timeout_sec: int = 30
    _repos_dir_override: Path | None = None
```

Add the property:

```python
    @property
    def repos_dir(self) -> Path:
        return self._repos_dir_override or (self.data_dir / "repos")
```

In `ensure_dirs`, append:

```python
        self.repos_dir.mkdir(parents=True, exist_ok=True)
```

In `_apply_yaml`, add:

```python
    if repos := data.get("repos"):
        if d := repos.get("dir"):
            cfg._repos_dir_override = _expand_path(d)
        cfg.codegraph_bin = repos.get("bin", cfg.codegraph_bin)
        cfg.codegraph_init_timeout_sec = int(
            repos.get("init_timeout_sec", cfg.codegraph_init_timeout_sec)
        )
        cfg.codegraph_query_timeout_sec = int(
            repos.get("query_timeout_sec", cfg.codegraph_query_timeout_sec)
        )
```

In `_apply_env`, add:

```python
    if v := os.getenv("DOCGRAPH_REPOS_DIR"):
        cfg._repos_dir_override = _expand_path(v)
    if v := os.getenv("DOCGRAPH_CODEGRAPH_BIN"):
        cfg.codegraph_bin = v
    if v := os.getenv("DOCGRAPH_CODEGRAPH_INIT_TIMEOUT_SEC"):
        cfg.codegraph_init_timeout_sec = int(v)
    if v := os.getenv("DOCGRAPH_CODEGRAPH_QUERY_TIMEOUT_SEC"):
        cfg.codegraph_query_timeout_sec = int(v)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `poetry run pytest tests/test_config.py -v`
Expected: PASS (all existing + 4 new tests).

- [ ] **Step 5: Commit**

```bash
git add docgraph/config.py tests/test_config.py
git commit -m "feat(config): add repos_dir, codegraph_bin, codegraph timeouts"
```

---

## Task 2: Models

**Files:**
- Modify: `docgraph/models.py`
- Test: `tests/test_models.py`

**Interfaces:**
- Consumes: existing `DocumentStatus` enum, `dataclass`, `field`.
- Produces:
  - `RepoRecord(id, name, source_url, local_path, status, progress_pct, progress_phase, error_message, folder, tags, doc_count, cancel_requested)`
  - `DocumentRecord.repo_id: str = ""`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_models.py`:

```python
from docgraph.models import DocumentRecord, RepoRecord, DocumentStatus


def test_repo_record_defaults():
    r = RepoRecord(id="repo_x", name="go-ethereum", local_path="/tmp/x")
    assert r.source_url == ""
    assert r.status == DocumentStatus.PROCESSING
    assert r.progress_pct == 0
    assert r.progress_phase == ""
    assert r.error_message is None
    assert r.folder == ""
    assert r.tags == []
    assert r.doc_count == 0
    assert r.cancel_requested is False


def test_document_record_has_repo_id_default():
    d = DocumentRecord(id="doc_1", filename="a.md")
    assert d.repo_id == ""
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `poetry run pytest tests/test_models.py -v`
Expected: FAIL with ImportError on `RepoRecord` and AttributeError on `repo_id`.

- [ ] **Step 3: Implement model changes**

In `docgraph/models.py`, add `repo_id` to `DocumentRecord`:

```python
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
    repo_id: str = ""
```

Add `RepoRecord` at the bottom of the file:

```python
@dataclass
class RepoRecord:
    id: str
    name: str
    source_url: str = ""
    local_path: str = ""
    status: DocumentStatus = DocumentStatus.PROCESSING
    progress_pct: int = 0
    progress_phase: str = ""
    error_message: Optional[str] = None
    folder: str = ""
    tags: list[str] = field(default_factory=list)
    doc_count: int = 0
    cancel_requested: bool = False
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `poetry run pytest tests/test_models.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add docgraph/models.py tests/test_models.py
git commit -m "feat(models): add RepoRecord and DocumentRecord.repo_id"
```

---

## Task 3: SQLiteStore repo schema + methods

**Files:**
- Modify: `docgraph/store/sqlite.py`
- Test: `tests/store/test_sqlite.py`

**Interfaces:**
- Consumes: existing `_connect`, `init_schema`, `_migrate_schema`, `RepoRecord`, `DocumentStatus`.
- Produces:
  - `repos` SQL table + 2 indexes
  - `documents.repo_id` column (additive migration) + index
  - `SQLiteStore.insert_repo(repo: RepoRecord) -> None`
  - `SQLiteStore.get_repo(repo_id: str) -> RepoRecord | None`
  - `SQLiteStore.get_repo_by_name(name: str) -> RepoRecord | None`
  - `SQLiteStore.get_repo_by_source(source_url: str) -> RepoRecord | None`
  - `SQLiteStore.list_repos() -> list[RepoRecord]`
  - `SQLiteStore.update_repo_progress(repo_id, pct, phase)`
  - `SQLiteStore.update_repo_status(repo_id, status, *, doc_count=0, error_message=None)`
  - `SQLiteStore.update_repo_cancel(repo_id, cancel: bool)`
  - `SQLiteStore.delete_repo(repo_id)` — caller deletes child documents separately
  - `SQLiteStore.list_documents_by_repo(repo_id) -> list[DocumentRecord]`
  - `_row_to_doc` reads `repo_id` (defaults to `""` if column missing on legacy DBs)

- [ ] **Step 1: Write the failing tests**

Append to `tests/store/test_sqlite.py`:

```python
from docgraph.models import RepoRecord, DocumentRecord, DocumentStatus
from docgraph.config import Config
from docgraph.store import SQLiteStore


def test_repos_lifecycle(tmp_data_dir):
    cfg = Config(data_dir=tmp_data_dir)
    cfg.ensure_dirs()
    sqlite = SQLiteStore(cfg)
    sqlite.init_schema()

    r = RepoRecord(
        id="repo_a", name="go-ethereum",
        source_url="https://github.com/ethereum/go-ethereum",
        local_path=str(tmp_data_dir / "repos" / "ethereum_go-ethereum"),
        folder="chains", tags=["evm", "core"],
    )
    sqlite.insert_repo(r)
    got = sqlite.get_repo("repo_a")
    assert got is not None
    assert got.name == "go-ethereum"
    assert got.tags == ["evm", "core"]
    assert got.cancel_requested is False

    assert sqlite.get_repo_by_name("go-ethereum").id == "repo_a"
    assert sqlite.get_repo_by_source(
        "https://github.com/ethereum/go-ethereum"
    ).id == "repo_a"

    sqlite.update_repo_progress("repo_a", 40, "Building code index")
    assert sqlite.get_repo("repo_a").progress_phase == "Building code index"

    sqlite.update_repo_status("repo_a", DocumentStatus.READY, doc_count=12)
    g = sqlite.get_repo("repo_a")
    assert g.status == DocumentStatus.READY
    assert g.doc_count == 12
    assert g.progress_pct == 100

    sqlite.update_repo_cancel("repo_a", True)
    assert sqlite.get_repo("repo_a").cancel_requested is True

    sqlite.delete_repo("repo_a")
    assert sqlite.get_repo("repo_a") is None


def test_documents_have_repo_id(tmp_data_dir):
    cfg = Config(data_dir=tmp_data_dir)
    cfg.ensure_dirs()
    sqlite = SQLiteStore(cfg)
    sqlite.init_schema()

    doc = DocumentRecord(id="doc_1", filename="README.md", repo_id="repo_a")
    sqlite.insert_document(doc)
    got = sqlite.get_document("doc_1")
    assert got.repo_id == "repo_a"

    by_repo = sqlite.list_documents_by_repo("repo_a")
    assert [d.id for d in by_repo] == ["doc_1"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `poetry run pytest tests/store/test_sqlite.py -v -k 'repos_lifecycle or repo_id'`
Expected: FAIL (`AttributeError: 'SQLiteStore' object has no attribute 'insert_repo'`).

- [ ] **Step 3: Implement schema + methods**

In `docgraph/store/sqlite.py`, extend `init_schema()` to add the `repos` table after the existing `CREATE TABLE documents`:

```python
                CREATE TABLE IF NOT EXISTS repos (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    source_url TEXT NOT NULL DEFAULT '',
                    local_path TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'processing',
                    progress_pct INTEGER NOT NULL DEFAULT 0,
                    progress_phase TEXT NOT NULL DEFAULT '',
                    error_message TEXT,
                    folder TEXT NOT NULL DEFAULT '',
                    tags TEXT NOT NULL DEFAULT '[]',
                    doc_count INTEGER NOT NULL DEFAULT 0,
                    cancel_requested INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL DEFAULT (datetime('now'))
                );
                CREATE INDEX IF NOT EXISTS idx_repos_status ON repos(status);
                CREATE INDEX IF NOT EXISTS idx_repos_created_at ON repos(created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_repos_name ON repos(name);
```

In `_migrate_schema(conn)`, add (next to the existing additive migrations):

```python
        if "repo_id" not in cols:
            conn.execute(
                "ALTER TABLE documents ADD COLUMN repo_id TEXT NOT NULL DEFAULT ''"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_documents_repo_id ON documents(repo_id)"
            )
```

In `_row_to_doc`, add the `repo_id` field at the end:

```python
            repo_id=row["repo_id"] if "repo_id" in keys else "",
```

Add a private helper and the new repo methods to the class:

```python
    def _row_to_repo(self, row) -> RepoRecord:
        return RepoRecord(
            id=row["id"],
            name=row["name"],
            source_url=row["source_url"],
            local_path=row["local_path"],
            status=DocumentStatus(row["status"]),
            progress_pct=row["progress_pct"],
            progress_phase=row["progress_phase"],
            error_message=row["error_message"],
            folder=row["folder"],
            tags=json.loads(row["tags"]),
            doc_count=row["doc_count"],
            cancel_requested=bool(row["cancel_requested"]),
        )

    def insert_repo(self, repo: RepoRecord) -> None:
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO repos
                   (id, name, source_url, local_path, status, progress_pct,
                    progress_phase, error_message, folder, tags, doc_count, cancel_requested)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    repo.id, repo.name, repo.source_url, repo.local_path,
                    repo.status.value, repo.progress_pct, repo.progress_phase,
                    repo.error_message, repo.folder, json.dumps(repo.tags),
                    repo.doc_count, 1 if repo.cancel_requested else 0,
                ),
            )

    def get_repo(self, repo_id: str) -> Optional[RepoRecord]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM repos WHERE id = ?", (repo_id,)
            ).fetchone()
        return self._row_to_repo(row) if row else None

    def get_repo_by_name(self, name: str) -> Optional[RepoRecord]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM repos WHERE name = ? COLLATE NOCASE LIMIT 1",
                (name,),
            ).fetchone()
        return self._row_to_repo(row) if row else None

    def get_repo_by_source(self, source_url: str) -> Optional[RepoRecord]:
        if not source_url:
            return None
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM repos WHERE source_url = ? LIMIT 1",
                (source_url,),
            ).fetchone()
        return self._row_to_repo(row) if row else None

    def list_repos(self) -> list[RepoRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM repos ORDER BY created_at DESC"
            ).fetchall()
        return [self._row_to_repo(r) for r in rows]

    def update_repo_progress(self, repo_id: str, pct: int, phase: str = "") -> None:
        pct = max(0, min(100, int(pct)))
        with self._connect() as conn:
            conn.execute(
                "UPDATE repos SET progress_pct=?, progress_phase=? WHERE id=?",
                (pct, phase, repo_id),
            )

    def update_repo_status(
        self,
        repo_id: str,
        status: DocumentStatus,
        *,
        doc_count: int = 0,
        error_message: Optional[str] = None,
    ) -> None:
        with self._connect() as conn:
            if status == DocumentStatus.READY:
                conn.execute(
                    """UPDATE repos SET status=?, doc_count=?, error_message=NULL,
                       progress_pct=100, progress_phase='' WHERE id=?""",
                    (status.value, doc_count, repo_id),
                )
            elif status == DocumentStatus.ERROR:
                # Preserve last progress_phase so UI shows where it failed.
                conn.execute(
                    "UPDATE repos SET status=?, error_message=? WHERE id=?",
                    (status.value, error_message, repo_id),
                )
            else:
                conn.execute(
                    """UPDATE repos SET status=?, error_message=NULL,
                       progress_pct=0, progress_phase='' WHERE id=?""",
                    (status.value, repo_id),
                )

    def update_repo_cancel(self, repo_id: str, cancel: bool) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE repos SET cancel_requested=? WHERE id=?",
                (1 if cancel else 0, repo_id),
            )

    def delete_repo(self, repo_id: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM repos WHERE id = ?", (repo_id,))

    def list_documents_by_repo(self, repo_id: str) -> list[DocumentRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM documents WHERE repo_id = ? ORDER BY created_at DESC",
                (repo_id,),
            ).fetchall()
        return [self._row_to_doc(r) for r in rows]
```

Add to the imports at the top of the file:

```python
from docgraph.models import DocumentRecord, DocumentStatus, RepoRecord, SourceType
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `poetry run pytest tests/store/test_sqlite.py -v`
Expected: PASS (existing + 2 new tests).

- [ ] **Step 5: Commit**

```bash
git add docgraph/store/sqlite.py tests/store/test_sqlite.py
git commit -m "feat(store): repos table + repo_id column on documents"
```

---

## Task 4: CodegraphClient (subprocess wrapper)

**Files:**
- Create: `docgraph/repo/__init__.py`, `docgraph/repo/codegraph_client.py`
- Create: `tests/repo/__init__.py`, `tests/repo/test_codegraph_client.py`

**Interfaces:**
- Consumes: stdlib `asyncio`, `json`, `subprocess.CalledProcessError`.
- Produces:
  - `class CodegraphClient`
  - `async health_check() -> str` — returns version string; raises `RuntimeError` with install hint if binary missing or non-zero exit
  - `async init(repo_path: Path, *, progress_cb: Callable[[str], None] | None = None) -> None` — runs `codegraph init`; calls `progress_cb(phase)` on heartbeat (every 5s while subprocess is running)
  - `async run(subcommand: str, *args: str, repo_path: Path, timeout: float | None = None) -> Any` — runs `codegraph <sub> [--json] <args>`, returns parsed JSON; raises `TimeoutError` or `RuntimeError`
  - `class CodegraphNotInstalled(RuntimeError)` — distinguishable error type
  - Class constant `INSTALL_HINT = "Install codegraph: curl -fsSL https://raw.githubusercontent.com/colbymchenry/codegraph/main/install.sh | sh"`

- [ ] **Step 1: Write the failing tests**

Create `tests/repo/__init__.py` (empty file).

Create `tests/repo/test_codegraph_client.py`:

```python
import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from docgraph.repo.codegraph_client import CodegraphClient, CodegraphNotInstalled


def _fake_proc(stdout: bytes = b"", stderr: bytes = b"", returncode: int = 0):
    proc = MagicMock()
    proc.communicate = AsyncMock(return_value=(stdout, stderr))
    proc.returncode = returncode
    proc.terminate = MagicMock()
    proc.kill = MagicMock()
    proc.wait = AsyncMock(return_value=returncode)
    return proc


@pytest.mark.asyncio
async def test_health_check_returns_version():
    client = CodegraphClient(bin="codegraph")
    proc = _fake_proc(stdout=b"codegraph 0.5.1\n", returncode=0)
    with patch(
        "asyncio.create_subprocess_exec",
        AsyncMock(return_value=proc),
    ) as mocked:
        version = await client.health_check()
    assert "0.5.1" in version
    args, kwargs = mocked.call_args
    assert args[0] == "codegraph"
    assert "--version" in args


@pytest.mark.asyncio
async def test_health_check_missing_binary():
    client = CodegraphClient(bin="codegraph")
    with patch(
        "asyncio.create_subprocess_exec",
        AsyncMock(side_effect=FileNotFoundError("no such file")),
    ):
        with pytest.raises(CodegraphNotInstalled) as exc_info:
            await client.health_check()
    assert "install.sh" in str(exc_info.value)


@pytest.mark.asyncio
async def test_run_passes_json_flag_and_parses(tmp_path):
    client = CodegraphClient(bin="codegraph")
    payload = {"results": [{"name": "Validator"}]}
    proc = _fake_proc(stdout=json.dumps(payload).encode(), returncode=0)
    with patch(
        "asyncio.create_subprocess_exec",
        AsyncMock(return_value=proc),
    ) as mocked:
        result = await client.run("search", "Validator", repo_path=tmp_path)
    assert result == payload
    args, kwargs = mocked.call_args
    assert args == ("codegraph", "search", "Validator", "--json")
    assert kwargs["cwd"] == str(tmp_path)


@pytest.mark.asyncio
async def test_run_raises_on_nonzero(tmp_path):
    client = CodegraphClient(bin="codegraph")
    proc = _fake_proc(stdout=b"", stderr=b"boom", returncode=2)
    with patch(
        "asyncio.create_subprocess_exec",
        AsyncMock(return_value=proc),
    ):
        with pytest.raises(RuntimeError) as exc_info:
            await client.run("search", "X", repo_path=tmp_path)
    assert "boom" in str(exc_info.value)


@pytest.mark.asyncio
async def test_init_invokes_progress_cb(tmp_path):
    client = CodegraphClient(bin="codegraph", init_heartbeat_sec=0.05)
    proc = _fake_proc(stdout=b"", returncode=0)

    async def slow_communicate():
        await asyncio.sleep(0.2)
        return (b"", b"")

    proc.communicate = slow_communicate
    phases: list[str] = []
    with patch(
        "asyncio.create_subprocess_exec",
        AsyncMock(return_value=proc),
    ):
        await client.init(tmp_path, progress_cb=phases.append)
    assert len(phases) >= 1
    assert all("Building code index" in p for p in phases)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `poetry run pytest tests/repo/test_codegraph_client.py -v`
Expected: FAIL with ModuleNotFoundError on `docgraph.repo.codegraph_client`.

- [ ] **Step 3: Implement CodegraphClient**

Create `docgraph/repo/__init__.py` (empty file).

Create `docgraph/repo/codegraph_client.py`:

```python
from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)


class CodegraphNotInstalled(RuntimeError):
    pass


class CodegraphClient:
    INSTALL_HINT = (
        "codegraph CLI not found. Install with: "
        "curl -fsSL https://raw.githubusercontent.com/colbymchenry/codegraph/main/install.sh | sh"
    )

    def __init__(
        self,
        *,
        bin: str = "codegraph",
        query_timeout_sec: float = 30.0,
        init_timeout_sec: float = 600.0,
        init_heartbeat_sec: float = 5.0,
    ) -> None:
        self._bin = bin
        self._query_timeout = query_timeout_sec
        self._init_timeout = init_timeout_sec
        self._heartbeat = init_heartbeat_sec

    async def _spawn(self, *argv: str, cwd: str | None = None):
        try:
            return await asyncio.create_subprocess_exec(
                self._bin, *argv,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
            )
        except FileNotFoundError as exc:
            raise CodegraphNotInstalled(self.INSTALL_HINT) from exc

    async def health_check(self) -> str:
        proc = await self._spawn("--version")
        stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            raise CodegraphNotInstalled(
                f"{self.INSTALL_HINT} (exit {proc.returncode}: {stderr.decode(errors='replace')})"
            )
        return stdout.decode(errors="replace").strip()

    async def run(
        self,
        subcommand: str,
        *args: str,
        repo_path: Path,
        timeout: float | None = None,
    ) -> Any:
        argv = (subcommand, *args, "--json")
        proc = await self._spawn(*argv, cwd=str(repo_path))
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=timeout or self._query_timeout
            )
        except asyncio.TimeoutError:
            proc.terminate()
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(proc.wait(), timeout=5)
            if proc.returncode is None:
                proc.kill()
                await proc.wait()
            raise TimeoutError(
                f"codegraph {subcommand} timed out after "
                f"{timeout or self._query_timeout}s"
            )
        if proc.returncode != 0:
            raise RuntimeError(
                f"codegraph {subcommand} failed (exit {proc.returncode}): "
                f"{stderr.decode(errors='replace').strip()}"
            )
        text = stdout.decode(errors="replace").strip()
        if not text:
            return None
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return text

    async def init(
        self,
        repo_path: Path,
        *,
        progress_cb: Callable[[str], None] | None = None,
    ) -> None:
        proc = await self._spawn("init", cwd=str(repo_path))
        comm_task = asyncio.create_task(proc.communicate())
        elapsed = 0.0
        while not comm_task.done():
            try:
                await asyncio.wait_for(asyncio.shield(comm_task), timeout=self._heartbeat)
            except asyncio.TimeoutError:
                elapsed += self._heartbeat
                if progress_cb:
                    progress_cb(
                        f"Building code index ({int(elapsed)}s elapsed)"
                    )
                if elapsed >= self._init_timeout:
                    proc.terminate()
                    with contextlib.suppress(asyncio.TimeoutError):
                        await asyncio.wait_for(comm_task, timeout=5)
                    if proc.returncode is None:
                        proc.kill()
                        await proc.wait()
                    raise TimeoutError(
                        f"codegraph init timed out after {self._init_timeout}s"
                    )
        stdout, stderr = comm_task.result()
        if proc.returncode != 0:
            raise RuntimeError(
                f"codegraph init failed (exit {proc.returncode}): "
                f"{stderr.decode(errors='replace').strip()}"
            )
        logger.debug("codegraph init done in %s", repo_path)

    async def aclose(self) -> None:
        return None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `poetry run pytest tests/repo/test_codegraph_client.py -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add docgraph/repo/__init__.py docgraph/repo/codegraph_client.py tests/repo/__init__.py tests/repo/test_codegraph_client.py
git commit -m "feat(repo): CodegraphClient subprocess wrapper"
```

---

## Task 5: RepoManager + watcher .codegraph ignore

**Files:**
- Create: `docgraph/repo/manager.py`
- Modify: `docgraph/watch/ignore.py`
- Test: `tests/repo/test_manager.py`

**Interfaces:**
- Consumes: `Config`, `SQLiteStore`, `Indexer`, `CodegraphClient`, `validate_url`, `DocumentRecord`, `RepoRecord`, `DocumentStatus`.
- Produces:
  - `class RepoManager`
  - `async import_repo(source: str, *, folder: str = "", tags: tuple[str, ...] = ()) -> str`
  - `async reindex_repo(repo_id: str) -> None`
  - `async delete_repo(repo_id: str) -> int` — returns count of cascaded docs
  - `resolve(ref: str | None) -> RepoRecord | None`
  - `list_repos() -> list[RepoRecord]`
  - `_repo_slug(url_or_path: str) -> str` — `<owner>_<name>` for URLs, `<dirname>` for paths
  - `MD_SKIP_DIRS = frozenset({".git", "node_modules", "vendor", "dist", "build", "target", "__pycache__", ".venv", ".next", ".codegraph"})`

- [ ] **Step 1: Write the failing tests**

Create `tests/repo/test_manager.py`:

```python
import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from docgraph.config import Config
from docgraph.models import DocumentStatus
from docgraph.repo.manager import RepoManager, _repo_slug
from docgraph.store import ChromaStore, FileStore, SQLiteStore


def test_repo_slug_from_https_url():
    assert _repo_slug("https://github.com/ethereum/go-ethereum") == "ethereum_go-ethereum"
    assert _repo_slug("https://github.com/ethereum/go-ethereum.git") == "ethereum_go-ethereum"


def test_repo_slug_from_local_path(tmp_path):
    d = tmp_path / "myproj"
    d.mkdir()
    assert _repo_slug(str(d)) == "myproj"


def _populate_repo(repo_dir: Path) -> None:
    repo_dir.mkdir(parents=True, exist_ok=True)
    (repo_dir / ".git").mkdir()
    (repo_dir / "README.md").write_text("# Hello\n\nIntro to the project.")
    (repo_dir / "docs").mkdir()
    (repo_dir / "docs" / "design.md").write_text("# Design\n\nArchitecture notes.")
    (repo_dir / "node_modules").mkdir()
    (repo_dir / "node_modules" / "ignored.md").write_text("should be skipped")
    (repo_dir / "main.go").write_text("package main\n")


@pytest.mark.asyncio
async def test_import_repo_local_path(tmp_data_dir, monkeypatch):
    cfg = Config(data_dir=tmp_data_dir)
    cfg.ensure_dirs()
    sqlite = SQLiteStore(cfg)
    sqlite.init_schema()
    chroma = ChromaStore(cfg)
    files = FileStore(cfg)

    local_repo = tmp_data_dir / "src_repo"
    _populate_repo(local_repo)

    codegraph = MagicMock()
    codegraph.init = AsyncMock()

    indexer = MagicMock()
    indexed_paths: list[str] = []

    async def fake_index_markdown(doc_id, markdown):
        indexed_paths.append(doc_id)

    indexer.index_markdown = AsyncMock(side_effect=fake_index_markdown)

    mgr = RepoManager(
        cfg=cfg, sqlite=sqlite, files=files, chroma=chroma,
        codegraph=codegraph, indexer_factory=lambda: indexer,
    )
    repo_id = await mgr.import_repo(str(local_repo), folder="chains", tags=("evm",))

    repo = sqlite.get_repo(repo_id)
    assert repo.status == DocumentStatus.READY
    assert repo.doc_count == 2  # README + docs/design.md ; node_modules skipped
    codegraph.init.assert_awaited_once()
    assert len(indexed_paths) == 2


@pytest.mark.asyncio
async def test_import_repo_rejects_duplicate_url(tmp_data_dir):
    cfg = Config(data_dir=tmp_data_dir)
    cfg.ensure_dirs()
    sqlite = SQLiteStore(cfg)
    sqlite.init_schema()
    chroma = ChromaStore(cfg)
    files = FileStore(cfg)

    codegraph = MagicMock()
    codegraph.init = AsyncMock()

    indexer = MagicMock()
    indexer.index_markdown = AsyncMock()

    mgr = RepoManager(
        cfg=cfg, sqlite=sqlite, files=files, chroma=chroma,
        codegraph=codegraph, indexer_factory=lambda: indexer,
    )

    local_repo = tmp_data_dir / "src_repo"
    _populate_repo(local_repo)
    await mgr.import_repo(str(local_repo))
    with pytest.raises(ValueError) as exc_info:
        await mgr.import_repo(str(local_repo))
    assert "already imported" in str(exc_info.value)


@pytest.mark.asyncio
async def test_delete_repo_cascades_docs(tmp_data_dir):
    cfg = Config(data_dir=tmp_data_dir)
    cfg.ensure_dirs()
    sqlite = SQLiteStore(cfg)
    sqlite.init_schema()
    chroma = ChromaStore(cfg)
    files = FileStore(cfg)
    codegraph = MagicMock()
    codegraph.init = AsyncMock()
    indexer = MagicMock()
    indexer.index_markdown = AsyncMock()

    mgr = RepoManager(
        cfg=cfg, sqlite=sqlite, files=files, chroma=chroma,
        codegraph=codegraph, indexer_factory=lambda: indexer,
    )

    local_repo = tmp_data_dir / "src_repo"
    _populate_repo(local_repo)
    repo_id = await mgr.import_repo(str(local_repo))
    assert len(sqlite.list_documents_by_repo(repo_id)) == 2

    cascaded = await mgr.delete_repo(repo_id)
    assert cascaded == 2
    assert sqlite.get_repo(repo_id) is None
    assert sqlite.list_documents_by_repo(repo_id) == []


def test_resolve_by_id_name_and_single(tmp_data_dir):
    cfg = Config(data_dir=tmp_data_dir)
    cfg.ensure_dirs()
    sqlite = SQLiteStore(cfg)
    sqlite.init_schema()
    chroma = ChromaStore(cfg)
    files = FileStore(cfg)
    codegraph = MagicMock()
    indexer = MagicMock()

    mgr = RepoManager(
        cfg=cfg, sqlite=sqlite, files=files, chroma=chroma,
        codegraph=codegraph, indexer_factory=lambda: indexer,
    )
    from docgraph.models import RepoRecord
    sqlite.insert_repo(RepoRecord(
        id="repo_x", name="go-ethereum",
        local_path=str(tmp_data_dir / "x"),
        status=DocumentStatus.READY,
    ))
    assert mgr.resolve("repo_x").id == "repo_x"
    assert mgr.resolve("GO-ETHEREUM").id == "repo_x"
    assert mgr.resolve(None).id == "repo_x"


def test_watcher_ignores_codegraph_dir():
    from docgraph.watch.ignore import HARDCODED_IGNORE_DIRS
    assert ".codegraph" in HARDCODED_IGNORE_DIRS
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `poetry run pytest tests/repo/test_manager.py -v`
Expected: FAIL with ModuleNotFoundError on `docgraph.repo.manager`.

- [ ] **Step 3: Modify watcher ignore list**

Edit `docgraph/watch/ignore.py`:

```python
HARDCODED_IGNORE_DIRS = frozenset({
    ".git", ".hg", ".svn",
    "node_modules", "__pycache__", ".venv", "venv", ".tox",
    ".mypy_cache", ".pytest_cache", ".ruff_cache",
    "target", "dist", "build", ".next", ".nuxt",
    ".codegraph",
})
```

- [ ] **Step 4: Implement RepoManager**

Create `docgraph/repo/manager.py`:

```python
from __future__ import annotations

import asyncio
import logging
import shutil
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Callable, Iterable
from urllib.parse import urlparse

from docgraph.config import Config
from docgraph.ingest.indexer import Indexer
from docgraph.ingest.urls import validate_url
from docgraph.models import DocumentRecord, DocumentStatus, RepoRecord
from docgraph.repo.codegraph_client import CodegraphClient
from docgraph.store.chroma import ChromaStore
from docgraph.store.files import FileStore
from docgraph.store.sqlite import SQLiteStore

logger = logging.getLogger(__name__)

MD_SKIP_DIRS = frozenset({
    ".git", "node_modules", "vendor", "dist", "build", "target",
    "__pycache__", ".venv", ".next", ".codegraph",
})


def _repo_slug(source: str) -> str:
    if source.startswith(("http://", "https://", "git@")):
        # Tolerate git@github.com:owner/repo(.git) too.
        if source.startswith("git@"):
            _, _, tail = source.partition(":")
            parts = tail.strip("/").split("/")
        else:
            parts = urlparse(source).path.strip("/").split("/")
        if len(parts) >= 2:
            owner, name = parts[-2], parts[-1]
        else:
            owner, name = "remote", parts[-1] if parts else "repo"
        if name.endswith(".git"):
            name = name[:-4]
        return f"{owner}_{name}"
    return Path(source).resolve().name


def _iter_markdown_files(root: Path) -> Iterable[Path]:
    for path in root.rglob("*.md"):
        if any(part in MD_SKIP_DIRS for part in path.relative_to(root).parts):
            continue
        yield path


class RepoManager:
    def __init__(
        self,
        *,
        cfg: Config,
        sqlite: SQLiteStore,
        files: FileStore,
        chroma: ChromaStore,
        codegraph: CodegraphClient,
        indexer_factory: Callable[[], Indexer],
    ) -> None:
        self._cfg = cfg
        self._sqlite = sqlite
        self._files = files
        self._chroma = chroma
        self._codegraph = codegraph
        self._indexer_factory = indexer_factory
        self._locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)

    def list_repos(self) -> list[RepoRecord]:
        return self._sqlite.list_repos()

    def get_repo(self, repo_id: str) -> RepoRecord | None:
        return self._sqlite.get_repo(repo_id)

    def resolve(self, ref: str | None) -> RepoRecord | None:
        if ref:
            if r := self._sqlite.get_repo(ref):
                return r
            if r := self._sqlite.get_repo_by_name(ref):
                return r
            return None
        ready = [r for r in self._sqlite.list_repos() if r.status == DocumentStatus.READY]
        return ready[0] if len(ready) == 1 else None

    def _is_url(self, source: str) -> bool:
        return source.startswith(("http://", "https://", "git@"))

    async def _clone(self, url: str, target: Path) -> None:
        proc = await asyncio.create_subprocess_exec(
            "git", "clone", "--depth", "1", url, str(target),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(
                f"git clone failed (exit {proc.returncode}): "
                f"{stderr.decode(errors='replace').strip()}"
            )

    async def import_repo(
        self,
        source: str,
        *,
        folder: str = "",
        tags: tuple[str, ...] = (),
    ) -> str:
        is_url = self._is_url(source)
        if is_url:
            validate_url(source)
            if self._sqlite.get_repo_by_source(source) is not None:
                raise ValueError(
                    f"repo already imported: {source}; use reindex to refresh"
                )
            slug = _repo_slug(source)
            target = self._cfg.repos_dir / slug
            source_url = source
        else:
            local = Path(source).resolve()
            if not local.is_dir():
                raise ValueError(f"local path not found or not a directory: {source}")
            if self._sqlite.get_repo_by_source(str(local)) is not None:
                raise ValueError(
                    f"repo already imported: {source}; use reindex to refresh"
                )
            slug = _repo_slug(str(local))
            target = local
            source_url = str(local)

        repo_id = f"repo_{uuid.uuid4().hex[:12]}"
        repo = RepoRecord(
            id=repo_id, name=slug.split("_", 1)[-1] if "_" in slug else slug,
            source_url=source_url, local_path=str(target),
            folder=folder, tags=list(tags),
        )
        self._sqlite.insert_repo(repo)
        self._sqlite.update_repo_progress(repo_id, 0, "Queued (0%)")
        try:
            async with self._locks[repo_id]:
                await self._run_import(repo_id, target, is_url, source_url)
        except Exception as exc:
            self._sqlite.update_repo_status(
                repo_id, DocumentStatus.ERROR, error_message=str(exc)
            )
            raise
        return repo_id

    async def _run_import(
        self, repo_id: str, target: Path, is_url: bool, source_url: str
    ) -> None:
        if is_url:
            self._sqlite.update_repo_progress(repo_id, 5, f"Cloning {source_url} (5%)")
            await self._clone(source_url, target)
        self._sqlite.update_repo_progress(repo_id, 30, "Building code index (30%)")

        def hb(phase: str) -> None:
            self._sqlite.update_repo_progress(repo_id, 50, phase)

        await self._codegraph.init(target, progress_cb=hb)

        md_files = list(_iter_markdown_files(target))
        total = len(md_files) or 1
        self._sqlite.update_repo_progress(
            repo_id, 80, f"Indexing docs (0/{total})"
        )
        indexer = self._indexer_factory()
        for idx, md in enumerate(md_files, start=1):
            content = md.read_text(encoding="utf-8", errors="replace")
            doc_id = f"doc_{uuid.uuid4().hex[:12]}"
            self._sqlite.insert_document(DocumentRecord(
                id=doc_id,
                filename=str(md.relative_to(target)),
                folder=self._sqlite.get_repo(repo_id).folder,
                tags=self._sqlite.get_repo(repo_id).tags,
                original_path=str(md),
                repo_id=repo_id,
            ))
            self._sqlite.update_progress(doc_id, 0, "Queued for indexing")
            try:
                await indexer.index_markdown(doc_id, content)
            except Exception as exc:
                logger.warning(
                    "md indexing failed in repo %s: %s — %s", repo_id, md, exc
                )
            pct = 80 + int(15 * idx / total)
            self._sqlite.update_repo_progress(
                repo_id, pct, f"Indexing docs ({idx}/{total})"
            )

        doc_count = len(self._sqlite.list_documents_by_repo(repo_id))
        self._sqlite.update_repo_progress(repo_id, 95, "Finalizing (95%)")
        self._sqlite.update_repo_status(
            repo_id, DocumentStatus.READY, doc_count=doc_count
        )

    async def reindex_repo(self, repo_id: str) -> None:
        repo = self._sqlite.get_repo(repo_id)
        if repo is None:
            raise ValueError(f"repo not found: {repo_id}")
        async with self._locks[repo_id]:
            for doc in self._sqlite.list_documents_by_repo(repo_id):
                self._chroma.delete_by_doc_id(doc.id)
                self._sqlite.delete_document(doc.id)
                self._files.delete_doc_files(doc.id)
            self._sqlite.update_repo_status(repo_id, DocumentStatus.PROCESSING)
            self._sqlite.update_repo_progress(repo_id, 0, "Starting re-index (0%)")
            target = Path(repo.local_path)
            await self._run_import(repo_id, target, is_url=False, source_url=repo.source_url)

    async def delete_repo(self, repo_id: str) -> int:
        repo = self._sqlite.get_repo(repo_id)
        if repo is None:
            return 0
        async with self._locks[repo_id]:
            docs = self._sqlite.list_documents_by_repo(repo_id)
            for doc in docs:
                self._chroma.delete_by_doc_id(doc.id)
                self._files.delete_doc_files(doc.id)
                self._sqlite.delete_document(doc.id)
            self._sqlite.delete_repo(repo_id)
            if repo.source_url.startswith(("http://", "https://", "git@")):
                # Only remove clones that DocGraph created; never touch user-owned paths.
                target = Path(repo.local_path)
                if target.is_dir() and target.is_relative_to(self._cfg.repos_dir):
                    shutil.rmtree(target, ignore_errors=True)
        return len(docs)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `poetry run pytest tests/repo/test_manager.py -v`
Expected: PASS (6 tests).

- [ ] **Step 6: Commit**

```bash
git add docgraph/repo/manager.py docgraph/watch/ignore.py tests/repo/test_manager.py
git commit -m "feat(repo): RepoManager lifecycle + watcher ignores .codegraph"
```

---

## Task 6: Repo_id propagation in chunk metadata

**Files:**
- Modify: `docgraph/ingest/indexer.py`, `docgraph/store/chroma.py`
- Test: `tests/store/test_chroma.py`, `tests/ingest/test_indexer.py`

**Interfaces:**
- Consumes: existing `Indexer.index_markdown` chunk-metadata loop; existing `ChromaStore.search` `where` builder.
- Produces:
  - Chroma chunk metadata gains `repo_id` when `doc.repo_id` is non-empty
  - `ChromaStore.search(..., repo_id: str | None = None)` adds equality filter to `where["repo_id"]`

- [ ] **Step 1: Write the failing tests**

Append to `tests/store/test_chroma.py`:

```python
def test_chroma_search_filters_by_repo_id(tmp_data_dir):
    from docgraph.config import Config
    from docgraph.store import ChromaStore

    cfg = Config(data_dir=tmp_data_dir)
    cfg.ensure_dirs()
    chroma = ChromaStore(cfg)
    chroma.upsert_chunks([
        {
            "id": "a_0", "embedding": [0.1] * 768, "text": "alpha",
            "metadata": {
                "doc_id": "a", "filename": "a.md", "folder": "",
                "tags": "[]", "chunk_index": 0, "repo_id": "repo_x",
            },
        },
        {
            "id": "b_0", "embedding": [0.1] * 768, "text": "beta",
            "metadata": {
                "doc_id": "b", "filename": "b.md", "folder": "",
                "tags": "[]", "chunk_index": 0, "repo_id": "repo_y",
            },
        },
    ])
    out = chroma.search(query_embedding=[0.1] * 768, top_k=5, repo_id="repo_x")
    assert [r["doc_id"] for r in out] == ["a"]
```

Append to `tests/ingest/test_indexer.py`:

```python
@pytest.mark.asyncio
@respx.mock
async def test_indexer_writes_repo_id_metadata(tmp_data_dir):
    from docgraph.models import DocumentRecord
    cfg = Config(data_dir=tmp_data_dir)
    cfg.ensure_dirs()
    sqlite = SQLiteStore(cfg)
    sqlite.init_schema()
    chroma = ChromaStore(cfg)
    files = FileStore(cfg)
    sqlite.insert_document(DocumentRecord(
        id="doc_r1", filename="README.md", repo_id="repo_z"
    ))
    respx.post(f"{cfg.ollama_url}/api/embed").mock(
        return_value=httpx.Response(200, json={"embeddings": [[0.1] * 768]})
    )
    indexer = Indexer(
        cfg, sqlite, files, chroma,
        OllamaEmbedder(cfg.ollama_url, cfg.ollama_model),
    )
    await indexer.index_markdown("doc_r1", "# Hello\n\nworld")
    hits = chroma.search(query_embedding=[0.1] * 768, top_k=5, repo_id="repo_z")
    assert hits and hits[0]["doc_id"] == "doc_r1"
```

(If the existing `tests/ingest/test_indexer.py` doesn't import `respx`/`httpx`/`OllamaEmbedder`/`Config`/etc., add the matching imports from neighbouring tests.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `poetry run pytest tests/store/test_chroma.py tests/ingest/test_indexer.py -v -k 'repo_id or filters_by_repo'`
Expected: FAIL because `ChromaStore.search` doesn't accept `repo_id` and `Indexer` doesn't emit it.

- [ ] **Step 3: Implement metadata propagation + filter**

In `docgraph/store/chroma.py`, extend `search` signature and `where` construction:

```python
    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        folder: Optional[str] = None,
        tags: Optional[list[str]] = None,
        repo_id: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        where: dict[str, Any] = {}
        if folder:
            where["folder"] = folder
        if repo_id:
            where["repo_id"] = repo_id
        ...
```

In `docgraph/ingest/indexer.py`, in the chunk-metadata builder add (before/after the existing `if doc.source_url:` block):

```python
                if doc.repo_id:
                    metadata["repo_id"] = doc.repo_id
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `poetry run pytest tests/store/test_chroma.py tests/ingest/test_indexer.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add docgraph/store/chroma.py docgraph/ingest/indexer.py tests/store/test_chroma.py tests/ingest/test_indexer.py
git commit -m "feat(ingest): propagate repo_id into chunk metadata + chroma filter"
```

---

## Task 7: AppState wiring

**Files:**
- Modify: `docgraph/web/deps.py`
- Test: `tests/web/test_deps.py` (create if missing)

**Interfaces:**
- Consumes: `Config`, `CodegraphClient`, `RepoManager`.
- Produces:
  - `AppState.codegraph: CodegraphClient`
  - `AppState.repos() -> RepoManager`

- [ ] **Step 1: Write the failing test**

Create or append to `tests/web/test_deps.py`:

```python
from docgraph.config import Config
from docgraph.repo.codegraph_client import CodegraphClient
from docgraph.repo.manager import RepoManager
from docgraph.web.deps import AppState


def test_appstate_has_codegraph_and_repos(tmp_data_dir):
    cfg = Config(data_dir=tmp_data_dir)
    state = AppState.create(cfg)
    assert isinstance(state.codegraph, CodegraphClient)
    assert isinstance(state.repos(), RepoManager)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `poetry run pytest tests/web/test_deps.py -v`
Expected: FAIL with `AttributeError` on `state.codegraph`.

- [ ] **Step 3: Implement wiring**

In `docgraph/web/deps.py`:

```python
from docgraph.repo.codegraph_client import CodegraphClient
from docgraph.repo.manager import RepoManager


@dataclass
class AppState:
    cfg: Config
    sqlite: SQLiteStore
    files: FileStore
    chroma: ChromaStore
    embedder: EmbeddingProvider
    codegraph: CodegraphClient

    @classmethod
    def create(cls, cfg: Config) -> "AppState":
        cfg.ensure_dirs()
        sqlite = SQLiteStore(cfg)
        sqlite.init_schema()
        return cls(
            cfg=cfg,
            sqlite=sqlite,
            files=FileStore(cfg),
            chroma=ChromaStore(cfg),
            embedder=create_embedder(cfg),
            codegraph=CodegraphClient(
                bin=cfg.codegraph_bin,
                init_timeout_sec=cfg.codegraph_init_timeout_sec,
                query_timeout_sec=cfg.codegraph_query_timeout_sec,
            ),
        )

    def indexer(self) -> Indexer:
        return Indexer(
            self.cfg, self.sqlite, self.files, self.chroma, self.embedder
        )

    def repos(self) -> RepoManager:
        return RepoManager(
            cfg=self.cfg, sqlite=self.sqlite, files=self.files,
            chroma=self.chroma, codegraph=self.codegraph,
            indexer_factory=self.indexer,
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `poetry run pytest tests/web/test_deps.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add docgraph/web/deps.py tests/web/test_deps.py
git commit -m "feat(deps): wire codegraph + repos into AppState"
```

---

## Task 8: REST endpoints + health codegraph block

**Files:**
- Modify: `docgraph/web/app.py`
- Create: `tests/web/test_repos_api.py`

**Interfaces:**
- Consumes: `RepoManager`, existing FastAPI `BackgroundTasks` + `Request` injection pattern.
- Produces:
  - `POST /api/repos` → `202 {"repo_id": "...", "status": "processing"}`
  - `GET /api/repos` → `[RepoRecord-as-json]`
  - `GET /api/repos/{repo_id}` → `RepoRecord-as-json` or `404`
  - `POST /api/repos/{repo_id}/reindex` → `202`
  - `DELETE /api/repos/{repo_id}` → `{"deleted": id, "cascaded_docs": N}` or `404`
  - `GET /api/health` returns `{"codegraph": {"ok": bool, "version": str, "error": str}}` in addition to existing fields

- [ ] **Step 1: Write the failing tests**

Create `tests/web/test_repos_api.py`:

```python
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from docgraph.config import Config
from docgraph.repo.codegraph_client import CodegraphNotInstalled
from docgraph.web.app import create_app
from docgraph.web.deps import AppState


@pytest.fixture
def client(tmp_data_dir):
    cfg = Config(data_dir=tmp_data_dir)
    state = AppState.create(cfg)
    app = create_app(cfg, state=state, mount_mcp=False)
    return TestClient(app), state, cfg


def _populate_repo(repo_dir: Path) -> None:
    repo_dir.mkdir(parents=True, exist_ok=True)
    (repo_dir / ".git").mkdir()
    (repo_dir / "README.md").write_text("# Hi")


def test_create_repo_returns_202(client, monkeypatch, tmp_data_dir):
    c, state, _ = client
    local = tmp_data_dir / "src_repo"
    _populate_repo(local)
    state.codegraph.init = AsyncMock()
    with patch(
        "docgraph.repo.manager.Indexer.index_markdown", AsyncMock()
    ):
        resp = c.post("/api/repos", json={"source": str(local)})
    assert resp.status_code == 202
    body = resp.json()
    assert body["repo_id"].startswith("repo_")


def test_list_get_delete_repo(client, tmp_data_dir):
    c, state, _ = client
    local = tmp_data_dir / "src_repo"
    _populate_repo(local)
    state.codegraph.init = AsyncMock()
    with patch(
        "docgraph.repo.manager.Indexer.index_markdown", AsyncMock()
    ):
        rid = c.post("/api/repos", json={"source": str(local)}).json()["repo_id"]
    assert c.get("/api/repos").status_code == 200
    assert c.get(f"/api/repos/{rid}").status_code == 200
    assert c.get("/api/repos/repo_unknown").status_code == 404
    d = c.delete(f"/api/repos/{rid}").json()
    assert d["deleted"] == rid


def test_create_repo_503_when_codegraph_missing(client, tmp_data_dir):
    c, state, _ = client
    state.codegraph.health_check = AsyncMock(side_effect=CodegraphNotInstalled("nope"))
    local = tmp_data_dir / "src_repo"
    _populate_repo(local)
    resp = c.post("/api/repos", json={"source": str(local)})
    assert resp.status_code == 503
    assert "install" in resp.json()["detail"].lower()


def test_health_reports_codegraph(client):
    c, state, _ = client
    state.codegraph.health_check = AsyncMock(return_value="codegraph 0.5.1")
    body = c.get("/api/health").json()
    assert body["codegraph"]["ok"] is True
    assert "0.5.1" in body["codegraph"]["version"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `poetry run pytest tests/web/test_repos_api.py -v`
Expected: FAIL (`404` because endpoints don't exist).

- [ ] **Step 3: Implement endpoints + health field**

In `docgraph/web/app.py`, extend the existing imports:

```python
from docgraph.repo.codegraph_client import CodegraphNotInstalled
```

Extend `/api/health` to compute and include codegraph state:

```python
@app.get("/api/health")
async def health(request: Request):
    st: AppState = request.app.state.docgraph
    ollama_ok = True
    ollama_error = ""
    try:
        await st.embedder.health_check()
    except Exception as exc:
        ollama_ok = False
        ollama_error = str(exc)
    cg_ok = True
    cg_version = ""
    cg_error = ""
    try:
        cg_version = await st.codegraph.health_check()
    except Exception as exc:
        cg_ok = False
        cg_error = str(exc)
    mcp_url = f"http://{st.cfg.web_host}:{st.cfg.web_port}/mcp/sse"
    return {
        "status": "ok",
        "ollama": {"ok": ollama_ok, "error": ollama_error},
        "embed_provider": st.cfg.embed_provider,
        "mcp_sse_url": mcp_url,
        "codegraph": {"ok": cg_ok, "version": cg_version, "error": cg_error},
    }
```

Add a helper that serializes a `RepoRecord`:

```python
def _repo_to_json(repo) -> dict:
    return {
        "id": repo.id,
        "name": repo.name,
        "source_url": repo.source_url,
        "local_path": repo.local_path,
        "status": repo.status.value,
        "progress_pct": repo.progress_pct,
        "progress_phase": repo.progress_phase,
        "error_message": repo.error_message,
        "folder": repo.folder,
        "tags": repo.tags,
        "doc_count": repo.doc_count,
    }


class CreateRepoBody(BaseModel):
    source: str
    folder: str = ""
    tags: str = ""


async def _ensure_codegraph_ready(st: AppState) -> None:
    try:
        await st.codegraph.health_check()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


async def _run_import_repo(state: AppState, source: str, folder: str, tags: tuple[str, ...]) -> None:
    try:
        await state.repos().import_repo(source, folder=folder, tags=tags)
    except Exception:
        logger.exception("repo import failed for source=%s", source)


async def _run_reindex_repo(state: AppState, repo_id: str) -> None:
    try:
        await state.repos().reindex_repo(repo_id)
    except Exception:
        logger.exception("repo reindex failed for repo_id=%s", repo_id)
```

Add the routes (inside `create_app`):

```python
    @app.post("/api/repos", status_code=202)
    async def create_repo(
        request: Request,
        body: CreateRepoBody,
        background_tasks: BackgroundTasks,
    ):
        st: AppState = request.app.state.docgraph
        await _ensure_codegraph_ready(st)
        tag_list = tuple(t.strip() for t in body.tags.split(",") if t.strip())
        # Pre-validate quickly so the client gets a synchronous 400 on bad input.
        try:
            mgr = st.repos()
            if mgr._is_url(body.source):
                from docgraph.ingest.urls import validate_url
                validate_url(body.source)
            else:
                p = Path(body.source).resolve()
                if not p.is_dir():
                    raise ValueError(f"not a directory: {body.source}")
            if st.sqlite.get_repo_by_source(body.source) is not None:
                raise HTTPException(status_code=409, detail="repo already imported")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        background_tasks.add_task(
            _run_import_repo, st, body.source, body.folder, tag_list
        )
        # We don't have the repo_id yet (the bg task creates it). Return a marker.
        return {"status": "processing", "source": body.source}

    @app.get("/api/repos")
    async def list_repos(request: Request):
        st: AppState = request.app.state.docgraph
        return [_repo_to_json(r) for r in st.sqlite.list_repos()]

    @app.get("/api/repos/{repo_id}")
    async def get_repo(request: Request, repo_id: str):
        st: AppState = request.app.state.docgraph
        repo = st.sqlite.get_repo(repo_id)
        if repo is None:
            raise HTTPException(status_code=404, detail="not found")
        return _repo_to_json(repo)

    @app.post("/api/repos/{repo_id}/reindex", status_code=202)
    async def reindex_repo(
        request: Request,
        repo_id: str,
        background_tasks: BackgroundTasks,
    ):
        st: AppState = request.app.state.docgraph
        await _ensure_codegraph_ready(st)
        if st.sqlite.get_repo(repo_id) is None:
            raise HTTPException(status_code=404, detail="not found")
        background_tasks.add_task(_run_reindex_repo, st, repo_id)
        return {"repo_id": repo_id, "status": "processing"}

    @app.delete("/api/repos/{repo_id}")
    async def delete_repo(request: Request, repo_id: str):
        st: AppState = request.app.state.docgraph
        if st.sqlite.get_repo(repo_id) is None:
            raise HTTPException(status_code=404, detail="not found")
        cascaded = await st.repos().delete_repo(repo_id)
        return {"deleted": repo_id, "cascaded_docs": cascaded}
```

To make the test `test_create_repo_returns_202` work synchronously (the assertion needs `repo_id`), update the POST handler to materialize the row up-front. Replace the `POST /api/repos` body's return value with:

```python
        # Create the row immediately so the response carries a stable repo_id.
        # The background task will fill in progress/status.
        from docgraph.repo.manager import _repo_slug
        import uuid
        slug = _repo_slug(body.source)
        repo_id = f"repo_{uuid.uuid4().hex[:12]}"
        from docgraph.models import RepoRecord, DocumentStatus
        st.sqlite.insert_repo(RepoRecord(
            id=repo_id, name=slug.split("_", 1)[-1] if "_" in slug else slug,
            source_url=body.source,
            local_path=str(st.cfg.repos_dir / slug) if mgr._is_url(body.source) else str(Path(body.source).resolve()),
            folder=body.folder, tags=list(tag_list),
        ))
        st.sqlite.update_repo_progress(repo_id, 0, "Queued (0%)")
        background_tasks.add_task(
            _run_import_repo_existing, st, repo_id
        )
        return {"repo_id": repo_id, "status": "processing"}
```

And add `_run_import_repo_existing`:

```python
async def _run_import_repo_existing(state: AppState, repo_id: str) -> None:
    try:
        repo = state.sqlite.get_repo(repo_id)
        await state.repos().import_repo(
            repo.source_url, folder=repo.folder, tags=tuple(repo.tags),
            existing_repo_id=repo_id,
        )
    except Exception:
        logger.exception("repo import failed for repo_id=%s", repo_id)
```

This requires extending `RepoManager.import_repo` to accept `existing_repo_id`. Update it to use the supplied id when present and skip the duplicate insertion:

```python
    async def import_repo(
        self,
        source: str,
        *,
        folder: str = "",
        tags: tuple[str, ...] = (),
        existing_repo_id: str | None = None,
    ) -> str:
        ...
        if existing_repo_id is None:
            repo_id = f"repo_{uuid.uuid4().hex[:12]}"
            self._sqlite.insert_repo(repo)
        else:
            repo_id = existing_repo_id
        ...
```

(Adjust the existing `_run_import` / `RepoRecord` insertion path to skip insertion when `existing_repo_id` is set.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `poetry run pytest tests/web/test_repos_api.py tests/repo/test_manager.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add docgraph/web/app.py docgraph/repo/manager.py tests/web/test_repos_api.py
git commit -m "feat(api): /api/repos CRUD + codegraph health block"
```

---

## Task 9: MCP tools

**Files:**
- Modify: `docgraph/mcp/server.py`
- Create: `tests/mcp/test_code_tools.py`

**Interfaces:**
- Consumes: `RepoManager`, `CodegraphClient`, existing FastMCP decorator pattern.
- Produces: new tools `list_repos`, `import_repo`, `code_search`, `code_explore`, `code_callers`, `code_callees`, `code_trace`, `code_context`, `code_files`. Extend `search_documents` with optional `repo` arg.

- [ ] **Step 1: Write the failing tests**

Create `tests/mcp/test_code_tools.py`:

```python
import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from docgraph.config import Config
from docgraph.mcp.server import create_mcp_server
from docgraph.models import DocumentStatus, RepoRecord
from docgraph.web.deps import AppState


@pytest.mark.asyncio
async def test_list_repos_returns_records(tmp_data_dir):
    cfg = Config(data_dir=tmp_data_dir)
    state = AppState.create(cfg)
    state.sqlite.insert_repo(RepoRecord(
        id="repo_x", name="go-ethereum",
        local_path=str(tmp_data_dir / "x"),
        status=DocumentStatus.READY,
    ))
    mcp = create_mcp_server(state)
    tools = await mcp.list_tools()
    names = {t.name for t in tools}
    assert {"list_repos", "import_repo", "code_search", "code_explore",
            "code_callers", "code_callees", "code_trace", "code_context",
            "code_files"}.issubset(names)


@pytest.mark.asyncio
async def test_code_search_uses_resolved_repo(tmp_data_dir):
    cfg = Config(data_dir=tmp_data_dir)
    state = AppState.create(cfg)
    state.sqlite.insert_repo(RepoRecord(
        id="repo_x", name="go-ethereum",
        local_path=str(tmp_data_dir / "x"),
        status=DocumentStatus.READY,
    ))
    state.codegraph.run = AsyncMock(return_value={"results": [{"name": "Validator"}]})
    mcp = create_mcp_server(state)
    # FastMCP exposes a `call_tool` for testing; if not available, instantiate the
    # tool callable directly and await it.
    payload = await _call_tool(mcp, "code_search", {"query": "Validator", "repo": "go-ethereum"})
    body = json.loads(payload)
    assert body["results"][0]["name"] == "Validator"
    args, kwargs = state.codegraph.run.call_args
    assert args[0] == "search"
    assert "Validator" in args
    assert kwargs["repo_path"].endswith("x")


@pytest.mark.asyncio
async def test_code_search_returns_error_when_no_repo(tmp_data_dir):
    cfg = Config(data_dir=tmp_data_dir)
    state = AppState.create(cfg)
    mcp = create_mcp_server(state)
    payload = await _call_tool(mcp, "code_search", {"query": "X"})
    body = json.loads(payload)
    assert "error" in body
```

Add at the top of the test file:

```python
async def _call_tool(mcp, name: str, arguments: dict) -> str:
    """Bridge to FastMCP's tool runner."""
    for t in await mcp.list_tools():
        if t.name == name:
            result = await mcp.call_tool(name, arguments)
            # FastMCP returns a CallToolResult; the text payload is in content[0].text
            return result.content[0].text
    raise AssertionError(f"tool {name!r} not found")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `poetry run pytest tests/mcp/test_code_tools.py -v`
Expected: FAIL (tools don't exist yet).

- [ ] **Step 3: Implement MCP tools**

In `docgraph/mcp/server.py`, after the existing tool registrations add:

```python
    from pathlib import Path
    from docgraph.repo.codegraph_client import CodegraphNotInstalled

    repos_mgr = state.repos()

    @mcp.tool()
    async def list_repos() -> str:
        """List imported repositories."""
        return json.dumps([{
            "id": r.id, "name": r.name, "status": r.status.value,
            "progress_pct": r.progress_pct, "progress_phase": r.progress_phase,
            "doc_count": r.doc_count, "source_url": r.source_url,
        } for r in repos_mgr.list_repos()])

    @mcp.tool()
    async def import_repo(source: str, folder: str = "", tags: list[str] | None = None) -> str:
        """Clone (if URL) and index a repository via codegraph + DocGraph."""
        try:
            repo_id = await repos_mgr.import_repo(
                source, folder=folder, tags=tuple(tags or ()),
            )
        except (CodegraphNotInstalled, ValueError, RuntimeError) as exc:
            return json.dumps({"error": str(exc)})
        return json.dumps({"repo_id": repo_id, "status": "processing"})

    async def _run_codegraph(subcommand: str, repo: str | None, *args: str) -> str:
        target = repos_mgr.resolve(repo)
        if target is None:
            available = [r.name for r in repos_mgr.list_repos()]
            return json.dumps({
                "error": "specify repo (id or name)",
                "available": available,
            })
        if target.status != DocumentStatus.READY:
            return json.dumps({
                "error": "repo not ready", "status": target.status.value,
                "progress_pct": target.progress_pct,
            })
        try:
            result = await state.codegraph.run(
                subcommand, *args, repo_path=Path(target.local_path),
            )
        except CodegraphNotInstalled as exc:
            return json.dumps({
                "error": str(exc), "install_hint": state.codegraph.INSTALL_HINT,
            })
        except (TimeoutError, RuntimeError) as exc:
            return json.dumps({"error": str(exc)})
        return json.dumps({"repo": target.name, "result": result})

    @mcp.tool()
    async def code_search(query: str, repo: str | None = None) -> str:
        """Find a symbol by name/text in an imported repo."""
        return await _run_codegraph("search", repo, query)

    @mcp.tool()
    async def code_explore(symbols: list[str], repo: str | None = None) -> str:
        """Fetch source/context for multiple symbols in one call."""
        return await _run_codegraph("explore", repo, *symbols)

    @mcp.tool()
    async def code_callers(symbol: str, repo: str | None = None) -> str:
        """List callers of a symbol."""
        return await _run_codegraph("callers", repo, symbol)

    @mcp.tool()
    async def code_callees(symbol: str, repo: str | None = None) -> str:
        """List callees of a symbol."""
        return await _run_codegraph("callees", repo, symbol)

    @mcp.tool()
    async def code_trace(from_sym: str, to_sym: str, repo: str | None = None) -> str:
        """Trace a call path from one symbol to another."""
        return await _run_codegraph("trace", repo, from_sym, to_sym)

    @mcp.tool()
    async def code_context(query: str, repo: str | None = None) -> str:
        """Composed search + node + edges for a query."""
        return await _run_codegraph("context", repo, query)

    @mcp.tool()
    async def code_files(path: str = "", repo: str | None = None) -> str:
        """List files in an imported repo (optionally under a path)."""
        extra = (path,) if path else ()
        return await _run_codegraph("files", repo, *extra)
```

Extend the existing `search_documents` tool to accept `repo`:

```python
    @mcp.tool()
    async def search_documents(
        query: str,
        tags: list[str] | None = None,
        folder: str | None = None,
        top_k: int | None = None,
        repo: str | None = None,
    ) -> str:
        """Semantic search over uploaded documents. Returns relevant chunks with metadata."""
        target_repo = repos_mgr.resolve(repo) if repo else None
        repo_id = target_repo.id if target_repo else None
        try:
            results = await search_svc.search(
                query=query, top_k=top_k, folder=folder, tags=tags, repo_id=repo_id,
            )
        ...
```

(Update `SearchService.search` to forward `repo_id` to `ChromaStore.search`.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `poetry run pytest tests/mcp/ -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add docgraph/mcp/server.py docgraph/mcp/search.py tests/mcp/test_code_tools.py
git commit -m "feat(mcp): code_* tools + repo filter on search_documents"
```

---

## Task 10: CLI subcommands

**Files:**
- Create: `docgraph/cli_repos.py`
- Modify: `docgraph/cli.py`
- Test: append to `tests/test_e2e.py` or create `tests/test_cli_repos.py`

**Interfaces:**
- Consumes: `argparse.REMAINDER` pattern from existing `watch` subcommand (see `cli.py:219`).
- Produces:
  - `docgraph import-repo <source> [--folder F] [--tag T1,T2]`
  - `docgraph list-repos`
  - `docgraph delete-repo <repo_id_or_name>`
  - HTTP if server running on `cfg.web_host:cfg.web_port`; in-process otherwise

- [ ] **Step 1: Write the failing test**

Create `tests/test_cli_repos.py`:

```python
from unittest.mock import AsyncMock, patch

import pytest

from docgraph.cli_repos import run_repos_command
from docgraph.config import Config
from docgraph.web.deps import AppState


def test_list_repos_empty(tmp_data_dir, capsys):
    cfg = Config(data_dir=tmp_data_dir)
    AppState.create(cfg)
    exit_code = run_repos_command(["list-repos"], cfg, in_process=True)
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "0 repos" in out or "no repos" in out.lower()


def test_import_repo_local_in_process(tmp_data_dir):
    cfg = Config(data_dir=tmp_data_dir)
    state = AppState.create(cfg)
    local = tmp_data_dir / "src_repo"
    local.mkdir()
    (local / ".git").mkdir()
    (local / "README.md").write_text("# Hi")
    state.codegraph.init = AsyncMock()
    with patch(
        "docgraph.repo.manager.Indexer.index_markdown", AsyncMock()
    ):
        code = run_repos_command(
            ["import-repo", str(local)], cfg, in_process=True, state=state,
        )
    assert code == 0
    assert len(state.sqlite.list_repos()) == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `poetry run pytest tests/test_cli_repos.py -v`
Expected: FAIL with ModuleNotFoundError.

- [ ] **Step 3: Implement CLI module**

Create `docgraph/cli_repos.py`:

```python
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from typing import Optional

import httpx

from docgraph.config import Config
from docgraph.web.deps import AppState


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="docgraph", add_help=False)
    sub = parser.add_subparsers(dest="command")
    imp = sub.add_parser("import-repo")
    imp.add_argument("source")
    imp.add_argument("--folder", default="")
    imp.add_argument("--tag", default="")
    sub.add_parser("list-repos")
    delp = sub.add_parser("delete-repo")
    delp.add_argument("ref")
    return parser


def _http_base(cfg: Config) -> str:
    return f"http://{cfg.web_host}:{cfg.web_port}"


def _print_repos(repos: list[dict]) -> None:
    if not repos:
        print("0 repos imported.")
        return
    print(f"{len(repos)} repo(s):")
    for r in repos:
        print(
            f"  {r['id']}  {r['name']:<30} {r['status']:<10} "
            f"{r['progress_pct']:>3}%  docs={r['doc_count']}"
        )


def run_repos_command(
    argv: list[str],
    cfg: Config,
    *,
    in_process: bool = False,
    state: Optional[AppState] = None,
) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "import-repo":
        return _do_import(args, cfg, in_process, state)
    if args.command == "list-repos":
        return _do_list(cfg, in_process, state)
    if args.command == "delete-repo":
        return _do_delete(args, cfg, in_process, state)
    parser.print_help()
    return 2


def _is_server_up(cfg: Config) -> bool:
    try:
        r = httpx.get(_http_base(cfg) + "/api/health", timeout=1.5)
        return r.status_code == 200
    except Exception:
        return False


def _do_import(args, cfg, in_process, state) -> int:
    tags = tuple(t.strip() for t in args.tag.split(",") if t.strip())
    if not in_process and _is_server_up(cfg):
        body = {"source": args.source, "folder": args.folder, "tags": args.tag}
        r = httpx.post(_http_base(cfg) + "/api/repos", json=body, timeout=30)
        print(json.dumps(r.json(), indent=2))
        return 0 if r.status_code in (200, 202) else 1
    st = state or AppState.create(cfg)
    try:
        rid = asyncio.run(
            st.repos().import_repo(args.source, folder=args.folder, tags=tags)
        )
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"imported repo_id={rid}")
    return 0


def _do_list(cfg, in_process, state) -> int:
    if not in_process and _is_server_up(cfg):
        r = httpx.get(_http_base(cfg) + "/api/repos", timeout=5)
        _print_repos(r.json())
        return 0
    st = state or AppState.create(cfg)
    repos = st.sqlite.list_repos()
    _print_repos([{
        "id": r.id, "name": r.name, "status": r.status.value,
        "progress_pct": r.progress_pct, "doc_count": r.doc_count,
    } for r in repos])
    return 0


def _do_delete(args, cfg, in_process, state) -> int:
    st = state or AppState.create(cfg)
    repo = st.sqlite.get_repo(args.ref) or st.sqlite.get_repo_by_name(args.ref)
    if repo is None:
        print(f"not found: {args.ref}", file=sys.stderr)
        return 1
    if not in_process and _is_server_up(cfg):
        r = httpx.delete(_http_base(cfg) + f"/api/repos/{repo.id}", timeout=30)
        print(json.dumps(r.json(), indent=2))
        return 0 if r.status_code == 200 else 1
    cascaded = asyncio.run(st.repos().delete_repo(repo.id))
    print(f"deleted {repo.id} (cascaded {cascaded} docs)")
    return 0
```

In `docgraph/cli.py`, register the subcommands inside `main()` (next to the `watch` block):

```python
    p_repos = sub.add_parser("import-repo", add_help=False)
    p_repos.add_argument("rest", nargs=argparse.REMAINDER)
    p_list = sub.add_parser("list-repos")
    p_del = sub.add_parser("delete-repo", add_help=False)
    p_del.add_argument("rest", nargs=argparse.REMAINDER)
```

After parsing, dispatch:

```python
    if args.command in ("import-repo", "list-repos", "delete-repo"):
        from docgraph.cli_repos import run_repos_command
        cfg = load_config()
        cfg.ensure_dirs()
        rest = getattr(args, "rest", [])
        cli_argv = [args.command] + (rest or [])
        sys.exit(run_repos_command(cli_argv, cfg))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `poetry run pytest tests/test_cli_repos.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add docgraph/cli_repos.py docgraph/cli.py tests/test_cli_repos.py
git commit -m "feat(cli): import-repo, list-repos, delete-repo subcommands"
```

---

## Task 11: Integration e2e test

**Files:**
- Create: `tests/integration/__init__.py`, `tests/integration/test_repo_e2e.py`

**Interfaces:**
- Consumes: real `codegraph` binary on PATH (test skipped otherwise); real `git`.
- Produces: smoke test asserting clone + codegraph init + code_files returns non-empty result.

- [ ] **Step 1: Write the test**

Create `tests/integration/__init__.py` (empty).

Create `tests/integration/test_repo_e2e.py`:

```python
import shutil
from pathlib import Path

import pytest

from docgraph.config import Config
from docgraph.repo.codegraph_client import CodegraphClient
from docgraph.web.deps import AppState


REQUIRES_CG = pytest.mark.skipif(
    shutil.which("codegraph") is None, reason="codegraph CLI not on PATH"
)
REQUIRES_GIT = pytest.mark.skipif(
    shutil.which("git") is None, reason="git not on PATH"
)


@pytest.mark.integration
@REQUIRES_CG
@REQUIRES_GIT
@pytest.mark.asyncio
async def test_import_real_repo_and_query(tmp_data_dir):
    cfg = Config(data_dir=tmp_data_dir)
    state = AppState.create(cfg)
    repo_id = await state.repos().import_repo(
        "https://github.com/octocat/Hello-World"
    )
    repo = state.sqlite.get_repo(repo_id)
    assert repo.status.value == "ready"

    files = await state.codegraph.run("files", repo_path=Path(repo.local_path))
    assert files
```

- [ ] **Step 2: Run the test**

Run: `poetry run pytest tests/integration -v -m integration`
Expected: SKIP if `codegraph` not installed; PASS otherwise. Recording exit reason in stdout is fine.

- [ ] **Step 3: Commit**

```bash
git add tests/integration/__init__.py tests/integration/test_repo_e2e.py
git commit -m "test(integration): real clone + codegraph init smoke test"
```

---

## Task 12: Frontend types + API wrappers

**Files:**
- Modify: `frontend/src/types.ts`, `frontend/src/api.ts`

**Interfaces:**
- Consumes: existing `Document`, `HealthInfo` types.
- Produces:
  - `type Repo = { id; name; source_url; status; progress_pct; progress_phase; doc_count; ... }`
  - `HealthInfo.codegraph?: { ok: boolean; version: string; error: string }`
  - `fetchRepos`, `importRepo`, `reindexRepo`, `deleteRepo` in `api.ts`

- [ ] **Step 1: Add types**

In `frontend/src/types.ts`, append:

```typescript
export interface Repo {
  id: string;
  name: string;
  source_url: string;
  local_path: string;
  status: "processing" | "ready" | "error";
  progress_pct: number;
  progress_phase: string;
  error_message: string | null;
  folder: string;
  tags: string[];
  doc_count: number;
}
```

Extend the existing `HealthInfo`:

```typescript
export interface HealthInfo {
  status: string;
  ollama: { ok: boolean; error: string };
  embed_provider: string;
  mcp_sse_url: string;
  codegraph?: { ok: boolean; version: string; error: string };
}
```

- [ ] **Step 2: Add API wrappers**

In `frontend/src/api.ts`, append:

```typescript
export async function fetchRepos(): Promise<Repo[]> {
  const resp = await fetch("/api/repos");
  if (!resp.ok) throw new Error("Failed to load repos");
  return resp.json();
}

export interface ImportRepoResult {
  repo_id: string;
  status: string;
}

export async function importRepo(
  source: string,
  folder: string,
  tags: string,
): Promise<ImportRepoResult> {
  const resp = await fetch("/api/repos", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ source, folder, tags }),
  });
  if (!resp.ok) {
    const body = await resp.json().catch(() => ({}));
    throw new Error(typeof body.detail === "string" ? body.detail : "Import failed");
  }
  return resp.json();
}

export async function reindexRepo(id: string): Promise<void> {
  const resp = await fetch(`/api/repos/${id}/reindex`, { method: "POST" });
  if (!resp.ok) throw new Error("Re-index failed");
}

export async function deleteRepo(id: string): Promise<void> {
  const resp = await fetch(`/api/repos/${id}`, { method: "DELETE" });
  if (!resp.ok) throw new Error("Delete failed");
}
```

Import `Repo` from `./types` at the top of `api.ts`:

```typescript
import type { Document, HealthInfo, Repo } from "./types";
```

- [ ] **Step 3: Verify build**

Run: `cd frontend && npm run build`
Expected: PASS (`tsc --noEmit` succeeds; Vite emits to `docgraph/web/static/`).

- [ ] **Step 4: Commit**

```bash
git add frontend/src/types.ts frontend/src/api.ts docgraph/web/static
git commit -m "feat(ui): Repo types + API wrappers"
```

---

## Task 13: RepoImportSection component

**Files:**
- Create: `frontend/src/components/RepoImportSection.tsx`

**Interfaces:**
- Consumes: `importRepo` from `api.ts`; existing `SectionRule`, form/button CSS classes used by `LinkImportSection`.
- Produces: React component `<RepoImportSection onImported={() => void} />`.

- [ ] **Step 1: Implement component**

Create `frontend/src/components/RepoImportSection.tsx`:

```tsx
import { useState } from "react";
import { importRepo } from "../api";

interface Props {
  onImported: () => void;
}

export function RepoImportSection({ onImported }: Props) {
  const [source, setSource] = useState("");
  const [folder, setFolder] = useState("");
  const [tags, setTags] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!source.trim()) return;
    setBusy(true);
    setError(null);
    setSuccess(null);
    try {
      const result = await importRepo(source.trim(), folder.trim(), tags.trim());
      setSuccess(`Queued repo_id=${result.repo_id}`);
      setSource("");
      setTags("");
      setFolder("");
      onImported();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Import failed");
    } finally {
      setBusy(false);
    }
  };

  return (
    <section className="ingest-section">
      <h2 className="section-title">Repositories</h2>
      <p className="section-lede">
        Import a GitHub URL or an absolute local path. Code is indexed via codegraph
        for structural Q&amp;A; <code>*.md</code> files are vectorized for semantic search.
      </p>
      <form className="form-grid" onSubmit={submit}>
        <label>
          <span>Source</span>
          <input
            value={source}
            onChange={(e) => setSource(e.target.value)}
            placeholder="https://github.com/ethereum/go-ethereum"
            disabled={busy}
            required
          />
        </label>
        <label>
          <span>Folder (optional)</span>
          <input
            value={folder}
            onChange={(e) => setFolder(e.target.value)}
            placeholder="chains"
            disabled={busy}
          />
        </label>
        <label>
          <span>Tags (comma-separated)</span>
          <input
            value={tags}
            onChange={(e) => setTags(e.target.value)}
            placeholder="evm,core"
            disabled={busy}
          />
        </label>
        <button type="submit" disabled={busy || !source.trim()}>
          {busy ? "Importing…" : "Import repo"}
        </button>
      </form>
      {error && <p className="form-error">{error}</p>}
      {success && <p className="form-success">{success}</p>}
    </section>
  );
}
```

- [ ] **Step 2: Verify build**

Run: `cd frontend && npm run build`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add frontend/src/components/RepoImportSection.tsx docgraph/web/static
git commit -m "feat(ui): RepoImportSection form component"
```

---

## Task 14: RepoTable component

**Files:**
- Create: `frontend/src/components/RepoTable.tsx`

**Interfaces:**
- Consumes: `Repo` type; `reindexRepo`, `deleteRepo` from `api.ts`; existing `ProgressBar`, `StatusCell` components for visual parity with `DocumentTable`.
- Produces: `<RepoTable repos onChanged loading />` component.

- [ ] **Step 1: Implement component**

Create `frontend/src/components/RepoTable.tsx`:

```tsx
import { useState } from "react";
import { deleteRepo, reindexRepo } from "../api";
import type { Repo } from "../types";
import { ProgressBar } from "./ProgressBar";
import { StatusCell } from "./StatusCell";

interface Props {
  repos: Repo[];
  loading: boolean;
  onChanged: () => void;
}

export function RepoTable({ repos, loading, onChanged }: Props) {
  const [busyId, setBusyId] = useState<string | null>(null);

  const handleReindex = async (id: string) => {
    setBusyId(id);
    try {
      await reindexRepo(id);
      onChanged();
    } finally {
      setBusyId(null);
    }
  };

  const handleDelete = async (id: string) => {
    if (!confirm(`Delete repo ${id} and its docs?`)) return;
    setBusyId(id);
    try {
      await deleteRepo(id);
      onChanged();
    } finally {
      setBusyId(null);
    }
  };

  if (loading && repos.length === 0) {
    return <p className="muted">Loading repositories…</p>;
  }
  if (repos.length === 0) {
    return <p className="muted">No repositories imported yet.</p>;
  }

  return (
    <table className="data-table">
      <thead>
        <tr>
          <th>Repo</th>
          <th>Source</th>
          <th>Status</th>
          <th>Progress</th>
          <th>Docs</th>
          <th>Actions</th>
        </tr>
      </thead>
      <tbody>
        {repos.map((r) => (
          <tr key={r.id}>
            <td>
              <div>{r.name}</div>
              <div className="muted small">{r.id}</div>
            </td>
            <td className="truncate">{r.source_url || r.local_path}</td>
            <td>
              <StatusCell status={r.status} error={r.error_message} />
            </td>
            <td>
              {r.status === "processing" ? (
                <ProgressBar pct={r.progress_pct} label={r.progress_phase} />
              ) : (
                "—"
              )}
            </td>
            <td>{r.doc_count}</td>
            <td>
              <button
                disabled={busyId === r.id || r.status === "processing"}
                onClick={() => handleReindex(r.id)}
              >
                Re-index
              </button>
              <button
                disabled={busyId === r.id}
                onClick={() => handleDelete(r.id)}
              >
                Delete
              </button>
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}
```

- [ ] **Step 2: Verify build**

Run: `cd frontend && npm run build`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add frontend/src/components/RepoTable.tsx docgraph/web/static
git commit -m "feat(ui): RepoTable with reindex/delete actions"
```

---

## Task 15: App wiring + Header banner + DocumentTable source cell

**Files:**
- Modify: `frontend/src/App.tsx`, `frontend/src/components/Header.tsx`, `frontend/src/components/DocumentTable.tsx`

**Interfaces:**
- Consumes: `Repo` type; `fetchRepos`; `RepoImportSection`, `RepoTable`; existing polling pattern.
- Produces: integrated UI layout; codegraph health warning; doc source column.

- [ ] **Step 1: Wire App.tsx**

In `frontend/src/App.tsx`, replace the body to add repo state, fetch, and section:

```tsx
import { useCallback, useEffect, useMemo, useState } from "react";
import { fetchDocuments, fetchHealth, fetchRepos } from "./api";
import { DocumentTable } from "./components/DocumentTable";
import { Header } from "./components/Header";
import { LinkImportSection } from "./components/LinkImportSection";
import { RepoImportSection } from "./components/RepoImportSection";
import { RepoTable } from "./components/RepoTable";
import { UploadSection } from "./components/UploadSection";
import { SectionRule } from "./components/ui/SectionRule";
import type { Document, HealthInfo, Repo } from "./types";

export default function App() {
  const [health, setHealth] = useState<HealthInfo | null>(null);
  const [healthError, setHealthError] = useState<string | null>(null);
  const [documents, setDocuments] = useState<Document[]>([]);
  const [docsLoading, setDocsLoading] = useState(false);
  const [repos, setRepos] = useState<Repo[]>([]);
  const [reposLoading, setReposLoading] = useState(false);

  const refreshDocs = useCallback(async () => {
    setDocsLoading(true);
    try {
      const docs = await fetchDocuments();
      setDocuments(docs);
      return docs;
    } catch {
      return [];
    } finally {
      setDocsLoading(false);
    }
  }, []);

  const refreshRepos = useCallback(async () => {
    setReposLoading(true);
    try {
      const rs = await fetchRepos();
      setRepos(rs);
      return rs;
    } catch {
      return [];
    } finally {
      setReposLoading(false);
    }
  }, []);

  const refreshHealth = useCallback(async () => {
    try {
      setHealth(await fetchHealth());
      setHealthError(null);
    } catch (e) {
      setHealthError(e instanceof Error ? e.message : "Health unavailable");
    }
  }, []);

  const processingCount = useMemo(
    () =>
      documents.filter((d) => d.status === "processing").length +
      repos.filter((r) => r.status === "processing").length,
    [documents, repos],
  );

  useEffect(() => {
    void refreshHealth();
    const id = window.setInterval(() => void refreshHealth(), 30_000);
    return () => window.clearInterval(id);
  }, [refreshHealth]);

  useEffect(() => {
    let cancelled = false;
    let timer = 0;
    const tick = async () => {
      if (cancelled) return;
      const [docs, rs] = await Promise.all([refreshDocs(), refreshRepos()]);
      if (cancelled) return;
      const busy =
        docs.some((d) => d.status === "processing") ||
        rs.some((r) => r.status === "processing");
      timer = window.setTimeout(tick, busy ? 1500 : 5000);
    };
    void tick();
    return () => {
      cancelled = true;
      window.clearTimeout(timer);
    };
  }, [refreshDocs, refreshRepos]);

  return (
    <div className="page">
      <a href="#main" className="skip-link">Skip to content</a>
      <div className="page-texture" aria-hidden="true" />
      <div className="container">
        <Header
          health={health}
          healthError={healthError}
          documentCount={documents.length}
          processingCount={processingCount}
        />
        <SectionRule ultra />
        <main id="main">
          <UploadSection onUploaded={() => void refreshDocs()} />
          <SectionRule />
          <LinkImportSection onImported={() => void refreshDocs()} />
          <SectionRule />
          <RepoImportSection onImported={() => void refreshRepos()} />
          <SectionRule thick />
          <RepoTable
            repos={repos}
            loading={reposLoading}
            onChanged={() => {
              void refreshRepos();
              void refreshDocs();
            }}
          />
          <SectionRule thick />
          <DocumentTable
            documents={documents}
            repos={repos}
            loading={docsLoading}
            onChanged={() => void refreshDocs()}
          />
        </main>
        <SectionRule thick />
        <footer className="label-mono" style={{ paddingTop: "1.5rem" }}>
          DocGraph · Monochrome editorial interface
        </footer>
      </div>
    </div>
  );
}
```

- [ ] **Step 2: Header banner**

In `frontend/src/components/Header.tsx`, render a warning when `health.codegraph?.ok === false`. Add inside the existing header markup (next to the ollama banner if present):

```tsx
{health?.codegraph && !health.codegraph.ok && (
  <div className="banner banner-error">
    <strong>codegraph CLI not found.</strong>{" "}
    Repositories will be unavailable until you install it:{" "}
    <code>curl -fsSL https://raw.githubusercontent.com/colbymchenry/codegraph/main/install.sh | sh</code>
  </div>
)}
```

- [ ] **Step 3: Document table source cell**

In `frontend/src/components/DocumentTable.tsx`, accept a `repos` prop and render a "Source" cell that links a `doc.repo_id` to the matching `repo.name`:

```tsx
interface Props {
  documents: Document[];
  repos?: Repo[];
  loading: boolean;
  onChanged: () => void;
}

// inside the row mapping, after the filename cell:
<td>
  {doc.repo_id
    ? (repos?.find((r) => r.id === doc.repo_id)?.name ?? doc.repo_id)
    : "—"}
</td>
```

Add `repos` to the props and the table header:

```tsx
<th>Source</th>
```

- [ ] **Step 4: Verify build**

Run: `cd frontend && npm run build`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/App.tsx frontend/src/components/Header.tsx frontend/src/components/DocumentTable.tsx docgraph/web/static
git commit -m "feat(ui): wire Repositories section + codegraph banner + doc source cell"
```

---

## Task 16: Dockerfile + docker-compose

**Files:**
- Modify: `Dockerfile`, `docker-compose.yml`

**Interfaces:**
- Consumes: existing multi-stage Dockerfile, runtime stage.
- Produces: `codegraph` binary on `$PATH` in the runtime image; a named volume `repos` mounted to `/data/repos`.

- [ ] **Step 1: Install codegraph in runtime stage**

In `Dockerfile`, in the runtime stage (after the existing `RUN apt-get install … curl` block), add:

```dockerfile
RUN curl -fsSL https://raw.githubusercontent.com/colbymchenry/codegraph/main/install.sh \
      | sh \
    && codegraph --version
```

- [ ] **Step 2: Add repos volume**

In `docker-compose.yml`, ensure the `docgraph` service has the existing `/data` mount but also document that the repos subdirectory may grow large. Add a comment block near the existing `volumes` section explaining that `~/.docgraph/repos/` lives at `/data/repos` inside the container and recommending at least 5 GB free space when importing large public repos. No new volume name is needed since `repos/` is under `/data`.

```yaml
    volumes:
      - docgraph_data:/data
      # Repos imported via the UI / `docgraph import-repo` are cloned to
      # /data/repos. A go-ethereum-sized repo is ~500 MB on disk; ensure the
      # docgraph_data volume has at least 5 GB free if you plan to import
      # large public repositories.
```

- [ ] **Step 3: Smoke build**

Run: `docker build -t docgraph:repos-check .`
Expected: PASS (codegraph install completes and `codegraph --version` prints during the build).

If you don't have Docker locally, mark this step skipped with a note in the commit message.

- [ ] **Step 4: Commit**

```bash
git add Dockerfile docker-compose.yml
git commit -m "build(docker): install codegraph in runtime + document repos volume"
```

---

## Task 17: README documentation

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Add Repositories section**

In `README.md`, insert a new section right after `## MCP Tools` (and before `## Configuration`):

```markdown
## Repositories

Import a GitHub URL or an absolute local path to make a repo's structural
code intelligence available to Cursor via the same MCP entry point.

```bash
# from the running web UI: paste URL into the "Repositories" form, or
docgraph import-repo https://github.com/ethereum/go-ethereum --folder chains --tag evm
docgraph list-repos
docgraph delete-repo go-ethereum
```

Each import:

1. Clones the default branch with `git clone --depth 1` into
   `~/.docgraph/repos/<owner>_<name>/`.
2. Runs `codegraph init` for AST + FTS5 code intelligence (`codegraph` CLI
   must be on `$PATH`; install it with
   `curl -fsSL https://raw.githubusercontent.com/colbymchenry/codegraph/main/install.sh | sh`).
3. Vectorizes every `*.md` file (README, docs/) through the existing
   Chroma pipeline so semantic search keeps working alongside structural queries.

The MCP server exposes new tools alongside the existing ones:

| Tool | Use |
|------|-----|
| `list_repos` | Show indexed repositories |
| `import_repo` | Queue an import from MCP |
| `code_search` | Find a symbol by name/text |
| `code_explore` | Source of multiple symbols at once |
| `code_callers` / `code_callees` | Caller / callee lists |
| `code_trace` | Call path between two symbols |
| `code_context` | Composed search + node + edges |
| `code_files` | List files in the repo |

`search_documents` accepts an optional `repo` argument that scopes vector
search to a single repository.
```

In the env-var table further down, append:

```markdown
| `DOCGRAPH_REPOS_DIR` | `<data_dir>/repos` | Where cloned repos live |
| `DOCGRAPH_CODEGRAPH_BIN` | `codegraph` | Path or name of the codegraph CLI |
| `DOCGRAPH_CODEGRAPH_INIT_TIMEOUT_SEC` | `600` | Per-repo init timeout |
| `DOCGRAPH_CODEGRAPH_QUERY_TIMEOUT_SEC` | `30` | Per-query timeout |
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs(readme): repo import + codegraph integration"
```

---

## Self-review

### Spec coverage

- §1 Goal → Tasks 5, 8, 9 (manager + REST + MCP).
- §2 Non-goals → enforced by absence of features (no auth flow, no scheduled refresh, no fusion).
- §3 Approach A → Task 6 propagates `repo_id` into Chroma so existing `search_documents` filter works.
- §4 Architecture → Tasks 4, 5, 7 set up CodegraphClient, RepoManager, AppState wiring.
- §5 Components → Tasks 1, 2, 3, 4, 5, 7 (each component is a task).
- §6 MCP tool surface → Task 9 (every tool listed).
- §7 REST API → Task 8 (every endpoint + health field).
- §8 Web UI → Tasks 12, 13, 14, 15.
- §9 CLI → Task 10.
- §10 Data flow (import) → covered by `RepoManager._run_import` in Task 5 and bg task in Task 8.
- §11 Data flow (query) → Task 9 `_run_codegraph` helper.
- §12 Error handling / security / concurrency → CodegraphClient timeout/kill (Task 4), per-repo lock (Task 5), 503 on missing CLI (Task 8), watcher exclusion (Task 5).
- §13 Testing → unit tests in Tasks 3, 4, 5, 6, 7, 8, 9, 10; integration in Task 11.
- §14 Docker → Task 16.
- §15 Out of scope → no task implements these (correct).

### Placeholder scan

No "TBD" / "TODO" / "implement later" remain. Every step shows the actual code or command.

### Type consistency

- `RepoRecord` fields used in Task 2 are the same fields read in Task 3's `_row_to_repo` and serialized in Task 8's `_repo_to_json` and Task 12's TypeScript `Repo` type.
- `CodegraphClient.run(subcommand, *args, repo_path, timeout)` signature is identical across the implementation in Task 4 and the call sites in Tasks 9 and 11.
- `RepoManager.import_repo(source, *, folder, tags, existing_repo_id=None)` is the same across Tasks 5, 8, 10.
- `ChromaStore.search(..., repo_id=None)` is the same across Tasks 6, 9.

---

## Execution handoff

**Plan complete and saved to `docs/superpowers/plans/2026-06-30-repo-codegraph-integration.md`. Two execution options:**

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration.

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints.

**Which approach?**
