# Repo Import via Codegraph — Design

**Status:** Brainstorming complete — pending user review, then implementation plan
**Date:** 2026-06-30
**Author:** brainstorm session, DocGraph maintainer
**Scope:** Local single-user DocGraph. New first-class concept "repository" backed by `colbymchenry/codegraph` for AST-aware code intelligence, layered on top of the existing document RAG pipeline.

---

## 1. Goal

Replace the current "convert repo to a single repomix bundle and upload" workflow with a native `Import Repo` flow. After one click (or one CLI command) a GitHub repo or local path is:

1. Cloned into DocGraph's data dir (if remote)
2. Indexed by `codegraph` into an AST + FTS5 knowledge graph
3. Its `*.md` files chunked and embedded into the existing Chroma vector store

After import, Cursor talks to a single MCP server (DocGraph) and gets both worlds — structural code Q&A via `code_*` tools (subprocess into codegraph CLI) and natural-language Q&A on docs via the existing `search_documents`.

## 2. Non-goals

- Authenticated/private repos (HTTPS token, OAuth). SSH-key-via-host-git may incidentally work; no UI for credentials.
- Multi-branch / specific refs. Default branch only.
- Auto-pull / scheduled refresh. Reindex is manual.
- Cross-repo result fusion (auto-merge results from N repos in one tool call).
- Auto-installing `codegraph` when missing. We surface a clear install hint instead.
- Replacing the existing file/URL upload pipelines.
- Custom code-aware chunking inside DocGraph. We delegate to `codegraph` rather than reinvent tree-sitter.

## 3. Approach choice

Three approaches were considered:

- **A. Dual indexing (chosen).** Code → codegraph (per repo). `*.md` → Chroma via existing pipeline. MCP exposes both `code_*` tools and the existing `search_documents`. Cursor picks the right tool from descriptions; no intent classifier inside DocGraph.
- **B. Codegraph-only for repos.** Skip Chroma for repos; rely on codegraph FTS5 for README/docs text. Simpler but weaker NL search on prose.
- **C. SourceAdapter refactor.** Generalize ingest into `FileAdapter | UrlAdapter | RepoAdapter`. Cleanest but high refactor risk on a stable pipeline.

A was chosen because vector search materially beats FTS5 for natural-language Q&A on README/whitepapers, the two indexing paths have well-bounded responsibilities, and we avoid touching the stable ingest path.

## 4. Architecture overview

```
Cursor (MCP client)
   │ SSE / stdio
   ▼
DocGraph MCP server
   │
   ├── Existing tools
   │     - search_documents (+ optional repo filter)
   │     - list_documents
   │     - get_document
   │
   └── New repo tools
         - list_repos
         - import_repo
         - code_search
         - code_explore
         - code_callers
         - code_callees
         - code_trace
         - code_context
         - code_files
                  │
                  ▼
          CodegraphClient (subprocess wrapper, per-repo asyncio.Lock)
                  │
                  ▼
          ~/.docgraph/repos/<owner>_<name>/
              ├── .git/
              ├── .codegraph/        (codegraph-managed)
              └── src/…              (source tree)
```

On-disk layout for an imported repo:

- Clone: `<data_dir>/repos/<owner>_<name>/`
- Codegraph index: `<data_dir>/repos/<owner>_<name>/.codegraph/`
- Chroma chunks for the repo's `*.md`: stored in the existing collection, with metadata `repo_id` set
- SQLite: new `repos` row, plus per-md `documents` rows linked via new `repo_id` column

A single MCP entry is intentional: Cursor only configures one MCP (docgraph), and routing is by tool selection rather than an internal classifier — which we explicitly reject as another failure point.

## 5. Components

### 5.1 `docgraph/repo/codegraph_client.py` — `CodegraphClient`

Thin async wrapper around the `codegraph` CLI.

- `__init__(bin: str = "codegraph", query_timeout_sec=30, init_timeout_sec=600)`
- `async health_check()` — runs `codegraph --version`; raises `RuntimeError` with install hint if the binary is missing or fails.
- `async init(repo_path: Path, *, progress_cb=None)` — `codegraph init` in `repo_path`. Stdout streamed to logger at DEBUG; non-zero exit raises with stderr captured. `progress_cb(phase: str)` invoked on heartbeat (every 5s) so callers can keep DB progress fresh.
- `async run(subcommand: str, *args, repo_path: Path, timeout: float | None = None) -> dict | list | str` — generic dispatcher. Prefers `--json` flag; falls back to text if subcommand lacks it. Times out using `asyncio.wait_for`.
- `async aclose()` — no resource state today, kept for symmetry with `OllamaEmbedder.aclose`.
- Concurrency: caller is responsible for serializing init/query on the same repo (see `RepoManager`'s per-repo lock).
- All subprocess invocations use `asyncio.create_subprocess_exec` with arg lists (no `shell=True`).

### 5.2 `docgraph/repo/manager.py` — `RepoManager`

Single-repo lifecycle. Holds a dict of `asyncio.Lock` keyed by `repo_id` to serialize init / reindex / delete per repo.

- `async import_repo(source: str, folder: str = "", tags: tuple[str, ...] = ()) -> str`
  - Detects URL vs local absolute path
  - For URL: validate via existing `validate_url` (blocks SSRF), derive `<owner>_<name>` slug, `git clone --depth 1` into `cfg.repos_dir / slug`
  - For path: must be absolute, must exist, must be a directory; no clone
  - Insert `RepoRecord` (status=processing, pct=0)
  - Run `codegraph_client.init(...)` under the per-repo lock with heartbeat → DB updates
  - Walk `*.md` files, skipping any path whose components include `.git`, `node_modules`, `vendor`, `dist`, `build`, `target`, `__pycache__`, `.venv`, `.next`, or `.codegraph`
  - For each md: create `DocumentRecord(repo_id=..., source_type=FILE)`, hand off to existing `Indexer.index_markdown`
  - Finalize: `update_repo_status(READY)`, `doc_count` set
- `async reindex_repo(repo_id)` — re-runs codegraph init and re-indexes md files. Existing repo md `documents` are deleted from SQLite + Chroma first to avoid stale chunks.
- `async delete_repo(repo_id)` — cancels in-flight tasks if any, deletes `repos_dir/<slug>` tree, deletes all `documents` rows with that `repo_id` and their Chroma chunks, then deletes the repo row.
- `list_repos() / get_repo()` — read-through to `SQLiteStore`.
- `resolve(ref: str | None) -> RepoRecord | None`
  - exact `repo_id` match, then case-insensitive `name` match
  - if `ref is None` and exactly one READY repo exists, return it
  - else `None` (caller emits the multi-repo / not-found error)

### 5.3 `docgraph/models.py` additions

```python
@dataclass
class RepoRecord:
    id: str
    name: str                           # e.g. "go-ethereum"
    source_url: str = ""                # empty for pure local-path imports
    local_path: str = ""                # absolute path on disk
    status: DocumentStatus = DocumentStatus.PROCESSING
    progress_pct: int = 0
    progress_phase: str = ""
    error_message: Optional[str] = None
    folder: str = ""
    tags: list[str] = field(default_factory=list)
    doc_count: int = 0
```

`DocumentRecord` gains `repo_id: str = ""`.

### 5.4 `docgraph/store/sqlite.py` additions

New table:

```sql
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
```

Migration for `documents`:

```sql
ALTER TABLE documents ADD COLUMN repo_id TEXT NOT NULL DEFAULT '';
CREATE INDEX IF NOT EXISTS idx_documents_repo_id ON documents(repo_id);
```

New methods: `insert_repo`, `get_repo`, `list_repos`, `update_repo_progress`, `update_repo_status`, `delete_repo`, `list_documents_by_repo`.

### 5.5 `docgraph/web/deps.py` — `AppState`

Add `codegraph: CodegraphClient` (singleton, init at startup, `aclose` at shutdown) and a `repos()` factory mirroring the existing `indexer()` pattern.

### 5.6 `docgraph/config.py` additions

```python
repos_dir: Path                            # default: data_dir / "repos"
codegraph_bin: str = "codegraph"
codegraph_init_timeout_sec: int = 600
codegraph_query_timeout_sec: int = 30
```

Loaded from `repos` YAML section and `DOCGRAPH_CODEGRAPH_BIN`, `DOCGRAPH_REPOS_DIR` env vars.

## 6. MCP tool surface

All new tools live in `docgraph/mcp/server.py`. Schemas use `repo` as a free-form identifier (id or name).

| Tool          | Args                                    | Behavior                                                                |
|---------------|------------------------------------------|-------------------------------------------------------------------------|
| `list_repos`  | —                                        | `[{id, name, status, progress_pct, progress_phase, doc_count, ...}]`     |
| `import_repo` | `source, folder?, tags?`                 | Non-blocking; returns `{repo_id, status: "processing"}`                  |
| `code_search` | `query, repo?`                           | Wraps `codegraph search <query> --json`                                  |
| `code_explore`| `symbols: list[str], repo?`              | Wraps `codegraph explore <symbols…> --json`                              |
| `code_callers`| `symbol, repo?`                          | Wraps `codegraph callers <symbol> --json`                                |
| `code_callees`| `symbol, repo?`                          | Wraps `codegraph callees <symbol> --json`                                |
| `code_trace`  | `from_sym, to_sym, repo?`                | Wraps `codegraph trace <from> <to> --json`                               |
| `code_context`| `query, repo?`                           | Wraps `codegraph context <query> --json` (composed search + node + edges)|
| `code_files`  | `path?, repo?`                           | Wraps `codegraph files [<path>] --json`                                  |

Repo resolution order: `repo_id` → name → if `None` and exactly one READY repo, use it → else return `{error, available: [...]}`. No magic cross-repo merging in v1.

JSON shape from codegraph is passed through verbatim. Rationale: preserving codegraph's schema means the user (and the LLM) reasons in the same vocabulary across direct codegraph MCP and docgraph wrapped tools. If codegraph changes shape, our wrappers don't lie about it.

`search_documents` gains an optional `repo` arg (filters Chroma metadata where `repo_id` matches). Existing folder/tag filters continue to work and compose.

## 7. REST API

| Method | Path                                | Body / Query                       | Response                                   |
|--------|-------------------------------------|------------------------------------|--------------------------------------------|
| POST   | `/api/repos`                        | `{source, folder?, tags?}`         | `202 {repo_id, status: "processing"}`      |
| GET    | `/api/repos`                        | `?folder=`                         | `[RepoRecord]`                             |
| GET    | `/api/repos/{repo_id}`              | —                                  | `RepoRecord`                               |
| POST   | `/api/repos/{repo_id}/reindex`      | —                                  | `202 {repo_id, status: "processing"}`      |
| DELETE | `/api/repos/{repo_id}`              | —                                  | `{deleted: repo_id, cascaded_docs: N}`     |

`/api/health` grows a `codegraph` block:

```json
{ "codegraph": { "ok": true, "version": "0.x.y" } }
```

## 8. Web UI (`frontend/src/`)

- `components/RepoImportSection.tsx` — form: source (URL or absolute path), folder, tags. Submit → `POST /api/repos`.
- `components/RepoTable.tsx` — list of repos with name, source, status pill, `ProgressBar`, doc_count, reindex/delete actions.
- `App.tsx` — add new section "Repositories" between `LinkImportSection` and `DocumentTable`. Polling tick already adapts on processing rows; extend the same logic to repos.
- `DocumentTable` — add a small "Source" cell that renders `repo.name` linked to the repos table if `doc.repo_id` is set.
- `Header` — if `health.codegraph.ok === false`, render an inline warning with the install command.

## 9. CLI

```
docgraph import-repo <source> [--folder F] [--tag T1,T2]
docgraph list-repos
docgraph delete-repo <repo_id_or_name>
```

CLI checks `_port_available(cfg.web_host, cfg.web_port)`:

- Port in use → assume the server is running; talk to it via HTTP.
- Port free → run the operation in-process (same `AppState.create`) so single-shot CLI use works without `docgraph serve`.

## 10. Data flow — import a repo

```
POST /api/repos {source}
  → uuid → INSERT repos (status=processing, pct=0)
  → 202 {repo_id}
  → BackgroundTasks.add_task(_run_import_repo)

_run_import_repo (under per-repo lock):
   0–5%   "Validating source"
   5–30%  "Cloning <url>"             (git clone --depth 1)
  30–80%  "Building code index"       (codegraph init; heartbeat phase update every 5s)
  80–95%  "Indexing docs (N md files)" (per-md: Indexer.index_markdown with repo_id set)
  95–100% "Finalizing"                 (update doc_count, status=READY)
  exception → status=ERROR with preserved phase, logger.exception
```

## 11. Data flow — code query via MCP

```
Cursor → code_search("Validator", repo="go-ethereum")
       → RepoManager.resolve("go-ethereum") → RepoRecord(local_path=…)
       → CodegraphClient.run("search", "Validator", repo_path=local_path, timeout=30)
            asyncio.create_subprocess_exec("codegraph", "search", "Validator", "--json", cwd=local_path)
       → parsed JSON returned verbatim to Cursor
```

If the repo is still processing, all `code_*` tools return `{error: "repo not ready", status, progress_pct}` rather than blocking.

## 12. Error handling, security, concurrency

### Codegraph absent
- Startup health check probes once and caches; UI health endpoint reflects current state.
- `POST /api/repos` and `POST /api/repos/{id}/reindex` return `503` with install command in the message.
- `code_*` MCP tools return `{error, install_hint}` JSON instead of crashing.
- Pre-existing repo rows continue to be visible; their queries fail with the same install hint.

### Subprocess hygiene
- Always arg-list form (no `shell=True`).
- Init phase: SIGTERM on cancel; SIGKILL after a 5-second grace so codegraph can flush its SQLite.
- DELETE on a processing repo sets `cancel_requested=1` in the row; the worker polls and terminates.

### SSRF / path safety
- URL imports go through existing `validate_url` (blocks localhost, private IPv4, link-local).
- Local-path imports require an absolute path and must exist as a directory; otherwise reject. No whitelist beyond that — single-user local tool.

### Concurrency
- Per-repo `asyncio.Lock` serializes init/reindex/delete on the same repo.
- Across repos: parallel imports are allowed; FastAPI BackgroundTasks already supports it.
- Repeated `POST /api/repos` with the same source URL → `409 {existing_repo_id}` and a hint to use reindex.

### File watcher coordination
- The file watcher (introduced in commit `cb21ee1`) must exclude `cfg.repos_dir` recursively. Otherwise codegraph's writes and `git clone` would trigger a re-index storm on every imported repo's source files.
- Codegraph owns "watch and update source index"; DocGraph re-indexes md files only on explicit `reindex_repo`. This boundary is documented in `repo/manager.py`.

### Resource bounds
- No hard cap on repo size in v1 (user explicitly chose this trade-off).
- Heartbeat ensures the UI never looks frozen.
- `max_chunks_per_doc` still applies per individual `.md` file (a giant single doc would still trip it).

## 13. Testing

Unit
- `tests/repo/test_codegraph_client.py` — mock `asyncio.create_subprocess_exec`; assert argv, JSON parsing, timeout path, missing-binary error message.
- `tests/repo/test_manager.py` — fake codegraph client, fake git via a fixture dir containing a `.git` shell; assert state machine transitions, cascading delete of repo + docs + Chroma chunks.
- `tests/web/test_repos_api.py` — `TestClient` covering all 5 endpoints (including `409` on duplicate source and `503` on missing codegraph).
- `tests/mcp/test_code_tools.py` — assert repo resolution order (id, name, None-with-one, None-with-many), JSON passthrough.

Integration (marker `integration`; skipped when `codegraph` is not on PATH)
- `tests/integration/test_repo_e2e.py` — clone a tiny public repo (e.g. `octocat/Hello-World`), run real `codegraph init`, query `code_files` and `code_search`, assert non-empty results.

## 14. Deployment (Docker)

- Multi-stage `Dockerfile`: existing Python layer plus a new layer that runs the codegraph install script (`curl … install.sh | sh`). Final image keeps the codegraph binary on `$PATH`.
- `docker-compose.yml` — add a named volume for `~/.docgraph/repos/` separate from chroma/sqlite, since repo clones can be hundreds of MB each.
- README Docker section: document the new volume, recommend ≥5GB free space when importing repos at the scale of `go-ethereum` or `solana`.

## 15. Out of scope (v1)

- Authenticated repo imports (HTTPS token, OAuth, GitHub App).
- Branch/tag/commit selection.
- Scheduled / push-driven repo updates.
- Cross-repo result fusion in MCP.
- Auto-install of codegraph CLI when missing.
- Non-English UI strings.

## 16. Open questions tracked into implementation plan

None — the brainstorm closed each section with a binary confirmation. Items currently flagged "v1 boundary" (see §15) become candidates for v2 specs once usage data accumulates.
