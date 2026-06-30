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


def _is_server_up(cfg: Config) -> bool:
    try:
        r = httpx.get(_http_base(cfg) + "/api/health", timeout=1.5)
        return r.status_code == 200
    except Exception:
        return False


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
