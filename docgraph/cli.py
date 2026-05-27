import argparse
import asyncio
import logging
import socket
import sys
import threading

import uvicorn

from docgraph.config import load_config
from docgraph.web.app import create_app
from docgraph.web.deps import AppState


def _port_available(host: str, port: int) -> bool:
    """Return True if the port is free on the given host."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind((host, port))
        except OSError:
            return False
    return True


def _log_startup(cfg, log: logging.Logger, *, mcp_sse: bool) -> None:
    log.info("Web UI at http://%s:%s", cfg.web_host, cfg.web_port)
    if mcp_sse:
        log.info("MCP SSE at http://%s:%s/mcp/sse", cfg.web_host, cfg.web_port)
        log.info(
            "Cursor mcp.json: npx -y mcp-remote@latest http://%s:%s/mcp/sse",
            cfg.web_host,
            cfg.web_port,
        )


def _health_check(cfg, log: logging.Logger) -> None:
    from docgraph.embed.factory import create_embedder

    embedder = create_embedder(cfg)
    try:
        asyncio.run(embedder.health_check())
        log.info("Embedding provider OK (%s)", cfg.embed_provider)
    except Exception as exc:
        log.warning("Embedding health check failed: %s", exc)


def _run_http(cfg) -> None:
    """Run Web UI + MCP SSE on HTTP (server runs independently)."""
    app = create_app(cfg, mount_mcp=True)
    uvicorn.run(
        app,
        host=cfg.web_host,
        port=cfg.web_port,
        log_level="info",
    )


def _run_stdio(cfg) -> None:
    """Legacy: Web UI in background thread + MCP stdio for Cursor direct launch."""
    log = logging.getLogger("docgraph")
    state = AppState.create(cfg)
    app = create_app(cfg, state=state, mount_mcp=False)

    if not _port_available(cfg.web_host, cfg.web_port):
        log.warning(
            "Web UI port %s:%s is already in use; another DocGraph may be running. "
            "MCP stdio will still start, but the Web UI will be unavailable.",
            cfg.web_host,
            cfg.web_port,
        )

    web_thread = threading.Thread(
        target=lambda: uvicorn.run(
            app, host=cfg.web_host, port=cfg.web_port, log_level="warning"
        ),
        daemon=True,
        name="docgraph-web",
    )
    web_thread.start()
    _log_startup(cfg, log, mcp_sse=False)

    from docgraph.mcp.server import create_mcp_server

    mcp = create_mcp_server(state)
    mcp.run(transport="stdio")


def _run_serve(stdio: bool) -> None:
    cfg = load_config()
    cfg.ensure_dirs()
    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger("docgraph")

    _health_check(cfg, log)

    if stdio:
        _run_stdio(cfg)
    else:
        _log_startup(cfg, log, mcp_sse=True)
        _run_http(cfg)


def main() -> None:
    parser = argparse.ArgumentParser(prog="docgraph")
    sub = parser.add_subparsers(dest="command")
    serve_parser = sub.add_parser("serve", help="Start Web UI + MCP server")
    serve_parser.add_argument(
        "--stdio",
        action="store_true",
        help="Use MCP stdio instead of HTTP SSE (Cursor launches process directly)",
    )
    args = parser.parse_args()
    if args.command == "serve":
        _run_serve(stdio=args.stdio)
        return
    parser.print_help()
    sys.exit(1)


if __name__ == "__main__":
    main()
