import argparse
import asyncio
import logging
import sys
import threading

import uvicorn

from boostmcp.config import load_config
from boostmcp.web.app import create_app


def _start_web(cfg) -> None:
    app = create_app(cfg)
    uvicorn.run(
        app,
        host=cfg.web_host,
        port=cfg.web_port,
        log_level="warning",
    )


def _run_serve() -> None:
    cfg = load_config()
    cfg.ensure_dirs()
    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger("boostmcp")

    web_thread = threading.Thread(
        target=_start_web, args=(cfg,), daemon=True, name="boostmcp-web"
    )
    web_thread.start()
    log.info("Web UI at http://%s:%s", cfg.web_host, cfg.web_port)

    from boostmcp.embed.factory import create_embedder

    embedder = create_embedder(cfg)
    try:
        asyncio.run(embedder.health_check())
        log.info("Embedding provider OK (%s)", cfg.embed_provider)
    except Exception as exc:
        log.warning("Embedding health check failed: %s", exc)

    from boostmcp.mcp.server import create_mcp_server
    from boostmcp.web.deps import AppState

    state = AppState.create(cfg)
    mcp = create_mcp_server(state)
    mcp.run(transport="stdio")


def main() -> None:
    parser = argparse.ArgumentParser(prog="boostmcp")
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("serve", help="Start MCP + Web UI server")
    args = parser.parse_args()
    if args.command == "serve":
        _run_serve()
        return
    parser.print_help()
    sys.exit(1)


if __name__ == "__main__":
    main()
