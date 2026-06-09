from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Iterator

from docgraph.models import WatchedDirRecord
from docgraph.watch.ignore import IgnoreMatcher
from docgraph.watch.types import WatchEvent

logger = logging.getLogger(__name__)


def walk_watched_dir(wd: WatchedDirRecord, matcher: IgnoreMatcher) -> Iterator[Path]:
    root = Path(wd.path)
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        dp = Path(dirpath)
        # Prune ignored dirs in-place (saves recursion).
        dirnames[:] = [
            d for d in dirnames
            if d not in {".git", "node_modules", "__pycache__", ".venv", ".hg", ".svn", "venv", ".tox"}
        ]
        for fn in filenames:
            p = dp / fn
            if matcher.should_ignore(p):
                continue
            yield p


async def reconcile_dir(manager, wd: WatchedDirRecord) -> None:
    """Disk-vs-DB delta scan. Enqueues UPSERT / DELETE events directly (bypass debounce)."""
    matcher = manager._ignore_matchers.get(wd.id) or IgnoreMatcher(wd)
    disk: dict[str, int] = {}
    count = 0
    for p in walk_watched_dir(wd, matcher):
        try:
            disk[str(p)] = p.stat().st_mtime_ns
        except FileNotFoundError:
            continue
        count += 1
        if count % 100 == 0:
            await asyncio.sleep(0)  # yield event loop

    db_docs = manager._sqlite.list_watched_docs(prefix=wd.path)
    db_by_path = {d.watched_path: d for d in db_docs}

    for path, mtime in disk.items():
        prev = db_by_path.get(path)
        if prev is None or prev.mtime_ns != mtime:
            manager._enqueue_direct(WatchEvent("UPSERT", path, None))
    for path in db_by_path.keys() - disk.keys():
        manager._enqueue_direct(WatchEvent("DELETE", path, None))

    manager.stats.reconcile_runs += 1
    from datetime import datetime, timezone
    manager.stats.last_reconcile_at = datetime.now(timezone.utc).isoformat()
    logger.info("reconcile dir=%s files=%d", wd.path, len(disk))
