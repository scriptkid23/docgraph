from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone

from watchdog.observers import Observer

from docgraph.watch.handler import DocgraphEventHandler
from docgraph.watch.types import WatcherState, WatcherStats, WatchEvent

_ACTION_MAP = {
    "created": "UPSERT",
    "modified": "UPSERT",
    "deleted": "DELETE",
    "moved": "RENAME",
}

logger = logging.getLogger(__name__)


class WatcherTransitionInProgress(Exception):
    """Raised when enable/disable is called while another transition is running."""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class WatcherManager:
    """Owns the watcher subsystem: state machine, observer, queue, workers."""

    def __init__(self, app_state) -> None:
        self._app = app_state
        self._cfg = app_state.cfg
        self._sqlite = app_state.sqlite
        self.state: WatcherState = WatcherState.DISABLED
        self.stats = WatcherStats()
        self._lock = asyncio.Lock()
        self._observer = None
        self._workers: list[asyncio.Task] = []
        self._queues: list[asyncio.Queue] = []
        self._debounce_tasks: dict[str, asyncio.TimerHandle] = {}
        self._shutdown = asyncio.Event()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._last_enabled_at: str | None = None
        self._recovery_task: asyncio.Task | None = None
        # Restore persisted state.
        persisted = self._sqlite.get_watcher_state("enabled")
        if persisted == "true":
            logger.info("watcher: persisted state is enabled, will auto-enable on startup hook")

    async def enable(self) -> dict:
        if self.state == WatcherState.ENABLED:
            return {"enabled": True, "reconcile_started": False, "dirs": len(self._sqlite.list_watched_dirs())}
        if self.state in (WatcherState.ENABLING, WatcherState.DISABLING):
            raise WatcherTransitionInProgress("watcher transition in progress")
        async with self._lock:
            if self.state == WatcherState.ENABLED:
                return {"enabled": True, "reconcile_started": False, "dirs": len(self._sqlite.list_watched_dirs())}
            self.state = WatcherState.ENABLING
            try:
                self._shutdown = asyncio.Event()
                self._loop = asyncio.get_running_loop()
                self.stats.reset()
                await self._start_observer()
                await self._start_workers()
                self._sqlite.set_watcher_state("enabled", "true")
                self._last_enabled_at = _now_iso()
                self._sqlite.set_watcher_state("last_enabled_at", self._last_enabled_at)
                self.state = WatcherState.ENABLED
                logger.info("watcher: enabled")
                # Schedule reconcile (background, non-blocking).
                dirs = self._sqlite.list_watched_dirs()
                for wd in dirs:
                    asyncio.create_task(self._reconcile_dir(wd))
                # Periodic recovery.
                self._recovery_task = asyncio.create_task(self._recovery_loop())
                return {"enabled": True, "reconcile_started": True, "dirs": len(dirs)}
            except Exception:
                self.state = WatcherState.DISABLED
                raise

    async def disable(self) -> dict:
        if self.state == WatcherState.DISABLED:
            return {"enabled": False, "queue_drained": 0, "queue_dropped": 0}
        if self.state in (WatcherState.ENABLING, WatcherState.DISABLING):
            raise WatcherTransitionInProgress("watcher transition in progress")
        async with self._lock:
            if self.state == WatcherState.DISABLED:
                return {"enabled": False, "queue_drained": 0, "queue_dropped": 0}
            self.state = WatcherState.DISABLING
            try:
                self._shutdown.set()
                # Cancel pending debounces.
                for h in list(self._debounce_tasks.values()):
                    h.cancel()
                self._debounce_tasks.clear()
                # Stop observer.
                if self._observer is not None:
                    self._observer.stop()
                    self._observer.join(timeout=5.0)
                    self._observer = None
                # Count remaining queue items, then cancel workers.
                drained = sum(q.qsize() for q in self._queues)
                for t in self._workers:
                    t.cancel()
                await asyncio.gather(*self._workers, return_exceptions=True)
                self._workers = []
                self._queues = []
                if self._recovery_task is not None:
                    self._recovery_task.cancel()
                    self._recovery_task = None
                self._sqlite.set_watcher_state("enabled", "false")
                self.state = WatcherState.DISABLED
                logger.info("watcher: disabled, queue_drained=%d", drained)
                return {"enabled": False, "queue_drained": drained, "queue_dropped": self.stats.events_dropped_queue_full}
            except Exception:
                self.state = WatcherState.ENABLED  # roll back state
                raise

    # ---- placeholders filled in by later tasks ----
    async def _start_observer(self) -> None:
        self._observer = Observer()
        handler = DocgraphEventHandler(self, self._loop)
        for wd in self._sqlite.list_watched_dirs():
            try:
                self._observer.schedule(handler, wd.path, recursive=True)
            except Exception as exc:
                logger.warning("failed to schedule watch for %s: %s", wd.path, exc)
        self._observer.start()

    def _on_raw_event(self, event_type: str, src_path: str, dest_path: str | None) -> None:
        """Called on the asyncio loop via call_soon_threadsafe."""
        self.stats.events_received += 1
        action = _ACTION_MAP.get(event_type)
        if action is None:
            return
        self._enqueue_debounced(action, src_path, dest_path)

    def _enqueue_debounced(self, action: str, src_path: str, dest_path: str | None) -> None:
        key = src_path
        if existing := self._debounce_tasks.get(key):
            existing.cancel()
            self.stats.events_debounced += 1
        handle = self._loop.call_later(
            self._cfg.watch_debounce_sec,
            self._flush_debounce, action, src_path, dest_path,
        )
        self._debounce_tasks[key] = handle

    def _flush_debounce(self, action: str, src_path: str, dest_path: str | None) -> None:
        self._debounce_tasks.pop(src_path, None)
        event = WatchEvent(action=action, src_path=src_path, dest_path=dest_path)
        self._enqueue_direct(event)

    def _enqueue_direct(self, event: WatchEvent) -> None:
        if not self._queues:
            self.stats.events_dropped_queue_full += 1
            return
        idx = hash(event.src_path) % len(self._queues)
        try:
            self._queues[idx].put_nowait(event)
            self.stats.events_processed += 1
        except asyncio.QueueFull:
            self.stats.events_dropped_queue_full += 1
            logger.warning("watcher queue full; dropping event: %s %s", event.action, event.src_path)

    async def _start_workers(self) -> None:
        # Filled by Task 13.
        self._queues = [asyncio.Queue(maxsize=max(1, self._cfg.watch_queue_capacity // self._cfg.watch_workers))
                        for _ in range(self._cfg.watch_workers)]

    async def _reconcile_dir(self, wd) -> None:
        # Filled by Task 15.
        pass

    async def _recovery_loop(self) -> None:
        # Filled by Task 15.
        try:
            while not self._shutdown.is_set():
                await asyncio.sleep(self._cfg.watch_recovery_interval_sec)
                if self._shutdown.is_set():
                    return
                for wd in self._sqlite.list_watched_dirs():
                    await self._reconcile_dir(wd)
        except asyncio.CancelledError:
            return
