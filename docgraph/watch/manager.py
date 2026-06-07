from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone

from docgraph.watch.types import WatcherState, WatcherStats

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
        # Filled by Task 11.
        pass

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
