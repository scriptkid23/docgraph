from __future__ import annotations

import asyncio
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class Reranker:
    """Cross-encoder reranker via Rust (docgraph_embed::rerank).

    Lazy-initialized; thread-safe through an asyncio.Lock. Mirrors the
    LocalEmbedder pattern in docgraph/embed/local.py.
    """

    HEALTH_MSG = (
        "Reranker unavailable. Build the Rust crate: "
        "cd crates/docgraph-embed && maturin develop --release"
    )

    def __init__(self, model: str, cache_dir: Path) -> None:
        self._model = model
        self._cache_dir = cache_dir
        self._init_lock = asyncio.Lock()
        self._initialized = False

    @property
    def is_ready(self) -> bool:
        """True once the native model has been initialized."""
        return self._initialized

    def _import_native(self):
        try:
            import docgraph_embed
        except ImportError as exc:
            raise RuntimeError(self.HEALTH_MSG) from exc
        # sys.modules[name] = None is the cached "not installed" sentinel;
        # import succeeds without raising but returns None.
        if docgraph_embed is None:
            raise RuntimeError(self.HEALTH_MSG)
        return docgraph_embed

    async def _ensure_init(self) -> None:
        if self._initialized:
            return
        async with self._init_lock:
            if self._initialized:
                return
            native = self._import_native()
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            await asyncio.to_thread(
                native.rerank_init, self._model, str(self._cache_dir)
            )
            self._initialized = True
            logger.info(
                "Reranker ready: model=%s cache=%s", self._model, self._cache_dir
            )

    async def prewarm(self) -> None:
        """Force model load + dummy inference at server lifespan startup.
        Swallows errors so server start cannot fail because of reranker.

        Two failure modes are handled distinctly so log messages match reality:
          - init failure: ``_initialized`` stays False, next ``rerank`` call
            will retry init.
          - warmup inference failure (init OK, dummy rerank crashed): the
            model loaded but inference is broken. ``_initialized`` stays True,
            so future calls will NOT retry init — they will hit the same
            inference bug. Log accordingly instead of misleading "will retry".
        """
        try:
            await self._ensure_init()
        except Exception as exc:
            logger.warning(
                "Reranker init failed; will retry on first call: %s", exc
            )
            return
        try:
            native = self._import_native()
            await asyncio.to_thread(native.rerank, "warmup", ["test passage"])
            logger.info("Reranker pre-warmed")
        except Exception as exc:
            logger.warning(
                "Reranker warmup inference failed; rerank() calls may fail "
                "the same way (model loaded but inference broken): %s",
                exc,
            )

    async def rerank(self, query: str, passages: list[str]) -> list[float]:
        if not passages:
            return []
        await self._ensure_init()
        native = self._import_native()
        scores = await asyncio.to_thread(native.rerank, query, passages)
        return [float(s) for s in scores]

    async def health_check(self) -> None:
        await self._ensure_init()
        native = self._import_native()
        await asyncio.to_thread(native.rerank_health_check)
