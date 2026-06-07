from __future__ import annotations

import asyncio
import logging
import re

logger = logging.getLogger(__name__)


class KeywordReranker:
    """Lightweight reranker using keyword overlap (no extra model download)."""

    async def rerank(self, query: str, documents: list[str]) -> list[float]:
        if not documents:
            return []
        q_terms = set(_tokenize(query))
        scores: list[float] = []
        for doc in documents:
            d_terms = set(_tokenize(doc))
            if not q_terms:
                scores.append(0.0)
                continue
            overlap = len(q_terms & d_terms) / len(q_terms)
            scores.append(overlap)
        return scores


class CrossEncoderReranker:
    """Optional cross-encoder reranker via sentence-transformers."""

    def __init__(self, model: str) -> None:
        self._model_name = model
        self._model = None
        self._lock = asyncio.Lock()

    async def _ensure(self) -> None:
        if self._model is not None:
            return
        async with self._lock:
            if self._model is not None:
                return
            from sentence_transformers import CrossEncoder

            self._model = await asyncio.to_thread(CrossEncoder, self._model_name)
            logger.info("Reranker ready: %s", self._model_name)

    async def rerank(self, query: str, documents: list[str]) -> list[float]:
        if not documents:
            return []
        await self._ensure()
        pairs = [(query, doc) for doc in documents]
        scores = await asyncio.to_thread(self._model.predict, pairs)
        return [float(s) for s in scores]


def _tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9_]+", text.lower()) if len(t) > 1]
