from __future__ import annotations

from typing import Optional

from docgraph.config import Config
from docgraph.embed.provider import EmbeddingProvider
from docgraph.models import SearchResult
from docgraph.store.chroma import ChromaStore
from docgraph.store.sqlite import SQLiteStore


class SearchService:
    def __init__(
        self,
        cfg: Config,
        sqlite: SQLiteStore,
        chroma: ChromaStore,
        embedder: EmbeddingProvider,
    ) -> None:
        self._cfg = cfg
        self._sqlite = sqlite
        self._chroma = chroma
        self._embedder = embedder

    async def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        folder: Optional[str] = None,
        tags: Optional[list[str]] = None,
    ) -> list[SearchResult]:
        k = top_k or self._cfg.default_top_k
        vectors = await self._embedder.embed([query])
        # Overfetch so the min_score filter doesn't starve the caller of results
        # when high-quality matches exist beyond the top_k cutoff.
        raw = self._chroma.search(
            query_embedding=vectors[0],
            top_k=max(k * 3, k),
            folder=folder,
            tags=tags,
        )
        filtered = [r for r in raw if r["score"] >= self._cfg.min_score]
        return [
            SearchResult(
                text=r["text"],
                doc_id=r["doc_id"],
                filename=r["filename"],
                folder=r["folder"],
                tags=r["tags"],
                chunk_index=r["chunk_index"],
                score=r["score"],
                source_page=r.get("source_page"),
            )
            for r in filtered[:k]
        ]
