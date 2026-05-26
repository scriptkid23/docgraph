from __future__ import annotations

from typing import Optional

from boostmcp.config import Config
from boostmcp.embed.provider import EmbeddingProvider
from boostmcp.models import SearchResult
from boostmcp.store.chroma import ChromaStore
from boostmcp.store.sqlite import SQLiteStore


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
        tag: Optional[str] = None,
    ) -> list[SearchResult]:
        k = top_k or self._cfg.default_top_k
        vectors = await self._embedder.embed([query])
        raw = self._chroma.search(
            query_embedding=vectors[0],
            top_k=k,
            folder=folder,
            tag=tag,
        )
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
            for r in raw
            if r["score"] >= self._cfg.min_score
        ]
