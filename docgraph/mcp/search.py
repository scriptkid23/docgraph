from __future__ import annotations

from typing import Optional

from docgraph.config import Config
from docgraph.embed.provider import EmbeddingProvider
from docgraph.mcp.diversify import mmr_select
from docgraph.models import SearchResult
from docgraph.rerank.factory import create_reranker
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
        self._reranker = create_reranker(cfg)

    async def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        folder: Optional[str] = None,
        tags: Optional[list[str]] = None,
    ) -> list[SearchResult]:
        k = top_k or self._cfg.default_top_k
        vectors = await self._embedder.embed([query], for_query=True)
        query_vec = vectors[0]
        fetch_k = max(k * 5, k)
        raw = self._chroma.search(
            query_embedding=query_vec,
            top_k=fetch_k,
            folder=folder,
            tags=tags,
            include_embeddings=self._cfg.mmr_lambda < 1.0,
        )
        filtered = [r for r in raw if r["score"] >= self._cfg.min_score]

        if self._reranker is not None and filtered:
            rerank_n = min(len(filtered), self._cfg.rerank_top_n)
            to_rerank = filtered[:rerank_n]
            scores = await self._reranker.rerank(
                query, [r["text"] for r in to_rerank]
            )
            for r, rs in zip(to_rerank, scores):
                r["rerank_score"] = rs
            to_rerank.sort(key=lambda x: x.get("rerank_score") or 0.0, reverse=True)
            filtered = to_rerank + filtered[rerank_n:]

        if self._cfg.mmr_lambda < 1.0 and len(filtered) > k:
            if not any(r.get("embedding") for r in filtered):
                raw_emb = self._chroma.search(
                    query_embedding=query_vec,
                    top_k=fetch_k,
                    folder=folder,
                    tags=tags,
                    include_embeddings=True,
                )
                filtered = [r for r in raw_emb if r["score"] >= self._cfg.min_score]
            filtered = mmr_select(
                query_vec, filtered, k=k, lambda_=self._cfg.mmr_lambda
            )
        else:
            filtered = filtered[:k]

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
                file_path=r.get("file_path"),
                heading_path=r.get("heading_path") or [],
                rerank_score=r.get("rerank_score"),
            )
            for r in filtered[:k]
        ]
