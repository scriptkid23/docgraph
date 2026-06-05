from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from docgraph.config import Config
from docgraph.embed.provider import EmbeddingProvider
from docgraph.models import SearchResult
from docgraph.store.chroma import ChromaStore
from docgraph.store.sqlite import SQLiteStore


@dataclass
class FusedHit:
    """A chunk that surfaced from one or both branches, with fused score."""
    chunk_id: str
    text: str
    doc_id: str
    filename: str
    folder: str
    tags: list[str]
    chunk_index: int
    source_page: Optional[int]
    vector_score: Optional[float]
    bm25_score: Optional[float]
    rrf_score: float
    rerank_score: Optional[float] = None


def _rrf_fuse(
    vector_results: list[dict[str, Any]],
    sparse_results: list[dict[str, Any]],
    k_rrf: int = 60,
) -> list[FusedHit]:
    """Reciprocal Rank Fusion. Sorts vector then sparse, adds 1/(k+rank+1) per
    branch occurrence. Returns hits sorted by descending rrf_score."""
    fused: dict[str, FusedHit] = {}
    for rank, hit in enumerate(vector_results):
        cid = hit["id"]
        if cid not in fused:
            fused[cid] = FusedHit(
                chunk_id=cid,
                text=hit.get("text", ""),
                doc_id=hit.get("doc_id", ""),
                filename=hit.get("filename", ""),
                folder=hit.get("folder", ""),
                tags=list(hit.get("tags") or []),
                chunk_index=int(hit.get("chunk_index", 0)),
                source_page=hit.get("source_page"),
                vector_score=float(hit["score"]) if "score" in hit else None,
                bm25_score=None,
                rrf_score=0.0,
            )
        fused[cid].rrf_score += 1.0 / (k_rrf + rank + 1)

    for rank, hit in enumerate(sparse_results):
        cid = hit["chunk_id"]
        if cid in fused:
            # existing (could be vector-seeded OR a prior sparse duplicate)
            if fused[cid].bm25_score is None:
                fused[cid].bm25_score = float(hit["bm25_score"])
        else:
            fused[cid] = FusedHit(
                chunk_id=cid,
                text="",  # sparse path doesn't carry text; will be filled later
                doc_id=hit.get("doc_id", ""),
                filename="",
                folder=hit.get("folder", ""),
                tags=list(hit.get("tags") or []),
                chunk_index=int(hit.get("chunk_index", 0)),
                source_page=None,
                vector_score=None,
                bm25_score=float(hit["bm25_score"]),
                rrf_score=0.0,
            )
        fused[cid].rrf_score += 1.0 / (k_rrf + rank + 1)

    return sorted(fused.values(), key=lambda h: (-h.rrf_score, h.chunk_id))


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
        vectors = await self._embedder.embed([query], for_query=True)
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
