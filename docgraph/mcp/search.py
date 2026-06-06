from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any, Optional

from docgraph.config import Config
from docgraph.embed.provider import EmbeddingProvider
from docgraph.models import SearchResult
from docgraph.store.chroma import ChromaStore
from docgraph.store.fts import FtsStore
from docgraph.store.sqlite import SQLiteStore

logger = logging.getLogger(__name__)


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
                text=hit.get("text", ""),
                doc_id=hit.get("doc_id", ""),
                filename=hit.get("filename", ""),
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


def _should_rerank(cfg: Config, fused: list[FusedHit], k: int) -> tuple[bool, str]:
    """Return (decision, reason) for observability."""
    if not cfg.rerank_enabled:
        return False, "skip_disabled"
    if len(fused) < 2:
        return False, "skip_too_few_candidates"

    top1 = fused[0].rrf_score

    # A. Floor — top-1 below recall floor; rerank can't fix bad recall
    if top1 < cfg.rerank_min_floor:
        return False, "skip_floor"

    # B. Single-branch override — only fires when hybrid was supposed to fire
    # but one branch came back empty. Skip when hybrid is intentionally disabled.
    if cfg.hybrid_enabled:
        window = fused[: max(k, 2)]
        has_vector = any(h.vector_score is not None for h in window)
        has_sparse = any(h.bm25_score is not None for h in window)
        if not (has_vector and has_sparse):
            return True, "force_single_branch"

    # Default: rerank when top-k scores are clustered (ambiguous)
    top_window = fused[:k] if len(fused) >= k else fused
    top_scores = [h.rrf_score for h in top_window]
    gap = top_scores[0] - top_scores[-1]
    if top1 > 0 and gap / top1 > cfg.rerank_score_gap_ratio:
        return False, "skip_gap"
    return True, "force_ambiguous"


class SearchService:
    def __init__(
        self,
        cfg: Config,
        sqlite: SQLiteStore,
        chroma: ChromaStore,
        embedder: EmbeddingProvider,
        fts: Optional[FtsStore] = None,
        reranker=None,
    ) -> None:
        self._cfg = cfg
        self._sqlite = sqlite
        self._chroma = chroma
        self._embedder = embedder
        self._fts = fts
        self._reranker = reranker

    async def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        folder: Optional[str] = None,
        tags: Optional[list[str]] = None,
    ) -> list[SearchResult]:
        start = time.perf_counter()
        # top_k=0 was a documented "use default" sentinel under the old vector-only
        # code path. Preserve that semantics: 0 and None both mean default.
        k_requested = top_k if top_k else self._cfg.default_top_k
        k_requested = max(k_requested, 1)  # guard negative values
        k = min(k_requested, 100)
        if k_requested > k:
            logger.warning("top_k capped from %d to %d", k_requested, k)
        overfetch = max(k * 6, 30)

        use_hybrid = self._cfg.hybrid_enabled and self._fts is not None

        # 1) Embed query (always needed for vector branch)
        vectors = await self._embedder.embed([query], for_query=True)
        query_vec = vectors[0]

        # 2) Parallel branches
        vector_task = asyncio.create_task(
            asyncio.to_thread(
                self._chroma.search, query_vec, overfetch, folder, tags
            )
        )
        if use_hybrid:
            sparse_task = asyncio.create_task(
                asyncio.to_thread(self._fts.search, query, overfetch, folder, tags)
            )
            vector_results, sparse_results = await asyncio.gather(
                vector_task, sparse_task, return_exceptions=True
            )
            if isinstance(vector_results, BaseException):
                logger.warning(
                    "Vector branch failed; falling back to sparse-only: %r",
                    vector_results,
                )
                vector_results = []
            if isinstance(sparse_results, BaseException):
                logger.warning(
                    "Sparse branch failed; falling back to vector-only: %r",
                    sparse_results,
                )
                sparse_results = []
        else:
            vector_results = await vector_task
            sparse_results = []

        # 3) Fuse with RRF
        fused = _rrf_fuse(vector_results, sparse_results, k_rrf=self._cfg.rrf_k)
        fused = fused[: max(k * 3, self._cfg.rerank_top_n)]

        # Backfill text for sparse-only hits (sparse branch doesn't carry text)
        chunk_to_text: dict[str, str] = {
            h["id"]: h.get("text", "") for h in vector_results
        }
        for fh in fused:
            if not fh.text and fh.chunk_id in chunk_to_text:
                fh.text = chunk_to_text[fh.chunk_id]

        # 4) Gate + rerank
        decision, reason = _should_rerank(self._cfg, fused, k)
        rerank_ran = False
        if decision and self._reranker is not None:
            rerank_window_raw = fused[: self._cfg.rerank_top_n]
            rerank_window = [h for h in rerank_window_raw if h.text]
            if not rerank_window:
                # All candidates lacked text; cannot rerank
                decision = False
            if decision:
                try:
                    scores = await asyncio.wait_for(
                        self._reranker.rerank(query, [h.text for h in rerank_window]),
                        timeout=self._cfg.rerank_timeout_sec,
                    )
                    for h, s in zip(rerank_window, scores):
                        h.rerank_score = float(s)
                    # Build score map; non-reranked hits (empty text) keep rerank_score=None
                    score_by_id = {h.chunk_id: h.rerank_score for h in rerank_window}

                    def _key(h: FusedHit) -> tuple[float, str]:
                        rs = score_by_id.get(h.chunk_id)
                        if rs is not None:
                            return (-rs, h.chunk_id)
                        return (-h.rrf_score, h.chunk_id)

                    rerank_window_raw.sort(key=_key)
                    fused = rerank_window_raw + fused[self._cfg.rerank_top_n :]
                    rerank_ran = True
                except asyncio.TimeoutError:
                    logger.warning(
                        "Rerank timed out after %ss; falling back to RRF",
                        self._cfg.rerank_timeout_sec,
                    )
                except Exception:
                    logger.exception("Rerank failed; falling back to RRF")

        # 5) Filter by min_score (only when no rerank score)
        # Note: sparse-only hits (vector_score is None) bypass min_score by design.
        # An exact identifier or filename match via BM25 should not be dropped just
        # because the vector branch had no high-confidence cosine match. The reranker
        # (when enabled) provides the precision gate for these hits. When reranker
        # is disabled or fails silently, low-quality BM25 hits CAN leak through —
        # tune `min_score` higher or set `cfg.rerank_enabled=True` to mitigate.
        filtered: list[FusedHit] = []
        for h in fused:
            if h.rerank_score is not None:
                filtered.append(h)
            elif h.vector_score is None or h.vector_score >= self._cfg.min_score:
                filtered.append(h)

        # 6) Emit metrics log
        total_ms = (time.perf_counter() - start) * 1000.0
        logger.info(
            "search_metrics",
            extra={
                "query_len": len(query),
                "vector_branch_size": len(vector_results),
                "sparse_branch_size": len(sparse_results),
                "fused_size": len(fused),
                "rerank_triggered": rerank_ran,
                "rerank_reason": reason,
                "k": k,
                "total_ms": round(total_ms, 2),
            },
        )

        return [self._to_result(h) for h in filtered[:k]]

    @staticmethod
    def _to_result(h: FusedHit) -> SearchResult:
        # Score priority: rerank > vector > normalized bm25
        if h.rerank_score is not None:
            score = float(h.rerank_score)
        elif h.vector_score is not None:
            score = float(h.vector_score)
        else:
            # bm25 is unbounded; map roughly to [0, 1) via 1 - 1/(1+x)
            score = 1.0 - 1.0 / (1.0 + max(float(h.bm25_score or 0.0), 0.0))
        return SearchResult(
            text=h.text,
            doc_id=h.doc_id,
            filename=h.filename,
            folder=h.folder,
            tags=h.tags,
            chunk_index=h.chunk_index,
            score=score,
            rrf_score=h.rrf_score,
            source_page=h.source_page,
        )
