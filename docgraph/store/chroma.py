from __future__ import annotations

import hashlib
import json
from typing import Any, Optional

import chromadb

from docgraph.config import Config


COLLECTION_NAME = "docgraph_chunks"


def _decode_tags(raw: Any) -> list[str]:
    """Decode tags metadata. Current writes are JSON; legacy chunks are CSV."""
    if not raw:
        return []
    if isinstance(raw, list):
        return [str(t) for t in raw]
    s = str(raw)
    if s.startswith("["):
        try:
            value = json.loads(s)
            if isinstance(value, list):
                return [str(t) for t in value]
        except json.JSONDecodeError:
            pass
    return [t for t in s.split(",") if t]


def _decode_heading_path(raw: Any) -> list[str]:
    if not raw:
        return []
    if isinstance(raw, list):
        return [str(h) for h in raw]
    s = str(raw)
    if s.startswith("["):
        try:
            value = json.loads(s)
            if isinstance(value, list):
                return [str(h) for h in value]
        except json.JSONDecodeError:
            pass
    return []


def _chunk_hash(text: str) -> str:
    return hashlib.sha1(text.strip().lower().encode("utf-8")).hexdigest()


class ChromaStore:
    def __init__(self, cfg: Config) -> None:
        self._cfg = cfg
        self._client = chromadb.PersistentClient(path=str(cfg.chroma_path))
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    def upsert_chunks(self, chunks: list[dict[str, Any]]) -> None:
        if not chunks:
            return
        dedup_enabled = getattr(self._cfg, "dedup_enabled", True)
        seen_hashes: set[str] = set()
        filtered: list[dict[str, Any]] = []
        for c in chunks:
            text = c.get("text", "")
            h = _chunk_hash(text)
            meta = dict(c.get("metadata") or {})
            meta["chunk_hash"] = h
            if dedup_enabled and h in seen_hashes:
                continue
            seen_hashes.add(h)
            filtered.append({**c, "metadata": meta})
        if not filtered:
            return
        self._collection.upsert(
            ids=[c["id"] for c in filtered],
            embeddings=[c["embedding"] for c in filtered],
            documents=[c["text"] for c in filtered],
            metadatas=[c["metadata"] for c in filtered],
        )

    def get_by_doc_id(self, doc_id: str) -> list[dict[str, Any]]:
        existing = self._collection.get(where={"doc_id": doc_id})
        out: list[dict[str, Any]] = []
        if not existing["ids"]:
            return out
        for i, chunk_id in enumerate(existing["ids"]):
            meta = existing["metadatas"][i] if existing.get("metadatas") else {}
            out.append({
                "id": chunk_id,
                "text": existing["documents"][i] if existing.get("documents") else "",
                "metadata": meta or {},
            })
        return out

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        folder: Optional[str] = None,
        tags: Optional[list[str]] = None,
        *,
        include_embeddings: bool = False,
    ) -> list[dict[str, Any]]:
        where: dict[str, Any] = {}
        if folder:
            where["folder"] = folder
        kwargs: dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": top_k,
        }
        if include_embeddings:
            kwargs["include"] = ["documents", "metadatas", "distances", "embeddings"]
        else:
            kwargs["include"] = ["documents", "metadatas", "distances"]
        if where:
            kwargs["where"] = where
        result = self._collection.query(**kwargs)
        out: list[dict[str, Any]] = []
        if not result["ids"] or not result["ids"][0]:
            return out
        required_tags = set(tags or ())
        embeddings = (
            result["embeddings"][0]
            if include_embeddings and result.get("embeddings")
            else [None] * len(result["ids"][0])
        )
        for i, chunk_id in enumerate(result["ids"][0]):
            meta = result["metadatas"][0][i]
            chunk_tags = _decode_tags(meta.get("tags", ""))
            if required_tags and not required_tags.issubset(chunk_tags):
                continue
            dist = result["distances"][0][i] if result.get("distances") else 0.0
            score = 1.0 - dist
            entry: dict[str, Any] = {
                "id": chunk_id,
                "text": result["documents"][0][i],
                "doc_id": meta.get("doc_id", ""),
                "filename": meta.get("filename", ""),
                "folder": meta.get("folder", ""),
                "tags": chunk_tags,
                "chunk_index": int(meta.get("chunk_index", 0)),
                "score": score,
                "source_page": meta.get("source_page"),
                "file_path": meta.get("file_path"),
                "heading_path": _decode_heading_path(meta.get("heading_path", "")),
                "rerank_score": None,
            }
            if include_embeddings and embeddings[i] is not None:
                entry["embedding"] = list(embeddings[i])
            out.append(entry)
        return out

    def delete_by_doc_id(self, doc_id: str) -> None:
        existing = self._collection.get(where={"doc_id": doc_id})
        if existing["ids"]:
            self._collection.delete(ids=existing["ids"])

    def all_chunk_token_stats(self) -> list[dict[str, Any]]:
        """Return per-chunk token counts for diagnostics."""
        data = self._collection.get(include=["metadatas"])
        stats: list[dict[str, Any]] = []
        if not data["ids"]:
            return stats
        for i, chunk_id in enumerate(data["ids"]):
            meta = data["metadatas"][i] if data.get("metadatas") else {}
            stats.append({
                "id": chunk_id,
                "doc_id": meta.get("doc_id", ""),
                "tokens": int(meta.get("tokens", 0)),
            })
        return stats
