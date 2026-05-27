from __future__ import annotations

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


class ChromaStore:
    def __init__(self, cfg: Config) -> None:
        self._client = chromadb.PersistentClient(path=str(cfg.chroma_path))
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    def upsert_chunks(self, chunks: list[dict[str, Any]]) -> None:
        if not chunks:
            return
        self._collection.upsert(
            ids=[c["id"] for c in chunks],
            embeddings=[c["embedding"] for c in chunks],
            documents=[c["text"] for c in chunks],
            metadatas=[c["metadata"] for c in chunks],
        )

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        folder: Optional[str] = None,
        tags: Optional[list[str]] = None,
    ) -> list[dict[str, Any]]:
        where: dict[str, Any] = {}
        if folder:
            where["folder"] = folder
        kwargs: dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": top_k,
        }
        if where:
            kwargs["where"] = where
        result = self._collection.query(**kwargs)
        out: list[dict[str, Any]] = []
        if not result["ids"] or not result["ids"][0]:
            return out
        required_tags = set(tags or ())
        for i, chunk_id in enumerate(result["ids"][0]):
            meta = result["metadatas"][0][i]
            chunk_tags = _decode_tags(meta.get("tags", ""))
            if required_tags and not required_tags.issubset(chunk_tags):
                continue
            dist = result["distances"][0][i] if result.get("distances") else 0.0
            score = 1.0 - dist
            out.append({
                "id": chunk_id,
                "text": result["documents"][0][i],
                "doc_id": meta.get("doc_id", ""),
                "filename": meta.get("filename", ""),
                "folder": meta.get("folder", ""),
                "tags": chunk_tags,
                "chunk_index": int(meta.get("chunk_index", 0)),
                "score": score,
                "source_page": meta.get("source_page"),
            })
        return out

    def delete_by_doc_id(self, doc_id: str) -> None:
        existing = self._collection.get(where={"doc_id": doc_id})
        if existing["ids"]:
            self._collection.delete(ids=existing["ids"])
