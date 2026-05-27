from __future__ import annotations

import asyncio
from pathlib import Path

from docgraph.config import Config
from docgraph.embed.ollama import EMBED_BATCH_SIZE
from docgraph.embed.provider import EmbeddingProvider
from docgraph.ingest.chunker import chunk_markdown
from docgraph.ingest.converter import convert_file_to_markdown
from docgraph.models import DocumentStatus
from docgraph.store.chroma import ChromaStore
from docgraph.store.files import FileStore
from docgraph.store.sqlite import SQLiteStore


class Indexer:
    def __init__(
        self,
        cfg: Config,
        sqlite: SQLiteStore,
        files: FileStore,
        chroma: ChromaStore,
        embedder: EmbeddingProvider,
    ) -> None:
        self._cfg = cfg
        self._sqlite = sqlite
        self._files = files
        self._chroma = chroma
        self._embedder = embedder

    def _progress(self, doc_id: str, pct: int, phase: str) -> None:
        self._sqlite.update_progress(doc_id, pct, phase)

    async def _embed_with_progress(
        self, doc_id: str, chunks: list[str]
    ) -> list[list[float]]:
        total = len(chunks)
        vectors: list[list[float]] = []
        for start in range(0, total, EMBED_BATCH_SIZE):
            batch = chunks[start : start + EMBED_BATCH_SIZE]
            vectors.extend(await self._embedder.embed(batch))
            done = min(start + len(batch), total)
            pct = 45 + int(50 * done / total)
            self._progress(
                doc_id,
                pct,
                f"Embedding {done}/{total} chunks ({pct}%)",
            )
        return vectors

    async def index_document(self, doc_id: str, original_path: Path) -> None:
        doc = self._sqlite.get_document(doc_id)
        if doc is None:
            raise ValueError(f"document not found: {doc_id}")
        try:
            self._progress(doc_id, 5, "Converting to text (5%)")
            markdown = await asyncio.to_thread(
                convert_file_to_markdown, original_path
            )
            self._progress(doc_id, 28, "Converted — splitting chunks (28%)")
            md_path = self._files.save_markdown(doc_id, markdown)
            chunks = await asyncio.to_thread(
                chunk_markdown,
                markdown,
                self._cfg.chunk_size,
                self._cfg.chunk_overlap,
            )
            if not chunks:
                raise ValueError("no chunks produced from document")

            self._progress(
                doc_id,
                42,
                f"Chunked into {len(chunks)} parts (42%)",
            )
            vectors = await self._embed_with_progress(doc_id, chunks)
            self._progress(doc_id, 96, "Saving to index (96%)")
            chroma_chunks = []
            for i, (text, vec) in enumerate(zip(chunks, vectors)):
                chroma_chunks.append({
                    "id": f"{doc_id}_{i}",
                    "embedding": vec,
                    "text": text,
                    "metadata": {
                        "doc_id": doc_id,
                        "filename": doc.filename,
                        "folder": doc.folder,
                        "tags": ",".join(doc.tags),
                        "chunk_index": i,
                    },
                })
            self._chroma.upsert_chunks(chroma_chunks)
            self._sqlite.update_status(
                doc_id,
                DocumentStatus.READY,
                chunk_count=len(chunks),
                markdown_path=str(md_path),
            )
        except Exception as exc:
            self._sqlite.update_status(
                doc_id,
                DocumentStatus.ERROR,
                error_message=str(exc),
            )
            raise

    async def reindex_document(self, doc_id: str) -> None:
        doc = self._sqlite.get_document(doc_id)
        if doc is None or not doc.original_path:
            raise ValueError(f"cannot reindex: {doc_id}")
        self._chroma.delete_by_doc_id(doc_id)
        self._sqlite.update_status(doc_id, DocumentStatus.PROCESSING)
        self._progress(doc_id, 0, "Starting re-index (0%)")
        await self.index_document(doc_id, Path(doc.original_path))
