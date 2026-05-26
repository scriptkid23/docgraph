from __future__ import annotations

from pathlib import Path

from boostmcp.config import Config
from boostmcp.embed.provider import EmbeddingProvider
from boostmcp.ingest.chunker import chunk_markdown
from boostmcp.ingest.converter import convert_file_to_markdown
from boostmcp.models import DocumentStatus
from boostmcp.store.chroma import ChromaStore
from boostmcp.store.files import FileStore
from boostmcp.store.sqlite import SQLiteStore


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

    async def index_document(self, doc_id: str, original_path: Path) -> None:
        doc = self._sqlite.get_document(doc_id)
        if doc is None:
            raise ValueError(f"document not found: {doc_id}")
        try:
            markdown = convert_file_to_markdown(original_path)
            md_path = self._files.save_markdown(doc_id, markdown)
            chunks = chunk_markdown(
                markdown,
                chunk_size=self._cfg.chunk_size,
                chunk_overlap=self._cfg.chunk_overlap,
            )
            if not chunks:
                raise ValueError("no chunks produced from document")

            vectors = await self._embedder.embed(chunks)
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
        await self.index_document(doc_id, Path(doc.original_path))
