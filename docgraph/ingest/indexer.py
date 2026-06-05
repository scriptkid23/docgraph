from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

from docgraph.config import Config
from docgraph.embed.local import EMBED_BATCH_SIZE
from docgraph.embed.provider import EmbeddingProvider
from docgraph.ingest.chunker import chunk_markdown
from docgraph.ingest.converter import convert_file_to_markdown
from docgraph.ingest.crawler import UrlCrawler
from docgraph.models import DocumentStatus, SourceType
from docgraph.store.chroma import ChromaStore
from docgraph.store.files import FileStore
from docgraph.store.fts import FtsStore
from docgraph.store.sqlite import SQLiteStore

logger = logging.getLogger(__name__)


class Indexer:
    def __init__(
        self,
        cfg: Config,
        sqlite: SQLiteStore,
        files: FileStore,
        chroma: ChromaStore,
        embedder: EmbeddingProvider,
        fts: "FtsStore | None" = None,
    ) -> None:
        self._cfg = cfg
        self._sqlite = sqlite
        self._files = files
        self._chroma = chroma
        self._embedder = embedder
        self._fts = fts

    def _progress(self, doc_id: str, pct: int, phase: str) -> None:
        self._sqlite.update_progress(doc_id, pct, phase)

    async def _embed_with_progress(
        self, doc_id: str, chunks: list[str]
    ) -> list[list[float]]:
        total = len(chunks)
        vectors: list[list[float]] = []
        for start in range(0, total, EMBED_BATCH_SIZE):
            batch = chunks[start : start + EMBED_BATCH_SIZE]
            vectors.extend(await self._embedder.embed(batch, for_query=False))
            done = min(start + len(batch), total)
            pct = 45 + int(50 * done / total)
            self._progress(
                doc_id,
                pct,
                f"Embedding {done}/{total} chunks ({pct}%)",
            )
        return vectors

    async def index_markdown(self, doc_id: str, markdown: str) -> None:
        doc = self._sqlite.get_document(doc_id)
        if doc is None:
            raise ValueError(f"document not found: {doc_id}")
        try:
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
            if len(chunks) > self._cfg.max_chunks_per_doc:
                raise ValueError(
                    f"document exceeds max_chunks_per_doc "
                    f"({len(chunks)} > {self._cfg.max_chunks_per_doc}); "
                    f"increase DOCGRAPH_MAX_CHUNKS_PER_DOC or split the source"
                )

            self._progress(
                doc_id,
                42,
                f"Chunked into {len(chunks)} parts (42%)",
            )
            vectors = await self._embed_with_progress(doc_id, chunks)
            self._progress(doc_id, 96, "Saving to index (96%)")
            chroma_chunks = []
            fts_chunks = []
            for i, (text, vec) in enumerate(zip(chunks, vectors)):
                chunk_id = f"{doc_id}_{i}"
                metadata = {
                    "doc_id": doc_id,
                    "filename": doc.filename,
                    "folder": doc.folder,
                    # JSON-encoded so tags containing commas survive round-tripping.
                    "tags": json.dumps(doc.tags),
                    "chunk_index": i,
                }
                if doc.source_url:
                    metadata["source_url"] = doc.source_url
                chroma_chunks.append({
                    "id": chunk_id,
                    "embedding": vec,
                    "text": text,
                    "metadata": metadata,
                })
                fts_chunks.append({
                    "chunk_id": chunk_id,
                    "doc_id": doc_id,
                    "folder": doc.folder,
                    "tags": json.dumps(doc.tags),
                    "chunk_index": i,
                    "text": text,
                    "filename": doc.filename,
                })
            self._chroma.upsert_chunks(chroma_chunks)
            if self._fts is not None and self._cfg.hybrid_enabled:
                try:
                    self._fts.upsert_chunks(fts_chunks)
                except Exception as exc:
                    logger.warning(
                        "FTS5 upsert failed for doc_id=%s (search will degrade to vector-only): %s",
                        doc_id, exc,
                    )
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

    async def index_document(self, doc_id: str, original_path: Path) -> None:
        doc = self._sqlite.get_document(doc_id)
        if doc is None:
            raise ValueError(f"document not found: {doc_id}")
        logger.info(
            "indexing file doc_id=%s path=%s", doc_id, original_path.name
        )
        try:
            self._progress(doc_id, 5, "Converting to text (5%)")
            markdown = await asyncio.to_thread(
                convert_file_to_markdown, original_path
            )
            logger.debug(
                "converted doc_id=%s markdown_chars=%d", doc_id, len(markdown)
            )
            self._progress(doc_id, 20, "Converted to markdown (20%)")
            await self.index_markdown(doc_id, markdown)
        except Exception as exc:
            self._sqlite.update_status(
                doc_id,
                DocumentStatus.ERROR,
                error_message=str(exc),
            )
            raise

    async def index_url(
        self,
        doc_id: str,
        url: str,
        *,
        crawler: UrlCrawler | None = None,
    ) -> None:
        if crawler is not None:
            await self._index_url_with_crawler(doc_id, url, crawler)
            return
        async with UrlCrawler(self._cfg) as c:
            await self._index_url_with_crawler(doc_id, url, c)

    async def _index_url_with_crawler(
        self, doc_id: str, url: str, crawler: UrlCrawler
    ) -> None:
        doc = self._sqlite.get_document(doc_id)
        if doc is None:
            raise ValueError(f"document not found: {doc_id}")
        logger.info("indexing url doc_id=%s url=%s", doc_id, url)
        try:
            self._progress(doc_id, 5, "Fetching page (5%)")
            markdown, title = await crawler.crawl(url)
            logger.debug(
                "crawled doc_id=%s title=%r markdown_chars=%d",
                doc_id,
                title,
                len(markdown),
            )
            self._sqlite.update_filename(doc_id, title)
            self._progress(doc_id, 20, "Converted to markdown (20%)")
            await self.index_markdown(doc_id, markdown)
        except Exception as exc:
            self._sqlite.update_status(
                doc_id,
                DocumentStatus.ERROR,
                error_message=str(exc),
            )
            raise

    async def index_urls_batch(self, items: list[tuple[str, str]]) -> None:
        """Crawl and index multiple URLs reusing one browser session."""
        if not items:
            return
        async with UrlCrawler(self._cfg) as crawler:
            for doc_id, url in items:
                try:
                    await self._index_url_with_crawler(doc_id, url, crawler)
                except Exception:
                    # _index_url_with_crawler records ERROR status; continue to next URL.
                    logger.exception("URL index failed for doc_id=%s url=%s", doc_id, url)

    async def reindex_document(self, doc_id: str) -> None:
        doc = self._sqlite.get_document(doc_id)
        if doc is None:
            raise ValueError(f"cannot reindex: {doc_id}")
        self._chroma.delete_by_doc_id(doc_id)
        if self._fts is not None:
            try:
                self._fts.delete_by_doc_id(doc_id)
            except Exception as exc:
                logger.warning("FTS5 delete failed for doc_id=%s: %s", doc_id, exc)
        self._sqlite.update_status(doc_id, DocumentStatus.PROCESSING)
        self._progress(doc_id, 0, "Starting re-index (0%)")
        if doc.source_type == SourceType.URL and doc.source_url:
            await self.index_url(doc_id, doc.source_url)
        elif doc.original_path:
            await self.index_document(doc_id, Path(doc.original_path))
        else:
            raise ValueError(f"cannot reindex: {doc_id}")
