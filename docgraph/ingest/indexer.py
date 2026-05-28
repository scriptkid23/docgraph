from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

from docgraph.config import Config
from docgraph.embed.local import EMBED_BATCH_SIZE
from docgraph.embed.provider import EmbeddingProvider
from docgraph.ingest.chunker import chunk_code, chunk_markdown
from docgraph.ingest.code_dump import detect_repomix, infer_language, parse_repomix
from docgraph.ingest.converter import convert_file_to_markdown
from docgraph.ingest.crawler import UrlCrawler
from docgraph.models import DocumentStatus, SourceType
from docgraph.store.chroma import ChromaStore
from docgraph.store.files import FileStore
from docgraph.store.sqlite import SQLiteStore

logger = logging.getLogger(__name__)


def _read_text_prefix(path: Path, limit: int = 65536) -> str:
    """Read up to `limit` chars of a file as text, ignoring decode errors."""
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        return f.read(limit)


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
            for i, (text, vec) in enumerate(zip(chunks, vectors)):
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
                    "id": f"{doc_id}_{i}",
                    "embedding": vec,
                    "text": text,
                    "metadata": metadata,
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

    async def index_code_dump(self, doc_id: str, text: str) -> None:
        doc = self._sqlite.get_document(doc_id)
        if doc is None:
            raise ValueError(f"document not found: {doc_id}")
        try:
            self._progress(doc_id, 10, "Parsing code dump (10%)")
            parsed = parse_repomix(text)
            if not parsed:
                raise ValueError("could not parse code dump")
            md_path = self._files.save_markdown(doc_id, text)
            self._progress(doc_id, 28, "Splitting code into chunks (28%)")
            chunk_texts: list[str] = []
            chunk_files: list[str] = []
            for file_path, content in parsed:
                pieces = await asyncio.to_thread(
                    chunk_code,
                    content,
                    self._cfg.chunk_size,
                    self._cfg.chunk_overlap,
                )
                for piece in pieces:
                    chunk_texts.append(piece)
                    chunk_files.append(file_path)
            if not chunk_texts:
                raise ValueError("no chunks produced from code dump")
            if len(chunk_texts) > self._cfg.max_chunks_per_doc:
                raise ValueError(
                    f"document exceeds max_chunks_per_doc "
                    f"({len(chunk_texts)} > {self._cfg.max_chunks_per_doc}); "
                    f"increase DOCGRAPH_MAX_CHUNKS_PER_DOC or split the source"
                )
            self._progress(
                doc_id, 42, f"Chunked into {len(chunk_texts)} parts (42%)"
            )
            vectors = await self._embed_with_progress(doc_id, chunk_texts)
            self._progress(doc_id, 96, "Saving to index (96%)")
            chroma_chunks = []
            for i, (piece, vec, file_path) in enumerate(
                zip(chunk_texts, vectors, chunk_files)
            ):
                metadata = {
                    "doc_id": doc_id,
                    "filename": doc.filename,
                    "folder": doc.folder,
                    "tags": json.dumps(doc.tags),
                    "chunk_index": i,
                    "file_path": file_path,
                    "language": infer_language(file_path) or "",
                }
                if doc.source_url:
                    metadata["source_url"] = doc.source_url
                chroma_chunks.append({
                    "id": f"{doc_id}_{i}",
                    "embedding": vec,
                    "text": piece,
                    "metadata": metadata,
                })
            self._chroma.upsert_chunks(chroma_chunks)
            self._sqlite.update_status(
                doc_id,
                DocumentStatus.READY,
                chunk_count=len(chunk_texts),
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
            prefix = await asyncio.to_thread(_read_text_prefix, original_path)
            if detect_repomix(prefix):
                full_text = await asyncio.to_thread(
                    original_path.read_text, "utf-8", "ignore"
                )
                await self.index_code_dump(doc_id, full_text)
                return
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
        self._sqlite.update_status(doc_id, DocumentStatus.PROCESSING)
        self._progress(doc_id, 0, "Starting re-index (0%)")
        if doc.source_type == SourceType.URL and doc.source_url:
            await self.index_url(doc_id, doc.source_url)
        elif doc.original_path:
            await self.index_document(doc_id, Path(doc.original_path))
        else:
            raise ValueError(f"cannot reindex: {doc_id}")
