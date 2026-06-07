from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

from docgraph.config import Config
from docgraph.embed.factory import create_embedder, create_reranker
from docgraph.embed.provider import EmbeddingProvider
from docgraph.ingest.indexer import Indexer
from docgraph.store import ChromaStore, FileStore, FtsStore, SQLiteStore

if TYPE_CHECKING:
    from docgraph.embed.rerank import Reranker
    from docgraph.mcp.search import SearchService


@dataclass
class AppState:
    cfg: Config
    sqlite: SQLiteStore
    files: FileStore
    chroma: ChromaStore
    embedder: EmbeddingProvider
    fts: Optional[FtsStore] = field(default=None)
    reranker: Optional["Reranker"] = field(default=None)

    @classmethod
    def create(cls, cfg: Config) -> "AppState":
        cfg.ensure_dirs()
        sqlite = SQLiteStore(cfg)
        sqlite.init_schema()
        fts = FtsStore(cfg) if cfg.hybrid_enabled else None
        return cls(
            cfg=cfg,
            sqlite=sqlite,
            files=FileStore(cfg),
            chroma=ChromaStore(cfg),
            embedder=create_embedder(cfg),
            fts=fts,
            reranker=create_reranker(cfg),
        )

    def indexer(self) -> Indexer:
        from docgraph.ingest.tokenizer import get_token_counter

        return Indexer(
            self.cfg,
            self.sqlite,
            self.files,
            self.chroma,
            self.embedder,
            fts=self.fts,
            counter=get_token_counter(self.cfg),
        )

    def search_service(self) -> "SearchService":
        from docgraph.mcp.search import SearchService  # deferred to avoid circular import
        return SearchService(
            self.cfg, self.sqlite, self.chroma, self.embedder,
            fts=self.fts, reranker=self.reranker,
        )
