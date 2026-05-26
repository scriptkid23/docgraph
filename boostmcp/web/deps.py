from __future__ import annotations

from dataclasses import dataclass

from boostmcp.config import Config
from boostmcp.embed.factory import create_embedder
from boostmcp.embed.provider import EmbeddingProvider
from boostmcp.ingest.indexer import Indexer
from boostmcp.store import ChromaStore, FileStore, SQLiteStore


@dataclass
class AppState:
    cfg: Config
    sqlite: SQLiteStore
    files: FileStore
    chroma: ChromaStore
    embedder: EmbeddingProvider

    @classmethod
    def create(cls, cfg: Config) -> "AppState":
        cfg.ensure_dirs()
        sqlite = SQLiteStore(cfg)
        sqlite.init_schema()
        return cls(
            cfg=cfg,
            sqlite=sqlite,
            files=FileStore(cfg),
            chroma=ChromaStore(cfg),
            embedder=create_embedder(cfg),
        )

    def indexer(self) -> Indexer:
        return Indexer(
            self.cfg, self.sqlite, self.files, self.chroma, self.embedder
        )
