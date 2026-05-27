from __future__ import annotations

from dataclasses import dataclass

from docgraph.config import Config
from docgraph.embed.factory import create_embedder
from docgraph.embed.provider import EmbeddingProvider
from docgraph.ingest.indexer import Indexer
from docgraph.store import ChromaStore, FileStore, SQLiteStore


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
