from __future__ import annotations

from docgraph.config import Config
from docgraph.rerank.local import CrossEncoderReranker, KeywordReranker


def create_reranker(cfg: Config):
    if not getattr(cfg, "rerank_enabled", False):
        return None
    model = getattr(cfg, "rerank_model", "BAAI/bge-reranker-base")
    try:
        import sentence_transformers  # noqa: F401

        return CrossEncoderReranker(model)
    except ImportError:
        return KeywordReranker()
