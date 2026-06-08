from docgraph.config import Config
from docgraph.embed.local import LocalEmbedder
from docgraph.embed.ollama import OllamaEmbedder
from docgraph.embed.provider import EmbeddingProvider


def create_embedder(cfg: Config) -> EmbeddingProvider:
    if cfg.embed_provider == "local":
        return LocalEmbedder(cfg.local_model, cfg.local_model_dir)
    if cfg.embed_provider == "ollama":
        return OllamaEmbedder(cfg.ollama_url, cfg.ollama_model)
    if cfg.embed_provider == "openai":
        from docgraph.embed.openai_provider import OpenAIEmbedder
        if not cfg.openai_api_key:
            raise ValueError("OPENAI_API_KEY required when embed_provider=openai")
        return OpenAIEmbedder(cfg.openai_api_key, cfg.openai_model)
    raise ValueError(f"unknown embed provider: {cfg.embed_provider}")


def create_reranker(cfg: Config):
    """Create a Reranker or return None if disabled."""
    from docgraph.embed.rerank import Reranker
    if not cfg.rerank_enabled:
        return None
    return Reranker(model=cfg.rerank_model, cache_dir=cfg.local_model_dir)
